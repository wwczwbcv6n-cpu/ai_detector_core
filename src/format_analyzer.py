"""
format_analyzer.py — Universal Format-Aware Forensics Analyzer
==============================================================
Extracts a fixed 128-dim feature vector from any image or video file,
capturing format-specific artifacts that are invisible to pixel-level models.

128-dim vector layout
---------------------
[0..11]   Format one-hot (jpeg, png, webp, heic, avif, raw, h264, h265, av1, vp9, vp8, other)
[12..15]  Scalar properties (bit_depth_norm, is_lossless, has_alpha, is_animated)
[16..27]  JPEG/DCT artifacts (quality, double-JPEG flag, periodicity, Q1 estimate,
          progressive, restart markers, Huffman optimized, encoder confidence ×5)
[28..35]  Chroma/color space (4:2:0, 4:2:2, 4:4:4, grayscale, BT.709, BT.2020, sRGB, HDR)
[36..47]  PNG structure (compression_norm; [37..47] reserved)
[48..63]  WebP (VP8/VP8L/VP8X flags, quality, alpha, animated, bpp; [55..63] reserved)
[64..79]  HEIC/AVIF (HEVC/AV1/MIF1 brand, film_grain, seq_profile, hevc_profile,
          bit_depth, bpp; [72..79] reserved)
[80..95]  Video codec (h264/h265/av1/vp9/vp8 flags, profile_norm, level_norm,
          bit_depth, bpp, quality tier low/medium/high/lossless; [93..95] reserved)
[96..111] Video GOP/temporal (gop_mean, gop_std, transcoding_flag, re_encoding_flag,
          n_gop_sizes, mv_entropy; [102..111] reserved)
[112..119] Metadata integrity (stripped, n_evidence, has_exif, has_gps,
           has_software, software_is_postproc, has_icc, reserved)
[120..127] Pattern restoration quality (recovery_applied, recovery_confidence,
           recovery_delta_energy, blockiness_before, blockiness_reduction,
           compression_quality_norm, prnu_noise_strength, prnu_freq_ratio)

Reuses umfre.ForensicRecoverer for all heavy parsing — no duplication.
Always returns valid 128-dim zero vector on failure — never raises.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Optional: prnu_recovery for restoration quality metrics ────────────────
try:
    from prnu_recovery import recover_prnu_signal
    _HAS_RECOVERY = True
except ImportError:
    _HAS_RECOVERY = False

# ── umfre for all format parsing ───────────────────────────────────────────
try:
    from umfre import ForensicRecoverer, MediaInfo, CompressionReport
    _HAS_UMFRE = True
except ImportError:
    _HAS_UMFRE = False
    logger.warning("umfre not available — FormatAnalyzer will return zero vectors")

# ── Postprocessing software keywords ────────────────────────────────────────
_POSTPROC_TOOLS = frozenset([
    'photoshop', 'lightroom', 'gimp', 'paint.net', 'affinity',
    'capture one', 'darktable', 'rawtherapee', 'snapseed', 'vsco',
    'facetune', 'meitu', 'picsart', 'canva', 'pixelmator',
])


# ══════════════════════════════════════════════════════════════════════════════
#  Data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FormatFeatures:
    """Result of FormatAnalyzer.analyze()."""
    feature_vector: np.ndarray          # shape (128,), float32
    extraction_ok:  bool = True
    fmt_name:       str  = "unknown"    # human-readable format name
    media_type:     str  = "unknown"    # "image" | "video"


# ══════════════════════════════════════════════════════════════════════════════
#  FormatAnalyzer
# ══════════════════════════════════════════════════════════════════════════════

class FormatAnalyzer:
    """
    Universal format-aware forensics analyzer.

    Parameters
    ----------
    recovery_net        : optional PRNURecoveryNet for restoration metrics
    enable_restoration  : compute restoration quality metrics [120..127].
                          Set False during training for speed.
    """

    FORMAT_DIM = 128

    def __init__(self, recovery_net=None, enable_restoration: bool = False):
        self.recovery_net       = recovery_net
        self.enable_restoration = enable_restoration and _HAS_RECOVERY

        if _HAS_UMFRE:
            # Lightweight ForensicRecoverer — format parsing only, no tiling
            self._recoverer = ForensicRecoverer(
                device='cpu',
                denoise_method='fast_nlm',
                max_frames=4,
                tile_size=0,
                workers=0,
                verbose=False,
            )
        else:
            self._recoverer = None

    # ------------------------------------------------------------------

    def analyze(self, path_or_bytes) -> FormatFeatures:
        """
        Analyze a file and return FormatFeatures with a 128-dim vector.

        Parameters
        ----------
        path_or_bytes : str | bytes | bytearray
            Either a file path or raw file bytes.

        Returns
        -------
        FormatFeatures — always valid; feature_vector is zeros on failure.
        """
        _zero = FormatFeatures(np.zeros(self.FORMAT_DIM, dtype=np.float32),
                               extraction_ok=False)
        tmp_path = None
        try:
            if isinstance(path_or_bytes, (bytes, bytearray)):
                suffix   = _guess_suffix(bytes(path_or_bytes[:12]))
                fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix='fmt_ana_')
                os.close(fd)
                with open(tmp_path, 'wb') as fh:
                    fh.write(path_or_bytes)
                path = tmp_path
            else:
                path = str(path_or_bytes)
                if not os.path.exists(path):
                    return _zero

            if self._recoverer is None:
                return _zero

            # Parse with umfre
            try:
                media_info = self._recoverer.ingest(path)
            except Exception as e:
                logger.debug("[FormatAnalyzer] ingest failed: %s", e)
                return _zero

            try:
                comp = self._recoverer.analyze_compression(media_info)
            except Exception as e:
                logger.debug("[FormatAnalyzer] analyze_compression failed: %s", e)
                comp = None

            vec = self._build_vector(media_info, comp, path)
            return FormatFeatures(
                feature_vector=vec,
                extraction_ok=True,
                fmt_name=media_info.format or "unknown",
                media_type=media_info.media_type,
            )

        except Exception as e:
            logger.warning("[FormatAnalyzer] analyze failed: %s", e)
            return _zero

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    #  Internal builder
    # ------------------------------------------------------------------

    def _build_vector(
        self,
        mi: 'MediaInfo',
        comp: Optional['CompressionReport'],
        path: str,
    ) -> np.ndarray:
        v   = np.zeros(self.FORMAT_DIM, dtype=np.float32)
        fmt = (mi.format or '').upper()

        # ── [0..11] Format one-hot ────────────────────────────────────
        _FMT_MAP   = {'JPEG': 0, 'JPG': 0, 'PNG': 1, 'WEBP': 2,
                      'HEIC': 3, 'HEIF': 3, 'AVIF': 4}
        _RAW_FMTS  = {'RAW', 'CR2', 'NEF', 'ARW', 'DNG', 'ORF', 'RW2', 'RAF'}
        _VID_CODEC = {'h264': 6, 'avc': 6, 'h265': 7, 'hevc': 7,
                      'av1': 8, 'vp9': 9, 'vp8': 10}

        codec = (mi.codec or '').lower()
        fmt_idx = _FMT_MAP.get(fmt)
        if fmt_idx is not None:
            v[fmt_idx] = 1.0
        elif fmt in _RAW_FMTS:
            v[5] = 1.0
        else:
            vid_idx = next((idx for k, idx in _VID_CODEC.items() if k in codec), None)
            v[vid_idx if vid_idx is not None else 11] = 1.0

        # ── [12..15] Scalar properties ───────────────────────────────
        _s(v, 12, float(mi.bit_depth or 8) / 16.0)
        is_lossless = (
            fmt == 'PNG'
            or (fmt == 'WEBP' and comp is not None and comp.webp_lossless)
            or (comp is not None and getattr(comp, 'quality_tier', None) == 'lossless')
        )
        _s(v, 13, 1.0 if is_lossless else 0.0)
        _s(v, 14, 1.0 if fmt in ('PNG', 'WEBP', 'HEIC', 'HEIF', 'AVIF') or
                         (comp is not None and getattr(comp, 'webp_has_alpha', False))
                       else 0.0)
        _s(v, 15, 1.0 if ((comp is not None and getattr(comp, 'webp_animated', False))
                           or mi.frame_count > 1) else 0.0)

        # ── [16..27] JPEG/DCT artifacts ──────────────────────────────
        if comp is not None:
            _s(v, 16, (comp.jpeg_quality_current or 0) / 100.0)
            _s(v, 17, 1.0 if comp.double_jpeg_detected else 0.0)
            _s(v, 18, float(np.clip(comp.dct_periodicity_score / 10.0, 0, 1)))
            _s(v, 19, (comp.estimated_q1 or 0) / 100.0)
            _s(v, 20, 1.0 if comp.is_progressive else 0.0)
            _s(v, 21, 1.0 if comp.has_restart_markers else 0.0)
            _s(v, 22, 1.0 if comp.huffman_optimized else 0.0)
            enc      = (comp.encoder_signature or 'unknown').lower()
            enc_conf = float(comp.encoder_confidence or 0.0)
            _s(v, 23, enc_conf if 'libjpeg'    in enc else 0.0)
            _s(v, 24, enc_conf if 'mozjpeg'    in enc else 0.0)
            _s(v, 25, enc_conf if 'photoshop'  in enc else 0.0)
            _s(v, 26, enc_conf if 'guetzli'    in enc else 0.0)
            _s(v, 27, enc_conf if enc == 'unknown' else 0.0)

        # ── [28..35] Chroma/color space ──────────────────────────────
        if comp is not None:
            cs = (comp.chroma_subsampling or '').lower()
            _s(v, 28, 1.0 if '4:2:0' in cs else 0.0)
            _s(v, 29, 1.0 if '4:2:2' in cs else 0.0)
            _s(v, 30, 1.0 if '4:4:4' in cs else 0.0)
            _s(v, 31, 1.0 if 'gray'  in cs else 0.0)
        cp = (getattr(comp, 'color_primaries',         None) or '').lower() if comp else ''
        tc = (getattr(comp, 'transfer_characteristics', None) or '').lower() if comp else ''
        cs2 = (mi.color_space or '').lower()
        _s(v, 32, 1.0 if ('bt709' in cp  or '709'  in cs2) else 0.0)
        _s(v, 33, 1.0 if ('bt2020' in cp or '2020' in cs2) else 0.0)
        _s(v, 34, 1.0 if ('srgb'  in cp  or 'srgb' in cs2) else 0.0)
        _s(v, 35, 1.0 if ('pq'    in tc  or 'hlg'  in tc or 'hdr' in cs2) else 0.0)

        # ── [36..47] PNG structure ────────────────────────────────────
        if comp is not None and comp.png_compression is not None:
            _s(v, 36, float(comp.png_compression) / 9.0)
        # [37..47] reserved

        # ── [48..63] WebP ─────────────────────────────────────────────
        if comp is not None and fmt == 'WEBP':
            wt = (getattr(comp, 'webp_type', None) or '').upper()
            _s(v, 48, 1.0 if wt == 'VP8'  else 0.0)
            _s(v, 49, 1.0 if wt == 'VP8L' else 0.0)
            _s(v, 50, 1.0 if wt == 'VP8X' else 0.0)
            _s(v, 51, (getattr(comp, 'webp_quality_estimate', None) or 0) / 100.0)
            _s(v, 52, 1.0 if getattr(comp, 'webp_has_alpha', False) else 0.0)
            _s(v, 53, 1.0 if getattr(comp, 'webp_animated', False) else 0.0)
            bpp = getattr(comp, 'bits_per_pixel', None)
            _s(v, 54, float(bpp or 0) / 24.0)
        # [55..63] reserved

        # ── [64..79] HEIC/AVIF ────────────────────────────────────────
        if comp is not None and fmt in ('HEIC', 'HEIF', 'AVIF'):
            brand = (comp.heif_brand or '').lower()
            _s(v, 64, 1.0 if ('heic' in brand or 'heix' in brand) else 0.0)
            _s(v, 65, 1.0 if 'avif' in brand else 0.0)
            _s(v, 66, 1.0 if 'mif1' in brand else 0.0)
            _s(v, 67, 1.0 if getattr(comp, 'av1_film_grain', False) else 0.0)
            _s(v, 68, (getattr(comp, 'av1_seq_profile', None) or 0) / 2.0)
            heif_bd = comp.heif_bit_depth or 8
            _s(v, 69, 0.5 if heif_bd <= 8 else (0.75 if heif_bd <= 10 else 1.0))
            _s(v, 70, float(heif_bd) / 16.0)
            bpp = getattr(comp, 'bits_per_pixel', None)
            _s(v, 71, float(bpp or 0) / 24.0)
        # [72..79] reserved

        # ── [80..95] Video codec ──────────────────────────────────────
        if comp is not None and mi.media_type == 'video':
            vc = (comp.video_codec or '').lower()
            _s(v, 80, 1.0 if ('h264' in vc or 'avc'  in vc) else 0.0)
            _s(v, 81, 1.0 if ('h265' in vc or 'hevc' in vc) else 0.0)
            _s(v, 82, 1.0 if 'av1' in vc  else 0.0)
            _s(v, 83, 1.0 if 'vp9' in vc  else 0.0)
            _s(v, 84, 1.0 if 'vp8' in vc  else 0.0)
            vp = (comp.video_profile or '').lower()
            prof = 0.25 if 'base' in vp else (0.5 if 'main' in vp else
                   (0.75 if 'high' in vp else 0.5))
            _s(v, 85, prof)
            try:
                vl_str = (comp.video_level or '0').replace(',', '.')
                _s(v, 86, float(vl_str) / 6.2)
            except Exception:
                pass
            _s(v, 87, float(comp.video_bit_depth or 8) / 16.0)
            bpp = comp.bits_per_pixel
            _s(v, 88, float(bpp or 0) / 20.0)
            qt = (comp.quality_tier or '').lower()
            _s(v, 89, 1.0 if 'low'      in qt else 0.0)
            _s(v, 90, 1.0 if 'medium'   in qt else 0.0)
            _s(v, 91, 1.0 if 'high'     in qt else 0.0)
            _s(v, 92, 1.0 if 'lossless' in qt else 0.0)
        # [93..95] reserved

        # ── [96..111] Video GOP/temporal ─────────────────────────────
        if comp is not None and mi.media_type == 'video':
            _s(v, 96,  float(comp.gop_mean or 0) / 100.0)
            _s(v, 97,  float(comp.gop_std  or 0) / 100.0)
            _s(v, 98,  1.0 if comp.transcoding_detected else 0.0)
            _s(v, 99,  1.0 if comp.re_encoding_detected else 0.0)
            _s(v, 100, float(len(comp.gop_sizes or [])) / 50.0)
            _s(v, 101, float(np.clip((comp.mv_entropy or 0) / 10.0, 0, 1)))
        # [102..111] reserved

        # ── [112..119] Metadata integrity ────────────────────────────
        _s(v, 112, 1.0 if mi.metadata_stripped else 0.0)
        _s(v, 113, float(np.clip(len(mi.stripping_evidence or []) / 10.0, 0, 1)))
        _s(v, 114, 1.0 if mi.has_exif else 0.0)
        has_gps = any('gps' in str(k).lower() for k in (mi.exif_fields or {}).keys())
        _s(v, 115, 1.0 if has_gps else 0.0)
        sw = str(
            (mi.exif_fields or {}).get('Software',
            (mi.exif_fields or {}).get('software', '')) or ''
        ).lower()
        _s(v, 116, 1.0 if sw else 0.0)
        _s(v, 117, 1.0 if any(t in sw for t in _POSTPROC_TOOLS) else 0.0)
        has_icc = False
        try:
            from PIL import Image as _PIL
            with _PIL.open(path) as _pil:
                has_icc = bool(_pil.info.get('icc_profile'))
        except Exception:
            pass
        _s(v, 118, 1.0 if has_icc else 0.0)
        # [119] reserved

        # ── [120..127] Pattern restoration quality ────────────────────
        if self.enable_restoration:
            _fill_restoration(v, path, self.recovery_net, 120)
        # else: zeros (fast training path)

        return v


# ══════════════════════════════════════════════════════════════════════════════
#  Restoration quality sub-section [120..127]
# ══════════════════════════════════════════════════════════════════════════════

def _fill_restoration(v: np.ndarray, path: str, recovery_net, start: int) -> None:
    """Fill [start..start+7] with pattern restoration quality metrics."""
    try:
        from PIL import Image as _PIL
        import numpy as _np
        _pil_bilinear = getattr(_PIL, 'Resampling', _PIL).BILINEAR  # Pillow 10+ compat

        pil = _PIL.open(path).convert('RGB')
        arr = _np.array(
            pil.resize((256, 256), _pil_bilinear), dtype=np.float32
        ) / 255.0

        def _blockiness(a: _np.ndarray) -> float:
            try:
                gray = a.mean(axis=2) if a.ndim == 3 else a
                h, w = gray.shape
                bv = sum(
                    float(_np.abs(gray[r:r+1, :] - gray[r-1:r, :]).mean())
                    for r in range(8, h, 8)
                ) + sum(
                    float(_np.abs(gray[:, c:c+1] - gray[:, c-1:c]).mean())
                    for c in range(8, w, 8)
                )
                n = (h // 8) + (w // 8)
                return bv / max(n, 1)
            except Exception:
                return 0.0

        block_before = _blockiness(arr)
        _s(v, start + 3, float(_np.clip(block_before * 20.0, 0, 1)))
        _s(v, start + 5, float(_np.clip(1.0 - block_before * 10.0, 0, 1)))

        if recovery_net is not None and _HAS_RECOVERY:
            try:
                restored     = recover_prnu_signal(arr, recovery_net, device=None)
                block_after  = _blockiness(restored)
                delta_energy = float(_np.mean(_np.abs(restored - arr)))
                reduction    = float(_np.clip(
                    (block_before - block_after) / (block_before + 1e-8), 0, 1))

                _s(v, start + 0, 1.0)
                noise_strength = float(_np.std(restored - arr))
                _s(v, start + 1, float(_np.clip(noise_strength * 20.0, 0, 1)))
                _s(v, start + 2, float(_np.clip(delta_energy * 10.0, 0, 1)))
                _s(v, start + 4, float(_np.clip(reduction, 0, 1)))

                noise_arr    = restored - arr
                prnu_strength = float(_np.std(noise_arr))
                _s(v, start + 6, float(_np.clip(prnu_strength * 20.0, 0, 1)))

                try:
                    from scipy.fft import fft2, fftshift as _fftshift
                    gray_noise = noise_arr.mean(axis=2) if noise_arr.ndim == 3 else noise_arr
                    fft_mag    = _np.abs(_fftshift(fft2(gray_noise)))
                    h2, w2     = fft_mag.shape
                    cy, cx     = h2 // 2, w2 // 2
                    r_hf       = int(min(h2, w2) * 0.4)
                    Y, X       = _np.ogrid[:h2, :w2]
                    hf_mask    = (Y - cy)**2 + (X - cx)**2 > r_hf**2
                    total_e    = float(fft_mag.sum()) + 1e-8
                    _s(v, start + 7, float(fft_mag[hf_mask].sum()) / total_e)
                except Exception:
                    pass
            except Exception as e:
                logger.debug("[FormatAnalyzer._fill_restoration] recovery failed: %s", e)

    except Exception as e:
        logger.debug("[FormatAnalyzer._fill_restoration] failed: %s", e)


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _s(v: np.ndarray, idx: int, val: float) -> None:
    """Clip-assign a scalar to v[idx].  NaN / Inf → 0."""
    try:
        fval = float(val)
        v[idx] = 0.0 if (fval != fval or abs(fval) == float('inf')) \
                 else float(np.clip(fval, 0.0, 1.0))
    except Exception:
        v[idx] = 0.0


def _guess_suffix(data: bytes) -> str:
    """Guess a file suffix from the first 12 magic bytes."""
    if len(data) >= 2 and data[:2] == b'\xff\xd8':
        return '.jpg'
    if len(data) >= 8 and data[:8] == b'\x89PNG\r\n\x1a\n':
        return '.png'
    if len(data) >= 12 and data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return '.webp'
    if len(data) >= 12 and data[4:8] == b'ftyp':
        brand = data[8:12]
        if brand in (b'heic', b'heix', b'mif1'):
            return '.heic'
        if brand == b'avif':
            return '.avif'
        return '.mp4'
    if len(data) >= 3 and data[:3] == b'GIF':
        return '.gif'
    if len(data) >= 4 and data[:4] in (b'MM\x00*', b'II*\x00'):
        return '.tif'
    return '.bin'


# ══════════════════════════════════════════════════════════════════════════════
#  Singleton accessor
# ══════════════════════════════════════════════════════════════════════════════

_shared_analyzer: Optional[FormatAnalyzer] = None


def get_shared_analyzer(
    recovery_net=None,
    enable_restoration: bool = False,
) -> FormatAnalyzer:
    """
    Return a module-level singleton FormatAnalyzer.
    First call creates the instance; subsequent calls return the same object.
    Safe for DataLoader workers (create in main process, not forked workers).
    """
    global _shared_analyzer
    if _shared_analyzer is None:
        _shared_analyzer = FormatAnalyzer(
            recovery_net=recovery_net,
            enable_restoration=enable_restoration,
        )
    return _shared_analyzer
