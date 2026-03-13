"""
image_loader.py — Universal Image Loader

Single entry point for loading any image format:
  - HEIC/HEIF/AVIF : pillow_heif (replaces unmaintained pyheif)
  - RAW (CR2, NEF, ARW, DNG, RAF, RW2, ORF): rawpy (optional — graceful skip)
  - GIF             : first frame
  - TIFF multi-page : seek(0)
  - All others      : PIL fallback → imageio last resort

Max dimension guard: if any dimension > 4096 px, resize preserving aspect ratio.
"""

import io
import os
import numpy as np
from PIL import Image

# ── Optional dependencies ────────────────────────────────────────────────────

try:
    import pillow_heif
    pillow_heif.register_heif_opener()   # makes PIL.Image.open() handle HEIC/HEIF/AVIF
    _HEIF_SUPPORT = True
except ImportError:
    _HEIF_SUPPORT = False

try:
    import rawpy
    _RAWPY_SUPPORT = True
except ImportError:
    _RAWPY_SUPPORT = False

try:
    import imageio
    _IMAGEIO_SUPPORT = True
except ImportError:
    _IMAGEIO_SUPPORT = False

_RAW_EXTS = {'.cr2', '.nef', '.arw', '.dng', '.raf', '.rw2', '.orf', '.cr3'}
_MAX_DIM  = 4096


class UniversalImageLoader:
    """
    Loads images from any format into a PIL Image (RGB) or NumPy array.

    Usage:
        loader = UniversalImageLoader()
        img    = loader.load('/path/to/photo.HEIC')
        arr    = loader.load_for_prnu('/path/to/photo.NEF')
        resized = loader.load_for_model('/path/to/img.jpg', size=(512, 512))
    """

    # ──────────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────────

    def load(self, path_or_bytes, hint: str = None) -> Image.Image:
        """
        Load an image from a file path or raw bytes → PIL Image (RGB).

        Args:
            path_or_bytes : str file path or raw bytes
            hint          : optional format hint (e.g. 'heic', 'cr2', 'nef')

        Returns:
            PIL Image in RGB mode (max dimension capped at 4096 px)
        """
        ext = self._ext(path_or_bytes, hint)

        if ext in _RAW_EXTS:
            img = self._load_raw_pil(path_or_bytes)
            if img is not None:
                return self._guard_size(img)

        # PIL first (handles HEIC/HEIF via pillow_heif opener, GIF, TIFF, …)
        try:
            return self._guard_size(self._load_pil(path_or_bytes))
        except Exception:
            pass

        # imageio last resort
        if _IMAGEIO_SUPPORT:
            try:
                return self._guard_size(self._load_imageio(path_or_bytes))
            except Exception:
                pass

        raise RuntimeError(
            f"Could not load image from {type(path_or_bytes).__name__}"
            + (f" (ext={ext})" if ext else "")
        )

    def load_for_prnu(self, path_or_bytes, hint: str = None) -> np.ndarray:
        """
        Load at full native resolution as float64 ndarray [0, 1] (H, W, 3).

        No resize is applied before the 4096-px guard — PRNU extraction needs
        all pixels.  RAW images use camera white balance with no auto-brightness
        to preserve sensor noise.

        Returns:
            np.ndarray  shape (H, W, 3), dtype float64, values in [0, 1]
        """
        ext = self._ext(path_or_bytes, hint)

        if ext in _RAW_EXTS:
            arr = self._load_raw_array(path_or_bytes)
            if arr is not None:
                return self._guard_array(arr)

        img = self.load(path_or_bytes, hint=hint)
        return np.array(img, dtype=np.float64) / 255.0

    def load_for_model(self, path_or_bytes, size=(512, 512),
                       hint: str = None) -> Image.Image:
        """
        Load and resize image for model input.

        Args:
            path_or_bytes : str file path or raw bytes
            size          : (width, height) tuple
            hint          : optional format hint

        Returns:
            PIL Image in RGB mode, resized to `size`
        """
        img = self.load(path_or_bytes, hint=hint)
        return img.resize(size, Image.BILINEAR)

    # ──────────────────────────────────────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _ext(path_or_bytes, hint: str) -> str:
        if hint:
            return '.' + hint.lower().lstrip('.')
        if isinstance(path_or_bytes, str):
            return os.path.splitext(path_or_bytes)[1].lower()
        return ''

    def _load_pil(self, path_or_bytes) -> Image.Image:
        """Load via PIL (handles HEIC via pillow_heif opener, GIF, TIFF, …)"""
        if isinstance(path_or_bytes, bytes):
            img = Image.open(io.BytesIO(path_or_bytes))
        else:
            img = Image.open(path_or_bytes)
        # Multi-frame: seek to first frame
        if hasattr(img, 'n_frames') and img.n_frames > 1:
            img.seek(0)
        return img.convert('RGB')

    def _load_raw_pil(self, path_or_bytes) -> Image.Image | None:
        """Load RAW image via rawpy → PIL Image, or None if unavailable."""
        arr = self._load_raw_array(path_or_bytes)
        if arr is None:
            return None
        return Image.fromarray((arr * 255).astype(np.uint8))

    def _load_raw_array(self, path_or_bytes) -> np.ndarray | None:
        """
        Load RAW image via rawpy → float64 [0, 1] array (H, W, 3).
        Returns None if rawpy is unavailable or loading fails.
        Uses camera white balance + no auto-brightness for PRNU accuracy.
        """
        if not _RAWPY_SUPPORT:
            return None
        try:
            if isinstance(path_or_bytes, bytes):
                raw = rawpy.imread(io.BytesIO(path_or_bytes))
            else:
                raw = rawpy.imread(str(path_or_bytes))
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                output_bps=16,
            )
            return rgb.astype(np.float64) / 65535.0
        except Exception:
            return None

    def _load_imageio(self, path_or_bytes) -> Image.Image:
        """Load via imageio as a last resort."""
        if isinstance(path_or_bytes, bytes):
            arr = imageio.imread(io.BytesIO(path_or_bytes))
        else:
            arr = imageio.imread(str(path_or_bytes))
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return Image.fromarray(arr.astype(np.uint8))

    @staticmethod
    def _guard_size(img: Image.Image) -> Image.Image:
        """Resize if either dimension > _MAX_DIM, preserving aspect ratio."""
        w, h = img.size
        if max(w, h) > _MAX_DIM:
            scale = _MAX_DIM / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        return img

    @staticmethod
    def _guard_array(arr: np.ndarray) -> np.ndarray:
        """Resize array if either dimension > _MAX_DIM, preserving aspect ratio."""
        h, w = arr.shape[:2]
        if max(h, w) > _MAX_DIM:
            scale = _MAX_DIM / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            pil = Image.fromarray((arr * 255).astype(np.uint8))
            pil = pil.resize((new_w, new_h), Image.BILINEAR)
            return np.array(pil, dtype=np.float64) / 255.0
        return arr
