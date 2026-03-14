"""
training_dashboard.py — Real-Time Visual Training Dashboard
============================================================

Saves a multi-panel PNG every few seconds while training runs.
Open models/training_dashboard.png in any image viewer that
auto-refreshes (eog, feh, xviewer, Gwenview) to watch live.

Layout  (3 rows × 3 cols):
┌──────────────┬──────────────┬──────────────────────────┐
│  Loss Curve  │  Acc  Curve  │     GPU / Speed Stats    │
├──────────────┼──────────────┼──────────────────────────┤
│ Video Frame  │  PRNU Noise  │    FFT Spectrum (log)    │
│  (RGB tile)  │     Map      │   GAN peaks visible here │
├──────────────┼──────────────┼──────────────────────────┤
│  PRNU 64-dim │  Confidence  │   PRNU Explanation text  │
│  feature bar │  histogram   │                          │
└──────────────┴──────────────┴──────────────────────────┘

Usage (called from train_deep.py):

    from training_dashboard import TrainingDashboard

    dash = TrainingDashboard()
    # inside training loop:
    dash.update_metrics(step, loss, acc)
    dash.update_frame(img_cpu, prnu_feats_np, confidences_np, labels_np)
    # end of training:
    dash.close()
"""

import os
import time

import numpy as np

_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR    = os.path.join(_SCRIPT_DIR, '..', 'models')
DEFAULT_PATH   = os.path.join(_MODELS_DIR, 'training_dashboard.png')

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# SRM high-pass kernel — same as model's SRMBranch filter 0
_HP_KERNEL = np.array([
    [ 0,  0, -1,  0,  0],
    [ 0, -1,  2, -1,  0],
    [-1,  2,  0,  2, -1],
    [ 0, -1,  2, -1,  0],
    [ 0,  0, -1,  0,  0],
], dtype=np.float32) / 12.0

# PRNU feature group names (64-dim, matching prnu_features.py v4)
_PRNU_GROUPS = [
    ('Mean\n(0-7)',   slice(0,  8),  '#3498db'),
    ('Std\n(8-14)',   slice(8,  15), '#2ecc71'),
    ('Tile\nCorr',    slice(15, 16), '#e74c3c'),
    ('Freq\nBands',   slice(16, 20), '#9b59b6'),
    ('Chan\nCorr',    slice(20, 24), '#f39c12'),
    ('Recov\nΔ',      slice(24, 28), '#1abc9c'),
    ('DblCmp',        slice(28, 31), '#e67e22'),
    ('Conf',          slice(31, 32), '#e74c3c'),
    ('ExtFreq',       slice(32, 36), '#8e44ad'),
    ('Chan\nRMS',     slice(36, 40), '#27ae60'),
    ('Aniso\ntropy',  slice(40, 44), '#d35400'),
    ('Bayer\nCFA',    slice(44, 48), '#2980b9'),
    ('Phase\nCoh',    slice(48, 52), '#16a085'),
    ('Dir\nCorr',     slice(52, 56), '#c0392b'),
    ('Satur\nation',  slice(56, 60), '#7f8c8d'),
    ('Scale\nCons',   slice(60, 64), '#f1c40f'),
]

_PRNU_EXPLANATION = (
    "PRNU — Photo-Response Non-Uniformity\n"
    "─────────────────────────────────────────────────────\n"
    "Every real camera sensor has tiny manufacturing\n"
    "imperfections: pixel-level gain variations that stay\n"
    "constant across all photos — like a fingerprint.\n\n"
    "The model extracts this 64-dim PRNU vector:\n"
    "  [0-7]   Reliability-weighted mean (per tile)\n"
    "  [8-14]  Reliability-weighted std deviation\n"
    "  [15]    Inter-tile correlation\n"
    "  [16-19] Energy in 4 freq bands (LF→VHF)\n"
    "  [20-23] Cross-channel correlation (R-G, R-B, G-B)\n"
    "  [24-27] Recovery delta stats\n"
    "  [28-30] Double-compression signature\n"
    "  [31]    Recovery confidence\n"
    "  [32-63] Phase, anisotropy, Bayer CFA, saturation…\n\n"
    "AI generators produce NONE of this.  The noise map\n"
    "(centre panel, row 2) shows the raw residual that\n"
    "the model reads — flat for AI, textured for cameras."
)


class TrainingDashboard:
    """
    File-based live training dashboard.

    PNG is overwritten every SAVE_INTERVAL seconds so any
    auto-refresh image viewer shows the current state.
    """

    SAVE_INTERVAL = 4.0   # seconds between PNG saves

    def __init__(self, title: str = 'AI Detector — Video Training',
                 save_path: str = DEFAULT_PATH):
        self._ok        = False
        self._path      = save_path
        self._title     = title
        self._start     = time.time()
        self._last_save = 0.0

        # Metric history
        self.steps      = []
        self.losses     = []
        self.accs       = []

        # Latest visual data
        self._frame_rgb  = None   # (H,W,3) float32 [0,1]
        self._prnu_map   = None   # (H,W)   float32 noise residual
        self._fft_map    = None   # (H,W)   float32 log-magnitude
        self._prnu_feats = None   # (64,)   float32
        self._confs      = None   # (N,)    float32 batch confidences
        self._labels     = None   # (N,)    float32 0/1

        # Speed tracking
        self._step_times = []
        self._last_step  = time.time()

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            from matplotlib.colors import LinearSegmentedColormap

            self._plt = plt
            self._gs_mod = gridspec

            self.fig = plt.figure(figsize=(20, 13), facecolor='#1a1a2e')
            gs = gridspec.GridSpec(3, 3, figure=self.fig,
                                   hspace=0.42, wspace=0.35,
                                   left=0.06, right=0.97,
                                   top=0.93, bottom=0.05)

            def _ax(r, c, **kw):
                a = self.fig.add_subplot(gs[r, c], **kw)
                a.set_facecolor('#16213e')
                for sp in a.spines.values():
                    sp.set_color('#0f3460')
                a.tick_params(colors='#a0a0c0', labelsize=8)
                a.xaxis.label.set_color('#a0a0c0')
                a.yaxis.label.set_color('#a0a0c0')
                a.title.set_color('#e0e0ff')
                return a

            self._ax_loss   = _ax(0, 0)
            self._ax_acc    = _ax(0, 1)
            self._ax_gpu    = _ax(0, 2)
            self._ax_frame  = _ax(1, 0)
            self._ax_prnu   = _ax(1, 1)
            self._ax_fft    = _ax(1, 2)
            self._ax_bar    = _ax(2, 0)
            self._ax_hist   = _ax(2, 1)
            self._ax_expl   = _ax(2, 2)

            self.fig.suptitle(title, fontsize=14, fontweight='bold',
                              color='#e0e0ff', y=0.97)

            os.makedirs(os.path.dirname(os.path.abspath(self._path)),
                        exist_ok=True)
            self._ok = True
            print(f'\n  Dashboard → {self._path}')
            print('  Open that file in an image viewer with auto-refresh.')
            print('  (eog / feh / Gwenview — all support auto-refresh)\n')

        except Exception as exc:
            print(f'  Dashboard unavailable: {exc}  — training continues.')

    # ── Public API ────────────────────────────────────────────────────────

    def update_metrics(self, step: int, loss: float, acc: float):
        """Call every N optimizer steps with current loss and accuracy."""
        now = time.time()
        self._step_times.append(now - self._last_step)
        self._last_step = now
        if len(self._step_times) > 50:
            self._step_times.pop(0)

        self.steps.append(step)
        self.losses.append(loss)
        self.accs.append(acc * 100.0)

        if self._ok and (now - self._last_save) >= self.SAVE_INTERVAL:
            try:
                self._draw()
                self._last_save = now
            except Exception:
                pass

    def update_frame(self, img_tensor, prnu_feats_np: np.ndarray,
                     confidences_np: np.ndarray, labels_np: np.ndarray):
        """
        Call with one sample from the current batch (CPU tensors / numpy).

        img_tensor    : (3, H, W) ImageNet-normalised float32 torch.Tensor
        prnu_feats_np : (64,) float32 numpy array
        confidences_np: (N,)  float32 sigmoid confidences for the batch
        labels_np     : (N,)  float32 ground-truth labels (0=real, 1=AI)
        """
        try:
            img_np = self._denorm(img_tensor)   # (H,W,3) float32 [0,1]
            self._frame_rgb  = img_np
            self._prnu_map   = self._prnu_residual(img_np)
            self._fft_map    = self._fft_spectrum(img_np)
            self._prnu_feats = prnu_feats_np.astype(np.float32)
            self._confs      = confidences_np.astype(np.float32)
            self._labels     = labels_np.astype(np.float32)
        except Exception:
            pass

    def close(self):
        if self._ok:
            try:
                self._draw()
                print(f'  Final dashboard saved → {self._path}')
            except Exception:
                pass
            try:
                self._plt.close(self.fig)
            except Exception:
                pass

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _denorm(tensor) -> np.ndarray:
        """ImageNet-normalised (3,H,W) tensor → (H,W,3) float32 [0,1]."""
        try:
            arr = tensor.numpy() if hasattr(tensor, 'numpy') else np.array(tensor)
        except Exception:
            arr = np.array(tensor)
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)
        arr = arr * _IMAGENET_STD + _IMAGENET_MEAN
        return np.clip(arr, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _prnu_residual(img_np: np.ndarray) -> np.ndarray:
        """
        Apply SRM high-pass filter to expose PRNU / sensor noise residual.
        Returns (H,W) float32 in [0,1] — brighter = stronger noise texture.
        """
        try:
            from scipy.ndimage import convolve
            gray = img_np.mean(axis=2)
            residual = convolve(gray, _HP_KERNEL, mode='reflect')
            residual = np.abs(residual)
            lo, hi = np.percentile(residual, 1), np.percentile(residual, 99)
            return np.clip((residual - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)
        except Exception:
            return np.zeros(img_np.shape[:2], dtype=np.float32)

    @staticmethod
    def _fft_spectrum(img_np: np.ndarray) -> np.ndarray:
        """
        Log-magnitude FFT spectrum of grayscale image.
        Returns (H,W) float32 in [0,1] — bright spots = GAN periodic peaks.
        """
        try:
            gray = img_np.mean(axis=2)
            sz = min(gray.shape[0], gray.shape[1], 256)
            gray = gray[:sz, :sz]
            fft = np.fft.fftshift(np.fft.fft2(gray))
            mag = np.log1p(np.abs(fft))
            lo, hi = mag.min(), mag.max()
            return ((mag - lo) / (hi - lo + 1e-8)).astype(np.float32)
        except Exception:
            return np.zeros((128, 128), dtype=np.float32)

    def _gpu_text(self) -> str:
        lines = []
        try:
            import torch
            if torch.cuda.is_available():
                alloc   = torch.cuda.memory_allocated(0) / 1024**3
                reservd = torch.cuda.memory_reserved(0) / 1024**3
                total   = torch.cuda.get_device_properties(0).total_memory / 1024**3
                name    = torch.cuda.get_device_properties(0).name
                pct     = alloc / total * 100
                lines += [
                    f'GPU     {name}',
                    f'VRAM    {alloc:.2f} / {total:.1f} GB  ({pct:.0f}%)',
                    f'Reservd {reservd:.2f} GB',
                    f'Free    {total - reservd:.2f} GB',
                ]
        except Exception:
            lines.append('GPU  CPU only')

        elapsed = time.time() - self._start
        h, m = divmod(int(elapsed), 3600)
        m, s = divmod(m, 60)
        lines.append(f'\nElapsed  {h:02d}:{m:02d}:{s:02d}')

        if self._step_times:
            avg_step = np.mean(self._step_times)
            lines.append(f'Step/s   {1/avg_step:.2f}')

        if self.losses:
            lines.append(f'\nLoss     {self.losses[-1]:.4f}')
            lines.append(f'Acc      {self.accs[-1]:.1f}%')
            lines.append(f'Steps    {self.steps[-1]}')

        if self._confs is not None and self._labels is not None:
            n_real = (self._labels == 0).sum()
            n_ai   = (self._labels == 1).sum()
            lines.append(f'\nBatch    real={n_real}  AI={n_ai}')

        return '\n'.join(lines)

    # ── Draw ──────────────────────────────────────────────────────────────

    def _draw(self):
        dark_text = '#e0e0ff'
        green  = '#2ecc71'
        red    = '#e74c3c'
        blue   = '#3498db'
        orange = '#f39c12'

        # ── Row 0: Loss ──────────────────────────────────────────────────
        ax = self._ax_loss
        ax.cla(); ax.set_facecolor('#16213e')
        if self.steps:
            ax.plot(self.steps, self.losses, color=blue, lw=1.8, label='Loss')
            ax.set(title='Training Loss', xlabel='Step', ylabel='BCE')
            ax.grid(alpha=0.2, color='#0f3460')
            ax.legend(fontsize=8, labelcolor=dark_text,
                      facecolor='#1a1a2e', edgecolor='#0f3460')
        for sp in ax.spines.values(): sp.set_color('#0f3460')
        ax.tick_params(colors='#a0a0c0'); ax.title.set_color(dark_text)
        ax.xaxis.label.set_color('#a0a0c0'); ax.yaxis.label.set_color('#a0a0c0')

        # ── Row 0: Accuracy ──────────────────────────────────────────────
        ax = self._ax_acc
        ax.cla(); ax.set_facecolor('#16213e')
        if self.steps:
            ax.plot(self.steps, self.accs, color=green, lw=1.8, label='Acc %')
            ax.axhline(50, color='#555', ls='--', lw=1, alpha=0.6)
            ax.set_ylim(0, 100)
            ax.set(title='Accuracy', xlabel='Step', ylabel='%')
            ax.grid(alpha=0.2, color='#0f3460')
            ax.legend(fontsize=8, labelcolor=dark_text,
                      facecolor='#1a1a2e', edgecolor='#0f3460')
        for sp in ax.spines.values(): sp.set_color('#0f3460')
        ax.tick_params(colors='#a0a0c0'); ax.title.set_color(dark_text)
        ax.xaxis.label.set_color('#a0a0c0'); ax.yaxis.label.set_color('#a0a0c0')

        # ── Row 0: GPU Stats ─────────────────────────────────────────────
        ax = self._ax_gpu
        ax.cla(); ax.set_facecolor('#16213e'); ax.axis('off')
        ax.text(0.05, 0.95, self._gpu_text(),
                transform=ax.transAxes, va='top', ha='left',
                fontfamily='monospace', fontsize=9, color=dark_text,
                bbox=dict(boxstyle='round,pad=0.6', facecolor='#0f3460',
                          edgecolor='#3498db', alpha=0.9))
        ax.set_title('GPU & Speed', color=dark_text, fontsize=10)

        # ── Row 1: Video Frame ────────────────────────────────────────────
        ax = self._ax_frame
        ax.cla(); ax.set_facecolor('#16213e'); ax.axis('off')
        if self._frame_rgb is not None:
            ax.imshow(self._frame_rgb, aspect='auto')
            ax.set_title('Current Frame (from YouTube)', color=dark_text, fontsize=10)
        else:
            ax.text(0.5, 0.5, 'Waiting for\nfirst frame…',
                    ha='center', va='center', color='#888',
                    fontsize=11, transform=ax.transAxes)
            ax.set_title('Current Frame', color=dark_text, fontsize=10)

        # ── Row 1: PRNU Noise Map ─────────────────────────────────────────
        ax = self._ax_prnu
        ax.cla(); ax.set_facecolor('#16213e'); ax.axis('off')
        if self._prnu_map is not None:
            im = ax.imshow(self._prnu_map, cmap='inferno', aspect='auto',
                           vmin=0, vmax=1)
            self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                              label='Residual intensity')
            ax.set_title('PRNU Noise Map  (SRM residual)', color=dark_text, fontsize=10)
            # Annotate: real vs AI expectation
            noise_mean = float(self._prnu_map.mean())
            label_str  = f'Mean residual: {noise_mean:.4f}'
            texture    = 'Rich texture → likely REAL camera' if noise_mean > 0.08 \
                         else 'Flat residual → possible AI frame'
            ax.set_xlabel(f'{label_str}\n{texture}', color='#a0a0c0', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No frame yet', ha='center', va='center',
                    color='#888', fontsize=11, transform=ax.transAxes)
            ax.set_title('PRNU Noise Map', color=dark_text, fontsize=10)

        # ── Row 1: FFT Spectrum ───────────────────────────────────────────
        ax = self._ax_fft
        ax.cla(); ax.set_facecolor('#16213e'); ax.axis('off')
        if self._fft_map is not None:
            ax.imshow(self._fft_map, cmap='plasma', aspect='auto')
            ax.set_title('FFT Spectrum  (GAN peaks visible here)', color=dark_text, fontsize=10)
            ax.set_xlabel(
                'Centre = DC   |   Edges = high freq\n'
                'Bright spots off-centre → GAN periodic artifact',
                color='#a0a0c0', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No frame yet', ha='center', va='center',
                    color='#888', fontsize=11, transform=ax.transAxes)
            ax.set_title('FFT Spectrum', color=dark_text, fontsize=10)

        # ── Row 2: PRNU 64-dim bar ────────────────────────────────────────
        ax = self._ax_bar
        ax.cla(); ax.set_facecolor('#16213e')
        if self._prnu_feats is not None and len(self._prnu_feats) == 64:
            feats = self._prnu_feats
            x_pos, bar_vals, bar_cols, x_ticks, x_labels = [], [], [], [], []
            cursor = 0
            for name, sl, color in _PRNU_GROUPS:
                vals = feats[sl]
                for v in vals:
                    bar_vals.append(float(v))
                    bar_cols.append(color)
                    x_pos.append(cursor)
                    cursor += 1
                x_ticks.append(cursor - len(vals) / 2 - 0.5)
                x_labels.append(name)

            ax.bar(x_pos, bar_vals, color=bar_cols, width=0.85, alpha=0.85)
            ax.axhline(0, color='#555', lw=0.8)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels, fontsize=6, rotation=0, color='#a0a0c0')
            ax.set_title('PRNU 64-dim Feature Vector', color=dark_text, fontsize=10)
            ax.set_ylabel('Feature value', color='#a0a0c0', fontsize=8)
            ax.grid(axis='y', alpha=0.2, color='#0f3460')
        else:
            ax.text(0.5, 0.5, 'No PRNU features yet', ha='center', va='center',
                    color='#888', fontsize=11, transform=ax.transAxes)
            ax.set_title('PRNU 64-dim Feature Vector', color=dark_text, fontsize=10)
        for sp in ax.spines.values(): sp.set_color('#0f3460')
        ax.tick_params(colors='#a0a0c0')

        # ── Row 2: Confidence Histogram ───────────────────────────────────
        ax = self._ax_hist
        ax.cla(); ax.set_facecolor('#16213e')
        if self._confs is not None and self._labels is not None:
            bins = np.linspace(0, 1, 21)
            real_mask = self._labels == 0
            ai_mask   = self._labels == 1
            if real_mask.any():
                ax.hist(self._confs[real_mask], bins=bins, color=green,
                        alpha=0.7, label='Real (GT=0)', density=True)
            if ai_mask.any():
                ax.hist(self._confs[ai_mask],   bins=bins, color=red,
                        alpha=0.7, label='AI (GT=1)',   density=True)
            ax.axvline(0.5, color='white', ls='--', lw=1.2, alpha=0.7)
            ax.set(title='Batch Confidence', xlabel='P(AI)', ylabel='Density')
            ax.set_xlim(0, 1)
            ax.legend(fontsize=8, labelcolor=dark_text,
                      facecolor='#1a1a2e', edgecolor='#0f3460')
            ax.grid(alpha=0.2, color='#0f3460')
        else:
            ax.text(0.5, 0.5, 'No predictions yet', ha='center', va='center',
                    color='#888', fontsize=11, transform=ax.transAxes)
            ax.set_title('Batch Confidence', color=dark_text, fontsize=10)
        for sp in ax.spines.values(): sp.set_color('#0f3460')
        ax.tick_params(colors='#a0a0c0')
        ax.title.set_color(dark_text)
        ax.xaxis.label.set_color('#a0a0c0'); ax.yaxis.label.set_color('#a0a0c0')

        # ── Row 2: PRNU Explanation ───────────────────────────────────────
        ax = self._ax_expl
        ax.cla(); ax.set_facecolor('#16213e'); ax.axis('off')
        ax.text(0.04, 0.97, _PRNU_EXPLANATION,
                transform=ax.transAxes, va='top', ha='left',
                fontfamily='monospace', fontsize=7.5, color='#c0d0ff',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#0f3460',
                          edgecolor='#3498db', alpha=0.85),
                linespacing=1.4)
        ax.set_title('What is PRNU?', color=dark_text, fontsize=10)

        # ── Save ──────────────────────────────────────────────────────────
        self.fig.savefig(self._path, dpi=100, bbox_inches='tight',
                         facecolor=self.fig.get_facecolor())
