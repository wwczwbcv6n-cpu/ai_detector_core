"""
live_plot.py — Live training visualization for local GPU training.

Saves the plot as a PNG file after every update — no interactive window,
no "not responding" dialogs from GTK/Qt. The training loop is never
blocked waiting for a GUI event loop.

Open the PNG in any image viewer; many viewers (eog, feh, xviewer) can
auto-refresh when the file changes.

  Plot file: <project_root>/models/training_progress.png

Usage:
    from live_plot import LivePlot
    plot = LivePlot(title='Image Training', xlabel='Batch')
    ...
    plot.update(step, loss, acc)
    plot.update(step, loss, acc, val_loss, val_acc)
    plot.close()
"""

import os
import time


# Save next to the models directory so it's easy to find
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR  = os.path.join(_SCRIPT_DIR, '..', 'models')
DEFAULT_PATH = os.path.join(_MODELS_DIR, 'training_progress.png')


class LivePlot:
    """
    File-based live plot: draws loss + accuracy + stats, saves as PNG.

    No GUI window is opened — eliminates the GTK/Qt "not responding"
    dialog that appears when the training loop starves the event loop.

    The PNG is overwritten after every update so the user can keep an
    image viewer open alongside the terminal.
    """

    # Minimum seconds between consecutive PNG saves (avoids thrashing
    # disk on very fast loops while keeping the image fresh).
    _MIN_INTERVAL = 2.0

    def __init__(self, title='Training Progress', xlabel='Batch',
                 save_path=DEFAULT_PATH):
        self._ok        = False
        self.steps      = []
        self.losses     = []
        self.accs       = []
        self.val_losses = []
        self.val_accs   = []
        self._xlabel    = xlabel
        self._start     = time.time()
        self._last_save = 0.0
        self._path      = save_path

        try:
            import matplotlib
            matplotlib.use('Agg')          # non-interactive — no window
            import matplotlib.pyplot as plt
            self._plt = plt

            self.fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            self.fig.suptitle(title, fontsize=12, fontweight='bold')
            self._ax_loss, self._ax_acc, self._ax_info = axes
            plt.tight_layout(pad=2.0)

            os.makedirs(os.path.dirname(os.path.abspath(self._path)),
                        exist_ok=True)
            self._ok = True
            print(f'Live plot → {self._path}')
            print(f'  Open that file in an image viewer to watch training.')
        except Exception as e:
            print(f'Live plot unavailable: {e}')
            print('Training continues without visualization.')

    # ------------------------------------------------------------------

    def update(self, step, loss, acc, val_loss=None, val_acc=None):
        self.steps.append(step)
        self.losses.append(loss)
        self.accs.append(acc * 100.0)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if val_acc is not None:
            self.val_accs.append(val_acc * 100.0)

        if not self._ok:
            return

        now = time.time()
        if now - self._last_save >= self._MIN_INTERVAL:
            try:
                self._draw()
                self._last_save = now
            except Exception:
                pass

    def close(self):
        """Force a final save on training end."""
        if self._ok:
            try:
                self._draw()
                print(f'  Final plot saved → {self._path}')
            except Exception:
                pass
            try:
                self._plt.close(self.fig)
            except Exception:
                pass

    # ------------------------------------------------------------------

    def _gpu_info(self):
        try:
            import torch
            if torch.cuda.is_available():
                alloc    = torch.cuda.memory_allocated(0) / 1024 ** 3
                reserved = torch.cuda.memory_reserved(0) / 1024 ** 3
                total    = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                pct      = alloc / total * 100
                name     = torch.cuda.get_device_properties(0).name
                return (f'GPU      : {name}\n'
                        f'VRAM     : {alloc:.2f}/{total:.1f} GB  ({pct:.0f}%)\n'
                        f'Reserved : {reserved:.2f} GB')
        except Exception:
            pass
        return 'GPU : CPU only'

    def _draw(self):
        s       = self.steps
        elapsed = time.time() - self._start

        # ── Loss ──────────────────────────────────────────────────────
        self._ax_loss.cla()
        self._ax_loss.plot(s, self.losses, 'b-o', lw=2, ms=3, label='Train')
        if self.val_losses:
            n = len(self.val_losses)
            self._ax_loss.plot(s[:n], self.val_losses, 'r--s', lw=2, ms=3,
                               label='Val')
            self._ax_loss.legend(fontsize=9)
        self._ax_loss.set(title='Loss', xlabel=self._xlabel, ylabel='BCE Loss')
        self._ax_loss.grid(alpha=0.3)
        if len(s) > 1:
            self._ax_loss.set_xlim(s[0], s[-1])

        # ── Accuracy ──────────────────────────────────────────────────
        self._ax_acc.cla()
        self._ax_acc.plot(s, self.accs, color='#27ae60', marker='o',
                          lw=2, ms=3, label='Train')
        if self.val_accs:
            n = len(self.val_accs)
            self._ax_acc.plot(s[:n], self.val_accs, color='#e67e22',
                              linestyle='--', marker='s', lw=2, ms=3,
                              label='Val')
            self._ax_acc.legend(fontsize=9)
        self._ax_acc.set(title='Accuracy', xlabel=self._xlabel, ylabel='%')
        self._ax_acc.set_ylim(0, 100)
        self._ax_acc.axhline(50, color='gray', ls=':', alpha=0.4, lw=1)
        self._ax_acc.grid(alpha=0.3)
        if len(s) > 1:
            self._ax_acc.set_xlim(s[0], s[-1])

        # ── Stats panel ───────────────────────────────────────────────
        self._ax_info.cla()
        self._ax_info.axis('off')

        lines = [
            f'Step     : {s[-1]}',
            '',
            f'Loss     : {self.losses[-1]:.4f}',
            f'Accuracy : {self.accs[-1]:.1f}%',
        ]
        if self.val_losses:
            lines += ['', f'Val Loss : {self.val_losses[-1]:.4f}',
                      f'Val Acc  : {self.val_accs[-1]:.1f}%']
        lines += ['', f'Elapsed  : {elapsed / 60:.1f} min', self._gpu_info()]

        self._ax_info.text(
            0.06, 0.94, '\n'.join(lines),
            transform=self._ax_info.transAxes, va='top',
            fontfamily='monospace', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#fffbe6',
                      edgecolor='#ccc', alpha=0.95),
        )
        self._ax_info.set_title('Stats', fontsize=10)

        self.fig.savefig(self._path, dpi=96, bbox_inches='tight')
