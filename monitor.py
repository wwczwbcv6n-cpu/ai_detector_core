"""
monitor.py — AI Detector Live Dashboard
Run:  python monitor.py
"""
import os, sys, time, subprocess
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich import box
from rich.align import Align
from rich.rule import Rule

console = Console()

# ── Paths ────────────────────────────────────────────────────────────────────
TRANSCODE_IN  = "data/temp_frames/sTsjt8J4OmA.mp4"
TRANSCODE_OUT = "data/temp_frames/sTsjt8J4OmA_1080p.webm"
TRAIN_LOG     = "logs/train_video.log"
CKPT_PATH     = "models/ai_detector_unified_v1.pth"
SRC_DURATION  = 42896.481   # seconds
EST_OUT_GB    = 7.7         # VP9 webm download size

# ── Helpers ──────────────────────────────────────────────────────────────────
def fsize(path):
    try:
        b = os.path.getsize(path)
        for u in ["B","KB","MB","GB","TB"]:
            if b < 1024: return f"{b:.1f} {u}"
            b /= 1024
    except:
        return "—"

def ftime(s):
    s = max(0, int(s))
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"

def ffmpeg_running():
    try:
        return bool(subprocess.check_output(["pgrep","-x","ffmpeg"],
                                             stderr=subprocess.DEVNULL).strip())
    except:
        return False

def training_running():
    try:
        return bool(subprocess.check_output(
            ["pgrep","-f","train_video_section"],
            stderr=subprocess.DEVNULL).strip())
    except:
        return False

def last_log(n=14):
    try:
        with open(TRAIN_LOG) as f:
            return f.readlines()[-n:]
    except:
        return []

def parse_metrics(lines):
    loss = acc = epoch = step = None
    for ln in reversed(lines):
        if "loss=" in ln and acc is None:
            try:
                for p in ln.split():
                    if p.startswith("loss="): loss = float(p.split("=")[1].rstrip(","))
                    if p.startswith("acc="):  acc  = float(p.split("=")[1].rstrip(","))
            except: pass
        if "Epoch " in ln and epoch is None:
            try: epoch = int(ln.split("Epoch")[1].strip().split("/")[0])
            except: pass
        if "step " in ln and step is None:
            try: step = int(ln.strip().split("step")[1].strip().split()[0])
            except: pass
    return loss, acc, epoch, step

def ckpt_age():
    try:
        s = int(time.time() - os.path.getmtime(CKPT_PATH))
        return ftime(s) + " ago"
    except:
        return "not saved yet"

# ── Dashboard builder ────────────────────────────────────────────────────────
def build_dashboard(elapsed, tc_frac, tc_eta, ff_alive, tr_alive,
                    loss, acc, epoch, step, log_lines, pulse):

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=4),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(name="left"),
        Layout(name="right", ratio=2),
    )
    layout["body"]["left"].split_column(
        Layout(name="phase1"),
        Layout(name="phase2"),
    )

    # ── Header ───────────────────────────────────────────────────────────────
    spinner = ("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")[pulse % 10]
    header_text = Text(justify="center")
    header_text.append("  🤖  AI DETECTOR — LIVE DASHBOARD  ", style="bold white on dark_blue")
    header_text.append(f"  {spinner}  ", style="bold cyan")
    header_text.append(f"uptime {ftime(elapsed)}", style="dim white")
    layout["header"].update(Panel(Align.center(header_text), style="dark_blue", padding=(0,1)))

    # ── Phase 1: Transcode ────────────────────────────────────────────────────
    if ff_alive:
        tc_icon   = "[bold green]⟳ RUNNING[/]"
        tc_color  = "green"
    elif os.path.exists(TRANSCODE_OUT):
        tc_icon   = "[bold blue]✔ COMPLETE[/]"
        tc_color  = "blue"
    else:
        tc_icon   = "[dim]◌ WAITING[/]"
        tc_color  = "dim"

    bar_filled = int(tc_frac * 28)
    bar_empty  = 28 - bar_filled
    prog_bar   = f"[green]{'█' * bar_filled}[/][dim]{'░' * bar_empty}[/]"

    tc_table = Table.grid(padding=(0, 1))
    tc_table.add_column(style="dim", width=10)
    tc_table.add_column()
    tc_table.add_row("Status",   tc_icon)
    tc_table.add_row("Source",   "[cyan]YouTube VP9[/] → [green]1080p webm[/]")
    tc_table.add_row("Source",   f"[dim]{fsize(TRANSCODE_IN)}[/]")
    tc_table.add_row("Output",   f"[yellow]{fsize(TRANSCODE_OUT)}[/]")
    tc_table.add_row("Progress", f"{prog_bar} [bold]{tc_frac*100:.1f}%[/]")
    if ff_alive and tc_frac > 0.01:
        tc_table.add_row("ETA",  f"[yellow]{ftime(tc_eta)}[/] remaining")
    elif not ff_alive and os.path.exists(TRANSCODE_OUT):
        tc_table.add_row("ETA",  "[green]Done ✓[/]")

    layout["phase1"].update(
        Panel(tc_table, title="[bold]① DOWNLOAD[/]",
              border_style=tc_color, padding=(0, 1))
    )

    # ── Phase 2: Training ─────────────────────────────────────────────────────
    if tr_alive:
        tr_icon  = "[bold green]⟳ RUNNING[/]"
        tr_color = "green"
    elif ff_alive:
        tr_icon  = "[dim]◌ WAITING FOR TRANSCODE[/]"
        tr_color = "dim"
    else:
        tr_icon  = "[yellow]◌ READY TO START[/]"
        tr_color = "yellow"

    tr_table = Table.grid(padding=(0, 1))
    tr_table.add_column(style="dim", width=10)
    tr_table.add_column()
    tr_table.add_row("Status",  tr_icon)
    tr_table.add_row("Model",   "[cyan]UnifiedFusionNet v1[/]  16 branches")
    tr_table.add_row("PRNU",    "[magenta]64-dim[/]  spatial 128×128")
    tr_table.add_row("GPU",     "[green]RTX 3050 Laptop[/]  3760 MB")
    if epoch is not None:
        tr_table.add_row("Epoch",  f"[bold green]{epoch}[/]")
    if step is not None:
        tr_table.add_row("Step",   f"[green]{step:,}[/]")
    if loss is not None:
        loss_pct = max(0, 1.0 - loss)
        lb = f"[green]{'█'*int(loss_pct*12)}[/][dim]{'░'*(12-int(loss_pct*12))}[/]"
        tr_table.add_row("Loss",   f"[yellow]{loss:.4f}[/]  {lb}")
    if acc is not None:
        ab = f"[green]{'█'*int(acc*12)}[/][dim]{'░'*(12-int(acc*12))}[/]"
        tr_table.add_row("Acc",    f"[bold green]{acc*100:.2f}%[/]  {ab}")
    tr_table.add_row("Ckpt",    f"[dim]{ckpt_age()}[/]")

    layout["phase2"].update(
        Panel(tr_table, title="[bold]② VIDEO TRAINING[/]",
              border_style=tr_color, padding=(0, 1))
    )

    # ── Right: Log panel ──────────────────────────────────────────────────────
    log_text = Text()
    if log_lines:
        for ln in log_lines:
            ln = ln.rstrip()
            if not ln: continue
            if "loss=" in ln or "acc=" in ln:
                log_text.append("  " + ln + "\n", style="bold green")
            elif "Error" in ln.lower() or "warn" in ln.lower():
                log_text.append("  " + ln + "\n", style="bold red")
            elif "Epoch" in ln:
                log_text.append("  " + ln + "\n", style="bold yellow")
            elif "✓" in ln or "complete" in ln.lower():
                log_text.append("  " + ln + "\n", style="cyan")
            elif "SECTION" in ln or "===" in ln:
                log_text.append("  " + ln + "\n", style="bold blue")
            else:
                log_text.append("  " + ln + "\n", style="dim white")
    else:
        log_text.append("\n  Waiting for training to start...\n", style="dim")
        log_text.append("\n  Transcode is running in the background.\n", style="dim")
        log_text.append("  Training will auto-start when done.\n", style="dim")

    layout["right"].update(
        Panel(log_text, title="[bold]📋 TRAINING LOG[/]  logs/train_video.log",
              border_style="blue", padding=(0, 1))
    )

    # ── Footer ────────────────────────────────────────────────────────────────
    footer = Text(justify="center")
    footer.append("  [q] quit    ", style="dim")
    footer.append("  Saves → models/ai_detector_unified_v1.pth    ", style="dim")
    footer.append("  Refreshes every 3s  ", style="dim")
    layout["footer"].update(Panel(Align.center(footer), style="dim", padding=(0,1)))

    return layout

# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    start_t = time.time()
    pulse   = 0

    with Live(console=console, refresh_per_second=1, screen=True) as live:
        while True:
            try:
                now      = time.time()
                elapsed  = now - start_t
                ff_alive = ffmpeg_running()
                tr_alive = training_running()

                out_gb   = (os.path.getsize(TRANSCODE_OUT) / 1024**3
                            if os.path.exists(TRANSCODE_OUT) else 0.0)
                tc_frac  = min(out_gb / EST_OUT_GB, 1.0)
                tc_eta   = ((1.0 - tc_frac) * elapsed / tc_frac
                            if tc_frac > 0.01 else 0)

                lines        = last_log(14)
                loss, acc, epoch, step = parse_metrics(lines)

                dash = build_dashboard(elapsed, tc_frac, tc_eta,
                                       ff_alive, tr_alive,
                                       loss, acc, epoch, step, lines, pulse)
                live.update(dash)
                pulse += 1
                time.sleep(3)

            except KeyboardInterrupt:
                break

    console.print("\n[yellow]Monitor closed.[/]")

if __name__ == "__main__":
    main()
