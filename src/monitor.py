"""
monitor.py — Live web dashboard for meta_ai_cycle.py training.

Open http://localhost:8765 in your browser — auto-refreshes every 2 s.
Run alongside training:
    python src/monitor.py
"""

import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(SCRIPT_DIR, '..', 'models')
STATUS_FILE = os.path.join(MODELS_DIR, 'cycle_status.json')
PLOT_FILE   = os.path.join(MODELS_DIR, 'training_progress.png')
PORT        = 8765

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AI Detector — Cycle Trainer</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#0d1117;color:#e6edf3;font-family:'Courier New',monospace;padding:20px;min-height:100vh}
  h1{font-size:1.3rem;color:#a78bfa;margin-bottom:18px;letter-spacing:1px}
  .badge{background:#22c55e22;color:#22c55e;border:1px solid #22c55e44;
         border-radius:99px;padding:2px 10px;font-size:.7rem;margin-left:8px;vertical-align:middle}
  .badge.warn{background:#f59e0b22;color:#f59e0b;border-color:#f59e0b44}

  /* ── stat chips ── */
  .chips{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:20px}
  .chip{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:12px 18px;min-width:130px}
  .chip .val{font-size:1.7rem;font-weight:700;color:#a78bfa;line-height:1}
  .chip .lbl{font-size:.7rem;color:#8b949e;margin-top:4px;text-transform:uppercase;letter-spacing:.5px}

  /* ── panels ── */
  .row{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:16px}
  .panel{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;flex:1;min-width:280px}
  .panel h2{font-size:.75rem;color:#8b949e;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px}
  img{max-width:100%;border-radius:6px;display:block}

  /* ── loss chart (SVG) ── */
  #chart-wrap{width:100%;height:220px;position:relative}
  svg{width:100%;height:100%}
  .grid-line{stroke:#21262d;stroke-width:1}
  .loss-line{stroke:#a78bfa;stroke-width:2;fill:none}
  .loss-area{fill:url(#grad);opacity:.35}

  /* ── GPU bar ── */
  .bar-bg{background:#21262d;border-radius:99px;height:10px;overflow:hidden;margin:6px 0}
  .bar-fg{height:100%;border-radius:99px;transition:width .5s}
  .bar-gpu{background:linear-gradient(90deg,#7c3aed,#a78bfa)}
  .bar-ram{background:linear-gradient(90deg,#0ea5e9,#38bdf8)}
  .bar-vram{background:linear-gradient(90deg,#10b981,#34d399)}

  .stat-row{display:flex;justify-content:space-between;font-size:.78rem;margin-bottom:2px}
  .stat-row .k{color:#8b949e}
  .stat-row .v{color:#e6edf3;font-weight:600}

  .ts{color:#484f58;font-size:.68rem;margin-top:14px;text-align:right}
</style>
</head>
<body>
<h1>AI Detector — Cycle Trainer <span class="badge" id="live-badge">● LIVE</span></h1>

<div class="chips">
  <div class="chip"><div class="val" id="v-total">—</div><div class="lbl">Videos trained</div></div>
  <div class="chip"><div class="val" id="v-cycle">— / —</div><div class="lbl">This cycle</div></div>
  <div class="chip"><div class="val" id="v-loss">—</div><div class="lbl">Last loss</div></div>
  <div class="chip"><div class="val" id="v-frames">—</div><div class="lbl">Total frames</div></div>
  <div class="chip"><div class="val" id="v-elapsed">—</div><div class="lbl">Elapsed</div></div>
  <div class="chip"><div class="val" id="v-gpu">—%</div><div class="lbl">GPU util</div></div>
</div>

<div class="row">
  <!-- Loss curve -->
  <div class="panel" style="flex:2;min-width:340px">
    <h2>Loss curve (per video)</h2>
    <div id="chart-wrap">
      <svg id="svg-chart" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <linearGradient id="grad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#a78bfa"/>
            <stop offset="100%" stop-color="#a78bfa" stop-opacity="0"/>
          </linearGradient>
        </defs>
        <text x="50%" y="50%" text-anchor="middle" fill="#484f58" font-size="13">
          Waiting for data…
        </text>
      </svg>
    </div>
  </div>

  <!-- System stats -->
  <div class="panel" style="min-width:240px;max-width:320px">
    <h2>System</h2>

    <div class="stat-row"><span class="k">GPU util</span><span class="v" id="s-gpu">—</span></div>
    <div class="bar-bg"><div class="bar-fg bar-gpu" id="bar-gpu" style="width:0%"></div></div>

    <div class="stat-row" style="margin-top:8px"><span class="k">VRAM</span><span class="v" id="s-vram">—</span></div>
    <div class="bar-bg"><div class="bar-fg bar-vram" id="bar-vram" style="width:0%"></div></div>

    <div class="stat-row" style="margin-top:8px"><span class="k">RAM</span><span class="v" id="s-ram">—</span></div>
    <div class="bar-bg"><div class="bar-fg bar-ram" id="bar-ram" style="width:0%"></div></div>

    <div style="margin-top:16px">
      <div class="stat-row"><span class="k">Current video</span><span class="v" id="s-vid" style="font-size:.7rem;max-width:160px;text-align:right;word-break:break-all">—</span></div>
      <div class="stat-row" style="margin-top:6px"><span class="k">Updated</span><span class="v" id="s-upd" style="font-size:.7rem">—</span></div>
    </div>
  </div>
</div>

<!-- Training progress PNG from LivePlot -->
<div class="panel">
  <h2>Training plot (loss + accuracy + stats)</h2>
  <img id="plot-img" src="/plot.png" alt="training plot" onerror="this.style.display='none'">
</div>

<div class="ts" id="ts">—</div>

<script>
var _losses = [];

function drawChart(losses) {
  var svg = document.getElementById('svg-chart');
  var W = svg.clientWidth || 600, H = svg.clientHeight || 220;
  var pad = {l:42, r:12, t:10, b:28};
  var iW = W - pad.l - pad.r, iH = H - pad.t - pad.b;

  if (!losses || losses.length < 2) return;
  var mn = Math.min.apply(null, losses) * 0.95;
  var mx = Math.max.apply(null, losses) * 1.05;
  if (mx === mn) mx = mn + 0.1;

  function cx(i) { return pad.l + (i / (losses.length-1)) * iW; }
  function cy(v) { return pad.t + (1 - (v - mn)/(mx - mn)) * iH; }

  var lines = '', areas = '';
  // grid
  var steps = 4;
  for (var i=0; i<=steps; i++) {
    var v = mn + (mx-mn)*i/steps;
    var y = cy(v);
    lines += '<line class="grid-line" x1="'+pad.l+'" y1="'+y+'" x2="'+(W-pad.r)+'" y2="'+y+'"/>';
    lines += '<text x="'+(pad.l-4)+'" y="'+(y+4)+'" text-anchor="end" fill="#484f58" font-size="10">'+v.toFixed(2)+'</text>';
  }

  // x labels
  var step = Math.max(1, Math.floor(losses.length / 6));
  for (var i=0; i<losses.length; i+=step) {
    lines += '<text x="'+cx(i)+'" y="'+(H-6)+'" text-anchor="middle" fill="#484f58" font-size="10">'+(i+1)+'</text>';
  }

  // area path
  var pts = losses.map(function(v,i){return cx(i)+','+cy(v);}).join(' ');
  var apath = 'M'+cx(0)+','+cy(losses[0])+' '+
    losses.map(function(v,i){return 'L'+cx(i)+','+cy(v);}).join(' ')+
    ' L'+cx(losses.length-1)+','+(pad.t+iH)+' L'+pad.l+','+(pad.t+iH)+' Z';
  areas = '<path class="loss-area" d="'+apath+'"/>';

  // line
  var lpath = 'M'+cx(0)+','+cy(losses[0])+' '+
    losses.map(function(v,i){return 'L'+cx(i)+','+cy(v);}).join(' ');
  areas += '<path class="loss-line" d="'+lpath+'"/>';

  // last point dot
  var lx=cx(losses.length-1), ly=cy(losses[losses.length-1]);
  areas += '<circle cx="'+lx+'" cy="'+ly+'" r="4" fill="#a78bfa"/>';

  svg.innerHTML = '<defs><linearGradient id="grad" x1="0" y1="0" x2="0" y2="1">'+
    '<stop offset="0%" stop-color="#a78bfa"/><stop offset="100%" stop-color="#a78bfa" stop-opacity="0"/></linearGradient></defs>'+
    lines + areas;
}

function refresh() {
  // Stats
  fetch('/stats').then(function(r){return r.json();}).then(function(d) {
    document.getElementById('v-total').textContent   = d.total_videos || '—';
    document.getElementById('v-cycle').textContent   = (d.cycle_videos||'—') + ' / ' + (d.cycle_size||'—');
    document.getElementById('v-loss').textContent    = d.last_loss !== null ? d.last_loss.toFixed(4) : '—';
    document.getElementById('v-frames').textContent  = (d.total_frames||0).toLocaleString();
    document.getElementById('v-elapsed').textContent = d.elapsed_min ? d.elapsed_min.toFixed(0)+'m' : '—';
    document.getElementById('v-gpu').textContent     = (d.gpu_util_pct||0) + '%';

    var gu = d.gpu_util_pct || 0;
    document.getElementById('s-gpu').textContent  = gu + '%';
    document.getElementById('bar-gpu').style.width = Math.min(gu,100) + '%';

    var vu = d.vram_total_gb > 0 ? (d.vram_used_gb/d.vram_total_gb*100).toFixed(0) : 0;
    document.getElementById('s-vram').textContent  = d.vram_used_gb.toFixed(1) + ' / ' + d.vram_total_gb.toFixed(1) + ' GB';
    document.getElementById('bar-vram').style.width = Math.min(vu,100) + '%';

    var ru = d.ram_pct || 0;
    document.getElementById('s-ram').textContent  = ru.toFixed(0) + '%';
    document.getElementById('bar-ram').style.width = Math.min(ru,100) + '%';

    document.getElementById('s-vid').textContent  = d.current_video || '—';
    document.getElementById('s-upd').textContent  = d.updated_at || '—';

    if (d.losses && d.losses.length > 1) {
      _losses = d.losses;
      drawChart(_losses);
    }

    var badge = document.getElementById('live-badge');
    var age = (Date.now()/1000) - (d._ts||0);
    if (age < 30) {
      badge.textContent = '● LIVE'; badge.className = 'badge';
    } else {
      badge.textContent = '⚠ PAUSED'; badge.className = 'badge warn';
    }

    document.getElementById('ts').textContent = 'Last fetch: ' + new Date().toLocaleTimeString();
  }).catch(function(){});

  // Refresh plot image
  var img = document.getElementById('plot-img');
  img.src = '/plot.png?t=' + Date.now();
}

setInterval(refresh, 2000);
window.onload = refresh;
window.addEventListener('resize', function(){ drawChart(_losses); });
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        path = self.path.split('?')[0]

        if path == '/stats':
            data = {
                'total_videos': 0, 'cycle_videos': 0, 'cycle_size': 100,
                'last_loss': None, 'losses': [], 'total_frames': 0,
                'elapsed_min': 0, 'current_video': '',
                'vram_used_gb': 0.0, 'vram_total_gb': 0.0,
                'gpu_util_pct': 0, 'ram_pct': 0,
                'updated_at': '', '_ts': 0,
            }
            if os.path.exists(STATUS_FILE):
                try:
                    with open(STATUS_FILE) as f:
                        s = json.load(f)
                    cycle_size = s.get('cycle_size', 100)
                    total      = s.get('total_videos', 0)
                    data.update({
                        'total_videos':  total,
                        'cycle_videos':  total % cycle_size if cycle_size else 0,
                        'cycle_size':    cycle_size,
                        'last_loss':     s.get('last_loss'),
                        'losses':        s.get('losses', []),
                        'total_frames':  s.get('total_frames', 0),
                        'elapsed_min':   s.get('elapsed_min', 0),
                        'current_video': s.get('current_video', ''),
                        'vram_used_gb':  s.get('vram_used_gb', 0.0),
                        'vram_total_gb': s.get('vram_total_gb', 0.0),
                        'gpu_util_pct':  s.get('gpu_util_pct', 0),
                        'ram_pct':       s.get('ram_pct', 0),
                        'updated_at':    s.get('updated_at', ''),
                        '_ts':           os.path.getmtime(STATUS_FILE),
                    })
                except Exception:
                    pass
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == '/plot.png':
            if os.path.exists(PLOT_FILE):
                with open(PLOT_FILE, 'rb') as f:
                    body = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'image/png')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404); self.end_headers()
            return

        # Main page
        body = HTML.encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', PORT), Handler)
    print(f'\nMonitor → http://localhost:{PORT}')
    print(f'  Auto-refreshes every 2 s  |  Ctrl+C to stop (training continues)\n')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('Monitor stopped.')
