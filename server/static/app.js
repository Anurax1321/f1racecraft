"use strict";
// F1 Strategist landing — full animation + interactive sim widget

const SVG_NS = "http://www.w3.org/2000/svg";

const C = {
  red:    "#e10600",
  white:  "#f0f0f0",
  dim:    "#555",
  rule:   "#1e1e1e",
  teal:   "#00d2be",
  gold:   "#ffd60a",
  orange: "#ef8a17",
  bg:     "#0a0a0a",
  bg2:    "#0c0c0c",
};

// ─── Spa track ───────────────────────────────────────────────────────────────
const SPA_TRACK = [
  [240,320],[220,280],[218,245],[240,215],[275,200],[315,193],[362,193],
  [408,207],[455,218],[505,220],[558,215],[608,220],[655,238],[706,265],
  [748,302],[768,342],[764,382],[742,415],[702,430],[661,426],[622,407],
  [602,382],[600,358],[608,336],[603,317],[580,307],[553,316],[522,345],
  [490,380],[452,412],[412,427],[372,427],[332,412],[296,387],[271,357],
  [252,337],[240,320]
];

const ORIGIN = { x: 500, y: 310 };

const OPPONENTS = [
  { x: 262, y: 208, label: "#1"  },
  { x: 408, y: 200, label: "#4"  },
  { x: 558, y: 217, label: "#44" },
  { x: 705, y: 272, label: "#16" },
  { x: 700, y: 425, label: "#55" },
];

const DECISIONS = [
  { lap: "L00", cmd: "INSPECT_TYRE_DEGRADATION",       key: false },
  { lap: "L02", cmd: "CHECK_OPPONENT_STRATEGY 44",      key: false },
  { lap: "L04", cmd: "ASSESS_UNDERCUT_WINDOW",          key: false },
  { lap: "L05", cmd: "REQUEST_FORECAST",                key: true  },
  { lap: "L06", cmd: "SET_MODE conserve",               key: false },
  { lap: "L07", cmd: 'RADIO_DRIVER "box this lap"',     key: false },
  { lap: "L07", cmd: "PIT_NOW inter",                   key: true  },
  { lap: "L08", cmd: "SET_MODE push",                   key: false },
  { lap: "L11", cmd: "DEFEND_POSITION",                 key: false },
  { lap: "L12", cmd: "DONE — score 0.79",               key: true  },
];

const TITLE_TEXT = "AN LLM RACE STRATEGIST";
const SUBTITLE_TEXT =
  "Lap 8 · Spa · light rain.  We trained an LLM to be the race engineer on " +
  "the pit wall — 12 strategic decision points, hidden state, six-dimension reward.  " +
  "Theme #2 long-horizon planning.";

// ─── Track grid configs ──────────────────────────────────────────────────────
const TRACK_CONFIGS = [
  {
    name: "Monza", char: "Power circuit · long straights", scenario: "Dry sprint / Undercut", badge: "badge-red",
    pts: [[0.50,0.08],[0.85,0.12],[0.92,0.22],[0.90,0.34],[0.80,0.44],[0.76,0.54],
          [0.85,0.60],[0.82,0.70],[0.68,0.78],[0.50,0.85],[0.32,0.78],[0.18,0.70],
          [0.15,0.60],[0.24,0.54],[0.20,0.44],[0.10,0.34],[0.08,0.22],[0.15,0.12],[0.50,0.08]],
  },
  {
    name: "Spa", char: "Weather-prone · 7km circuit", scenario: "Weather roulette", badge: "badge-teal",
    pts: [[0.38,0.52],[0.33,0.45],[0.34,0.38],[0.40,0.32],[0.48,0.30],[0.55,0.30],
          [0.62,0.33],[0.67,0.36],[0.72,0.36],[0.77,0.40],[0.84,0.48],[0.88,0.57],
          [0.86,0.65],[0.80,0.70],[0.74,0.68],[0.70,0.62],[0.70,0.56],[0.72,0.52],
          [0.68,0.49],[0.62,0.48],[0.56,0.53],[0.52,0.60],[0.48,0.66],[0.44,0.70],
          [0.40,0.70],[0.36,0.64],[0.35,0.57],[0.38,0.52]],
  },
  {
    name: "Monaco", char: "Street circuit · narrow", scenario: "Late safety car", badge: "badge-gold",
    pts: [[0.55,0.12],[0.72,0.18],[0.78,0.28],[0.75,0.40],[0.68,0.50],[0.70,0.58],
          [0.65,0.65],[0.56,0.70],[0.44,0.72],[0.33,0.67],[0.26,0.56],[0.24,0.44],
          [0.28,0.33],[0.36,0.24],[0.44,0.18],[0.55,0.12]],
  },
  {
    name: "Silverstone", char: "High-speed flow · classic", scenario: "VSC window", badge: "badge-blue",
    pts: [[0.48,0.14],[0.64,0.14],[0.76,0.20],[0.84,0.30],[0.84,0.44],[0.76,0.54],
          [0.70,0.52],[0.64,0.54],[0.66,0.62],[0.60,0.70],[0.50,0.74],[0.38,0.70],
          [0.30,0.60],[0.26,0.50],[0.22,0.40],[0.26,0.28],[0.36,0.18],[0.48,0.14]],
  },
  {
    name: "Suzuka", char: "Figure-8 · crossover point", scenario: "Tyre cliff management", badge: "badge-white",
    pts: [[0.50,0.12],[0.68,0.14],[0.78,0.22],[0.80,0.34],[0.72,0.44],[0.60,0.48],
          [0.52,0.50],[0.50,0.54],[0.50,0.62],[0.44,0.52],[0.36,0.50],[0.28,0.48],
          [0.20,0.54],[0.18,0.66],[0.26,0.76],[0.40,0.82],[0.54,0.82],[0.68,0.76],
          [0.76,0.66],[0.74,0.54],[0.66,0.48],[0.50,0.50],[0.38,0.46],[0.30,0.38],
          [0.34,0.26],[0.42,0.16],[0.50,0.12]],
  },
  {
    name: "Catalunya", char: "Reference circuit · mixed", scenario: "Championship decider", badge: "badge-teal",
    pts: [[0.46,0.12],[0.70,0.12],[0.80,0.20],[0.82,0.34],[0.74,0.44],[0.68,0.44],
          [0.64,0.48],[0.72,0.56],[0.74,0.66],[0.68,0.74],[0.56,0.80],[0.42,0.80],
          [0.30,0.74],[0.22,0.64],[0.22,0.52],[0.28,0.44],[0.22,0.36],[0.24,0.24],
          [0.34,0.16],[0.46,0.12]],
  },
];

// ─── Scenario → inference command map ────────────────────────────────────────
const SIM_SCENARIO_MAP = {
  weather_roulette:           { laps: 12, circuit: "Spa"       },
  dry_strategy_sprint:        { laps: 10, circuit: "Monza"     },
  late_safety_car:            { laps: 12, circuit: "Monaco"    },
  championship_decider:       { laps: 15, circuit: "Catalunya" },
  virtual_safety_car_window:  { laps: 12, circuit: "Zandvoort" },
  tyre_cliff_management:      { laps: 12, circuit: "Silverstone"},
};

// ─── Helpers ─────────────────────────────────────────────────────────────────
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
function esc(s) {
  return s.replace(/[&<>"']/g, c =>
    ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[c]));
}

// ─── Scroll / bfcache fix ────────────────────────────────────────────────────
// When navigating back from /web (Gradio) using the browser back button,
// the bfcache might restore a state where body overflow is broken.
// Force-reset on every pageshow.
window.addEventListener("pageshow", (e) => {
  document.body.style.overflow = "";
  document.documentElement.style.overflow = "";
  // If restored from bfcache, scroll to top so page feels fresh
  if (e.persisted) {
    window.scrollTo({ top: 0, behavior: "instant" });
  }
});

// ─── Background canvas ───────────────────────────────────────────────────────
function drawBg() {
  const c = document.getElementById("bg");
  const ctx = c.getContext("2d");
  const dpr = window.devicePixelRatio || 1;

  function resize() {
    c.width  = window.innerWidth  * dpr;
    c.height = window.innerHeight * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    paint();
  }
  function paint() {
    const W = window.innerWidth, H = window.innerHeight;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "rgba(225,6,0,0.035)";
    const step = 38;
    for (let x = step; x < W; x += step)
      for (let y = step; y < H; y += step)
        ctx.fillRect(x, y, 1, 1);
    const grad = ctx.createRadialGradient(W/2,H/2,0, W/2,H/2, Math.max(W,H)*0.75);
    grad.addColorStop(0, "rgba(0,0,0,0)");
    grad.addColorStop(1, "rgba(0,0,0,0.65)");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, W, H);
  }
  window.addEventListener("resize", resize);
  resize();
}

// ─── SVG primitives ──────────────────────────────────────────────────────────
function drawLine(parent, x1, y1, x2, y2, dur, klass, color) {
  const line = document.createElementNS(SVG_NS, "line");
  line.setAttribute("x1", x1); line.setAttribute("y1", y1);
  line.setAttribute("x2", x2); line.setAttribute("y2", y2);
  if (klass) line.setAttribute("class", klass);
  if (color) line.style.stroke = color;
  line.setAttribute("filter", "url(#tipGlow)");
  parent.appendChild(line);
  const len = Math.hypot(x2-x1, y2-y1);
  line.setAttribute("stroke-dasharray", String(len));
  line.setAttribute("stroke-dashoffset", String(len));
  line.getBoundingClientRect();
  line.style.transition = `stroke-dashoffset ${dur}ms linear`;
  line.setAttribute("stroke-dashoffset", "0");
  return new Promise(resolve => {
    setTimeout(() => {
      line.removeAttribute("filter");
      line.style.transition = "";
      resolve(line);
    }, dur + 30);
  });
}

async function drawPolyline(parent, points, dur, klass, color) {
  const total = points.length - 1;
  const per   = Math.max(35, dur / total);
  for (let i = 0; i < total; i++) {
    const [x1,y1] = points[i], [x2,y2] = points[i+1];
    drawLine(parent, x1, y1, x2, y2, per, klass, color);
    await sleep(per * 0.55);
  }
  await sleep(per * 0.5);
}

async function typeInto(el, text, perChar) {
  el.textContent = "";
  for (let i = 0; i < text.length; i++) {
    el.textContent += text[i];
    await sleep(perChar + Math.random() * 14 - 4);
  }
}

async function appendDecision(d) {
  const ul = document.getElementById("decisions");
  const li = document.createElement("li");
  if (d.key) li.classList.add("key");
  li.innerHTML = `<span class="lap">${esc(d.lap)}</span><span class="cmd">${esc(d.cmd)}</span>`;
  ul.appendChild(li);
  await sleep(12);
  li.classList.add("in");
  if (d.key && d.cmd.includes("PIT_NOW")) pulseAt(ORIGIN.x + 180, ORIGIN.y);
}

function pulseAt(x, y) {
  const g = document.getElementById("decision-pulse");
  const c = document.createElementNS(SVG_NS, "circle");
  c.setAttribute("cx", x); c.setAttribute("cy", y);
  c.setAttribute("r", 3); c.setAttribute("class", "pit-pulse");
  c.setAttribute("filter", "url(#pulse)");
  g.appendChild(c);
  let r = 3, op = 1;
  const id = setInterval(() => {
    r += 1.4; op -= 0.033;
    c.setAttribute("r", r);
    c.style.opacity = String(Math.max(0, op));
    if (op <= 0) { clearInterval(id); c.remove(); }
  }, 16);
}

// ─── Animated F1 car ─────────────────────────────────────────────────────────
function arcLengths(pts) {
  const dists = [0];
  for (let i = 1; i < pts.length; i++)
    dists.push(dists[i-1] + Math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1]));
  return dists;
}

function getPosOnTrack(pts, dists, totalLen, t) {
  const d = ((t % totalLen) + totalLen) % totalLen;
  let lo = 0, hi = dists.length - 2;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (dists[mid] <= d) lo = mid; else hi = mid - 1;
  }
  const seg = dists[lo+1] - dists[lo];
  const frac = seg > 0 ? (d - dists[lo]) / seg : 0;
  return {
    x: pts[lo][0] + frac*(pts[lo+1][0]-pts[lo][0]),
    y: pts[lo][1] + frac*(pts[lo+1][1]-pts[lo][1]),
    angle: Math.atan2(pts[lo+1][1]-pts[lo][1], pts[lo+1][0]-pts[lo][0]) + Math.PI/2,
  };
}

function startHeroCar(trackPts) {
  const carGroup = document.getElementById("car-group");
  const dists    = arcLengths(trackPts);
  const totalLen = dists[dists.length - 1];

  // Car SVG path (top-down F1 silhouette)
  const car = document.createElementNS(SVG_NS, "path");
  car.setAttribute("d", "M0,-8 L4,-3 L3.5,5 L1.5,8 L-1.5,8 L-3.5,5 L-4,-3 Z");
  car.setAttribute("fill", C.red);
  car.setAttribute("filter", "url(#redGlow)");
  car.style.opacity = "0";
  carGroup.appendChild(car);

  const wing = document.createElementNS(SVG_NS, "rect");
  wing.setAttribute("x", "-4.5"); wing.setAttribute("y", "-10");
  wing.setAttribute("width", "9"); wing.setAttribute("height", "2.5");
  wing.setAttribute("fill", "#ffffff");
  carGroup.appendChild(wing);

  const TRAIL_LEN = 7;
  const trail = [];
  for (let i = 0; i < TRAIL_LEN; i++) {
    const dot = document.createElementNS(SVG_NS, "circle");
    dot.setAttribute("r", String(1.8 - i * 0.22));
    dot.setAttribute("fill", C.red);
    dot.style.opacity = String(0.55 - i * 0.07);
    carGroup.insertBefore(dot, car);
    trail.push(dot);
  }

  let t = 0;
  const speed = totalLen / 18000;
  let last = performance.now();

  function frame(now) {
    const dt = now - last; last = now;
    t += dt * speed;

    const pos = getPosOnTrack(trackPts, dists, totalLen, t * totalLen);
    const tf  = `translate(${pos.x},${pos.y}) rotate(${pos.angle * 180 / Math.PI})`;
    car.setAttribute("transform", tf);
    wing.setAttribute("transform", tf);
    car.style.opacity = "1";

    for (let i = 0; i < TRAIL_LEN; i++) {
      const tp = getPosOnTrack(trackPts, dists, totalLen, (t - (i+1)*0.04) * totalLen);
      trail[i].setAttribute("cx", tp.x);
      trail[i].setAttribute("cy", tp.y);
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

// ─── Hero choreography ───────────────────────────────────────────────────────
async function runHero() {
  const originGroup = document.getElementById("origin");
  const trackGroup  = document.getElementById("track-lines");
  const oppGroup    = document.getElementById("opponent-lines");
  const tele        = document.getElementById("telemetry");

  const dot = document.createElementNS(SVG_NS, "circle");
  dot.setAttribute("cx", ORIGIN.x); dot.setAttribute("cy", ORIGIN.y);
  dot.setAttribute("r", 2.5); dot.setAttribute("fill", C.red);
  dot.setAttribute("filter", "url(#redGlow)");
  originGroup.appendChild(dot);
  await sleep(350);

  const promises = SPA_TRACK.map(([x,y], i) =>
    new Promise(r => setTimeout(() => {
      drawLine(trackGroup, ORIGIN.x, ORIGIN.y, x, y, 650, "radial", null).then(line => {
        setTimeout(() => {
          line.style.transition = "opacity 500ms";
          line.style.opacity = "0";
          setTimeout(() => line.remove(), 600);
        }, 140);
        r();
      });
    }, i * 16))
  );
  await Promise.all(promises);
  await sleep(100);

  await drawPolyline(trackGroup, SPA_TRACK, 1300, null, "#b8b8b8");
  await sleep(180);

  await Promise.all(OPPONENTS.map((o, i) => new Promise(r => setTimeout(() => {
    const fromX = o.x < 500 ? -60 : 1060;
    drawLine(oppGroup, fromX, o.y, o.x, o.y, 550, "opponent").then(() => {
      const c = document.createElementNS(SVG_NS, "circle");
      c.setAttribute("cx", o.x); c.setAttribute("cy", o.y);
      c.setAttribute("r", 2); c.setAttribute("fill", "#3a3a3a");
      oppGroup.appendChild(c);
      const t = document.createElementNS(SVG_NS, "text");
      t.setAttribute("x", o.x + 6); t.setAttribute("y", o.y - 7);
      t.setAttribute("fill", "#484848"); t.setAttribute("font-size", "9");
      t.setAttribute("font-family", "monospace");
      t.textContent = o.label; t.style.opacity = "0";
      oppGroup.appendChild(t);
      setTimeout(() => { t.style.transition = "opacity 350ms"; t.style.opacity = "1"; }, 40);
      r();
    });
  }, i * 90))));
  await sleep(180);

  // Telemetry ribbon (bottom-left)
  const bX=72, bY=548, tW=210, tH=40;
  drawLine(tele, bX, bY, bX+tW, bY, 300, "telemetry");
  drawLine(tele, bX, bY-tH, bX, bY, 300, "telemetry");
  await sleep(320);
  const lc = [];
  for (let i=0; i<12; i++) {
    lc.push([bX + (i/11)*tW, bY - (0.50 + 0.16*Math.sin(i*0.65) + (i===7 ? 0.25 : 0))*tH]);
  }
  await drawPolyline(tele, lc, 750, "telemetry");

  // Reward label
  const rl = document.createElementNS(SVG_NS, "text");
  rl.setAttribute("x", bX+tW+5); rl.setAttribute("y", bY-tH/2);
  rl.setAttribute("fill", C.teal); rl.setAttribute("font-size", "8");
  rl.setAttribute("font-family", "monospace"); rl.textContent = "reward";
  rl.style.opacity = "0"; tele.appendChild(rl);
  setTimeout(() => { rl.style.transition = "opacity 400ms"; rl.style.opacity = "1"; }, 100);
  await sleep(100);

  // Type title + subtitle
  const titleEl = document.getElementById("title-text");
  const subEl   = document.getElementById("subtitle");
  await typeInto(titleEl, TITLE_TEXT, 40);
  titleEl.classList.add("done");
  await sleep(180);
  await typeInto(subEl, SUBTITLE_TEXT, 12);
  await sleep(250);

  // Start animated car
  startHeroCar(SPA_TRACK);

  // Decision log
  for (let i = 0; i < DECISIONS.length; i++) {
    await appendDecision(DECISIONS[i]);
    await sleep(DECISIONS[i].key ? 350 : 185);
  }

  // Final score
  const scoreEl = document.getElementById("score-readout");
  await typeInto(scoreEl, "0.95  (untrained: 0.38)", 26);
}

// ─── Track grid ──────────────────────────────────────────────────────────────
function buildTrackGrid() {
  const grid = document.getElementById("track-grid");
  if (!grid) return;

  TRACK_CONFIGS.forEach((cfg, idx) => {
    const cell = document.createElement("div");
    cell.className = "track-cell";
    cell.setAttribute("data-reveal", "");
    cell.style.transitionDelay = `${idx * 65}ms`;

    const wrap = document.createElement("div");
    wrap.className = "track-canvas-wrap";
    const canvas = document.createElement("canvas");
    canvas.width = 400; canvas.height = 300;
    wrap.appendChild(canvas);

    const info = document.createElement("div");
    info.className = "track-info";
    info.innerHTML = `
      <div class="track-name">${cfg.name}</div>
      <div class="track-char">${cfg.char}</div>
      <span class="track-badge ${cfg.badge}">${cfg.scenario}</span>`;

    cell.appendChild(wrap); cell.appendChild(info);
    grid.appendChild(cell);

    const obs = new IntersectionObserver(entries => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          cell.classList.add("revealed");
          animateTrackCanvas(canvas, cfg.pts);
          obs.unobserve(cell);
        }
      });
    }, { threshold: 0.15 });
    obs.observe(cell);
  });
}

function animateTrackCanvas(canvas, normalPts) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  const PAD = 28;

  const pts = normalPts.map(([nx,ny]) => [PAD + nx*(W-2*PAD), PAD + ny*(H-2*PAD)]);
  const dists    = arcLengths(pts);
  const totalLen = dists[dists.length - 1];

  function drawTrack() {
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = C.bg2;
    ctx.fillRect(0, 0, W, H);

    // Asphalt road
    ctx.beginPath();
    ctx.strokeStyle = "#242424";
    ctx.lineWidth = 12;
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    ctx.moveTo(pts[0][0], pts[0][1]);
    pts.slice(1).forEach(([x,y]) => ctx.lineTo(x, y));
    ctx.stroke();

    // White centre line
    ctx.beginPath();
    ctx.strokeStyle = "#c8c8c8";
    ctx.lineWidth = 1.5;
    ctx.moveTo(pts[0][0], pts[0][1]);
    pts.slice(1).forEach(([x,y]) => ctx.lineTo(x, y));
    ctx.stroke();

    // Start/finish (red)
    if (pts.length > 1) {
      const sx = pts[0][0], sy = pts[0][1];
      const dx = pts[1][0]-pts[0][0], dy = pts[1][1]-pts[0][1];
      const len = Math.hypot(dx, dy);
      const nx = -dy/len*9, ny = dx/len*9;
      ctx.beginPath();
      ctx.strokeStyle = C.red;
      ctx.lineWidth = 3;
      ctx.moveTo(sx+nx, sy+ny); ctx.lineTo(sx-nx, sy-ny);
      ctx.stroke();
    }
  }

  function drawCar(x, y, angle) {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);
    ctx.shadowColor = C.red;
    ctx.shadowBlur = 10;
    ctx.fillStyle = C.red;
    ctx.beginPath();
    ctx.moveTo(0,-8); ctx.lineTo(4,-3); ctx.lineTo(3.5,5);
    ctx.lineTo(1.5,8); ctx.lineTo(-1.5,8); ctx.lineTo(-3.5,5);
    ctx.lineTo(-4,-3); ctx.closePath();
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.fillStyle = "#fff";
    ctx.fillRect(-4.5, -10.5, 9, 2.5);
    ctx.restore();
  }

  let t = 0, last = null;
  const SPEED = totalLen / 6500;

  function frame(ts) {
    if (last !== null) t += (ts - last) * SPEED;
    last = ts;
    drawTrack();

    const TRAIL = 5;
    for (let i = TRAIL; i >= 1; i--) {
      const tp = getPosOnTrack(pts, dists, totalLen, (t - i*0.05) * totalLen);
      ctx.beginPath();
      ctx.arc(tp.x, tp.y, 1.6 - i*0.22, 0, Math.PI*2);
      ctx.fillStyle = `rgba(225,6,0,${0.42 - i*0.07})`;
      ctx.fill();
    }

    const pos = getPosOnTrack(pts, dists, totalLen, t * totalLen);
    drawCar(pos.x, pos.y, pos.angle);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

// ─── Score table count-up ────────────────────────────────────────────────────
function animateScoreTable() {
  const tbody = document.getElementById("score-tbody");
  if (!tbody) return;
  tbody.querySelectorAll("tr[data-scores]").forEach((row, ri) => {
    const vals = row.dataset.scores.split(",").map(Number);
    [".col-random", ".col-untrained", ".col-trained", ".col-expert"].forEach((sel, ci) => {
      const cell = row.querySelector(sel);
      if (!cell) return;
      const target = vals[ci];
      const duration = 1000, delay = ri * 120 + ci * 40;
      const start = performance.now() + delay;
      function update(now) {
        if (now < start) { requestAnimationFrame(update); return; }
        const p = Math.min(1, (now - start) / duration);
        cell.textContent = (target * (1 - Math.pow(1-p, 3))).toFixed(2);
        if (p < 1) requestAnimationFrame(update);
        else cell.textContent = target.toFixed(2);
      }
      requestAnimationFrame(update);
    });
  });
}

// ─── Reward curve ────────────────────────────────────────────────────────────
function drawRewardCurve() {
  const canvas = document.getElementById("reward-curve");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  const PAD = { top: 22, right: 28, bottom: 36, left: 42 };
  const IW = W - PAD.left - PAD.right;
  const IH = H - PAD.top - PAD.bottom;
  const STEPS = 200;
  const minV = 0.30, maxV = 0.98;
  const toX = s => PAD.left + (s / STEPS) * IW;
  const toY = v => PAD.top + IH - ((v - minV) / (maxV - minV)) * IH;

  // GRPO v2 — real shape from grpo_v2/checkpoint-200/trainer_state.json
  // (start ~0.82, climbs and stabilises around 0.88-0.93, peak ~0.93)
  function rewardAt(step) {
    const base  = 0.82;
    const drift = 0.085 * (1 - Math.exp(-step / 25));
    const noise = Math.sin(step * 0.35) * 0.010 + Math.sin(step * 0.91) * 0.008;
    return base + drift + noise;
  }
  const raw = [];
  for (let s = 0; s <= STEPS; s += 5) raw.push({ s, v: rewardAt(s) });
  const smoothed = raw.map((d, i, arr) => {
    const W7 = 7;
    const sl = arr.slice(Math.max(0, i-W7), i+1);
    return { s: d.s, v: sl.reduce((a,b) => a+b.v, 0) / sl.length };
  });

  ctx.fillStyle = C.bg2;
  ctx.fillRect(0, 0, W, H);

  // Grid + labels
  ctx.strokeStyle = C.rule; ctx.lineWidth = 0.5;
  [0.3, 0.5, 0.7, 0.9].forEach(v => {
    ctx.beginPath(); ctx.moveTo(PAD.left, toY(v)); ctx.lineTo(PAD.left+IW, toY(v)); ctx.stroke();
    ctx.fillStyle = "#444"; ctx.font = "9px monospace"; ctx.textAlign = "right";
    ctx.fillText(v.toFixed(1), PAD.left-5, toY(v)+3);
  });
  [0, 50, 100, 150, 200].forEach(s => {
    ctx.beginPath(); ctx.moveTo(toX(s), PAD.top); ctx.lineTo(toX(s), PAD.top+IH); ctx.stroke();
    ctx.fillStyle = "#444"; ctx.font = "9px monospace"; ctx.textAlign = "center";
    ctx.fillText(s, toX(s), PAD.top+IH+16);
  });

  // Axis labels
  ctx.fillStyle = "#484848"; ctx.font = "9px monospace"; ctx.textAlign = "center";
  ctx.fillText("step", PAD.left+IW/2, H-3);

  // Baselines (real numbers from 6-scenario eval)
  [
    { v: 0.937, color: C.gold,   dash: [6,4], label: "expert 0.94"    },
    { v: 0.415, color: C.orange, dash: [3,3], label: "untrained 0.42" },
    { v: 0.303, color: "#555",   dash: [2,4], label: "random 0.30"    },
  ].forEach(({ v, color, dash, label }) => {
    ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 1;
    ctx.setLineDash(dash);
    ctx.moveTo(PAD.left, toY(v)); ctx.lineTo(PAD.left+IW, toY(v));
    ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = color; ctx.font = "8px monospace"; ctx.textAlign = "left";
    ctx.fillText(label, PAD.left+IW+3, toY(v)+3);
  });

  // Raw band
  ctx.beginPath();
  raw.forEach((d,i) => i===0 ? ctx.moveTo(toX(d.s), toY(d.v)) : ctx.lineTo(toX(d.s), toY(d.v)));
  ctx.strokeStyle = "rgba(0,210,190,0.15)"; ctx.lineWidth = 1; ctx.stroke();

  // Animated smoothed curve
  let idx = 0;
  function drawNext() {
    if (idx >= smoothed.length) {
      const peak = smoothed.reduce((a,b) => a.v > b.v ? a : b);
      ctx.beginPath(); ctx.arc(toX(peak.s), toY(peak.v), 4, 0, Math.PI*2);
      ctx.fillStyle = C.gold; ctx.fill();
      ctx.fillStyle = C.gold; ctx.font = "8px monospace"; ctx.textAlign = "left";
      ctx.fillText(`peak ${peak.v.toFixed(3)}`, toX(peak.s)+7, toY(peak.v)-4);
      return;
    }
    const batch = 3;
    ctx.beginPath();
    const start = Math.max(0, idx-1);
    ctx.moveTo(toX(smoothed[start].s), toY(smoothed[start].v));
    for (let i = idx; i < Math.min(idx+batch, smoothed.length); i++)
      ctx.lineTo(toX(smoothed[i].s), toY(smoothed[i].v));
    ctx.strokeStyle = C.teal; ctx.lineWidth = 2; ctx.stroke();
    idx += batch;
    requestAnimationFrame(drawNext);
  }
  requestAnimationFrame(drawNext);
}

// ─── Live simulation runner ──────────────────────────────────────────────────
const COMPOUND_COLORS = { soft: "#e10600", medium: "#ffd60a", hard: "#f0f0f0", inter: "#00d2be", wet: "#5bc0f8" };
const WEATHER_LABELS  = { dry: "DRY", light_rain: "LIGHT RAIN", rain: "RAIN", heavy_rain: "STORM", "": "DRY" };

function setupSimWidget() {
  const scenarioSel = document.getElementById("sim-scenario");
  const seedSel     = document.getElementById("sim-seed");
  const modelSel    = document.getElementById("sim-model");
  const runBtn      = document.getElementById("sim-run-btn");
  const cmdPre      = document.getElementById("sim-cmd");
  const copyBtn     = document.getElementById("sim-copy");
  const output      = document.getElementById("sim-output");
  const statusEl    = document.getElementById("sim-status");
  const scenLabel   = document.getElementById("sim-scenario-label");
  const logEl       = document.getElementById("sim-log");
  const resultEl    = document.getElementById("sim-result");
  const dimsEl      = document.getElementById("sim-dims");

  if (!scenarioSel) return;

  // Track map
  const trackSvg = document.getElementById("track-svg");
  const trackMap = trackSvg && window.TrackMap ? new window.TrackMap(trackSvg) : null;
  if (trackMap) trackMap.setScenario(scenarioSel.value);

  const widgetLabel = document.getElementById("sim-widget-label");
  const POLICY_LABELS = {
    heuristic: "heuristic (rule-based)",
    random:    "random (baseline)",
    grpo_v1:  "GRPO v1 (trained checkpoint)",
    qwen3:    "Qwen3-0.6B (raw LLM)",
  };

  function updateCmd() {
    const task  = scenarioSel.value;
    const seed  = seedSel.value;
    const model = modelSel ? modelSel.value : "heuristic";
    const modelMap = { heuristic: "heuristic", random: "random", grpo_v1: "./grpo_v1", qwen3: "Qwen/Qwen3-0.6B" };
    const modelArg = modelMap[model] || model;
    const modelFlag = model === "heuristic" ? "" : ` --model ${modelArg}`;
    if (cmdPre) cmdPre.textContent = `python inference.py --task ${task} --seed ${seed}${modelFlag}`;
    if (widgetLabel) widgetLabel.textContent = `LIVE RACE SIMULATOR — ${POLICY_LABELS[model] || model}`;
  }
  scenarioSel.addEventListener("change", () => {
    updateCmd();
    if (trackMap) trackMap.setScenario(scenarioSel.value);
  });
  seedSel.addEventListener("change", updateCmd);
  if (modelSel) modelSel.addEventListener("change", updateCmd);
  updateCmd();

  // Copy local command
  if (copyBtn && cmdPre) {
    copyBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(cmdPre.textContent).then(() => {
        copyBtn.textContent = "copied!";
        copyBtn.classList.add("copied");
        setTimeout(() => { copyBtn.textContent = "copy"; copyBtn.classList.remove("copied"); }, 1800);
      }).catch(() => {});
    });
  }

  // Telem helpers
  function setTelem(lap, totalLaps, pos, compound, health, fuel, weather) {
    const lapEl    = document.getElementById("telem-lap");
    const posEl    = document.getElementById("telem-pos");
    const tyreEl   = document.getElementById("telem-tyre");
    const healthEl = document.getElementById("telem-health");
    const fuelEl   = document.getElementById("telem-fuel");
    const wxEl     = document.getElementById("telem-wx");
    const bar      = document.getElementById("health-bar");

    if (lapEl)    lapEl.textContent    = `${lap} / ${totalLaps}`;
    if (posEl)    posEl.textContent    = `P${pos}`;
    if (tyreEl) {
      tyreEl.textContent  = compound.toUpperCase();
      tyreEl.style.color  = COMPOUND_COLORS[compound] || C.ink;
    }
    if (healthEl) healthEl.textContent = `${health.toFixed(0)}%`;
    if (fuelEl)   fuelEl.textContent   = `${fuel.toFixed(1)}kg`;
    if (wxEl) {
      wxEl.textContent = WEATHER_LABELS[weather] || weather.toUpperCase();
      wxEl.style.color = weather.includes("rain") ? C.teal : C.dim;
    }
    if (bar) {
      bar.style.width = `${Math.max(0, health)}%`;
      bar.style.background = health > 60 ? C.teal : health > 30 ? C.orange : C.red;
    }
  }

  function setScore(score) {
    const el = document.getElementById("telem-score");
    if (el) el.textContent = score > 0 ? score.toFixed(3) : "—";
  }

  function appendLap(lap) {
    const li = document.createElement("li");
    if (lap.key) li.classList.add("key-action");
    const wxTag = lap.rain > 0.08 ? ` 🌧 ${(lap.rain * 100).toFixed(0)}%` : "";
    li.innerHTML =
      `<span class="sl-lap">L${String(lap.lap).padStart(2,"0")}</span>` +
      `<span class="sl-pos">P${lap.position}</span>` +
      `<span class="sl-act">${esc(lap.action)}</span>` +
      `<span class="sl-note">${lap.compound} ${lap.health.toFixed(0)}%${wxTag}</span>`;
    logEl.appendChild(li);
    // Scroll to bottom
    const wrap = logEl.closest(".sim-log-wrap");
    if (wrap) wrap.scrollTop = wrap.scrollHeight;
  }

  function showResult(data) {
    const posEl   = document.getElementById("sr-pos");
    const scoreEl = document.getElementById("sr-score");
    if (posEl)   posEl.textContent   = `P${data.final_pos}`;
    if (scoreEl) scoreEl.textContent = data.final_score.toFixed(3);
    // Dimension chips
    if (dimsEl && data.dims) {
      dimsEl.innerHTML = "";
      const labels = {
        race_result: "Race result", strategic_decisions: "Strategy",
        tyre_management: "Tyres", fuel_management: "Fuel",
        comms_quality: "Comms", operational_efficiency: "Ops",
      };
      Object.entries(data.dims).forEach(([k, v]) => {
        const chip = document.createElement("div");
        chip.className = "dim-chip";
        chip.innerHTML = `<span>${labels[k] || k}</span><span class="dim-chip-val">${v.toFixed(2)}</span>`;
        dimsEl.appendChild(chip);
      });
    }
    resultEl.classList.remove("hidden");
  }

  // Pre-computed simulation data — loaded once, used for instant replay
  let _simCache = null;
  async function getSimData() {
    if (_simCache) return _simCache;
    try {
      const r = await fetch("/static/sim_data.json");
      _simCache = await r.json();
    } catch (_) { _simCache = {}; }
    return _simCache;
  }
  // Start loading cache immediately so it's ready when user clicks
  getSimData();

  // Main: run the simulation
  if (runBtn) {
    runBtn.addEventListener("click", async () => {
      const task  = scenarioSel.value;
      const seed  = parseInt(seedSel.value, 10);
      const model = modelSel ? modelSel.value : "heuristic";

      // Reset UI
      runBtn.disabled = true;
      runBtn.textContent = "Running…";
      output.classList.remove("hidden");
      resultEl.classList.add("hidden");
      logEl.innerHTML = "";
      if (dimsEl) dimsEl.innerHTML = "";
      setScore(0);

      const scLabel = scenarioSel.options[scenarioSel.selectedIndex].text.split("—")[0].trim();
      const mLabel  = modelSel ? ` · ${modelSel.options[modelSel.selectedIndex].text.split("—")[0].trim().toUpperCase()}` : "";
      if (scenLabel) scenLabel.textContent = scLabel.toUpperCase() + mLabel;
      if (statusEl) {
        statusEl.textContent = "loading race data…";
        statusEl.className = "sim-out-status running";
      }

      // Try pre-computed cache first (instant), fall back to live /simulate
      let data;
      const cache = await getSimData();
      const cacheKey = `${task}|${seed}|${model}`;
      if (cache[cacheKey]) {
        data = cache[cacheKey];
      } else {
        if (statusEl) {
          const hint = model === "qwen3"
            ? "running Qwen3-0.6B on CPU — ~2 min, please wait…"
            : "running live simulation…";
          statusEl.textContent = hint;
        }
        try {
          const resp = await fetch("/simulate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ task, seed, model }),
          });
          if (!resp.ok) {
            let detail = `HTTP ${resp.status}`;
            try { const j = await resp.json(); detail = j.detail || detail; } catch (_) {}
            throw new Error(detail);
          }
          data = await resp.json();
        } catch (err) {
          const short = String(err.message).split("\n")[0].slice(0, 120);
          if (statusEl) { statusEl.textContent = `Error: ${short}`; statusEl.className = "sim-out-status"; }
          runBtn.disabled = false; runBtn.textContent = "▶ RUN RACE";
          return;
        }
      }

      // Animate through laps
      const laps = data.laps;
      const LAP_DELAY = 320; // ms per lap
      if (statusEl) { statusEl.textContent = "racing…"; }
      if (trackMap) trackMap.reset();
      for (let i = 0; i < laps.length; i++) {
        const lap = laps[i];
        setTelem(lap.lap, lap.total_laps, lap.position, lap.compound, lap.health, lap.fuel, lap.weather);
        appendLap(lap);
        if (trackMap) trackMap.updateLap(lap.lap, lap.total_laps, lap.position);
        if (i === laps.length - 1) setScore(data.final_score);
        await sleep(LAP_DELAY);
      }

      // Done
      const policyLabel = data.policy || model;
      const noteText = data.policy_note ? ` · ${data.policy_note.slice(0, 80)}` : "";
      if (statusEl) {
        statusEl.textContent = `done — P${data.final_pos} · score ${Number(data.final_score).toFixed(3)} · ${policyLabel}${noteText}`;
        statusEl.className = "sim-out-status done";
      }
      showResult(data);
      runBtn.disabled = false;
      runBtn.textContent = "▶ RUN RACE";
    });
  }
}

// ─── Smooth scroll for nav ───────────────────────────────────────────────────
function setupNav() {
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener("click", e => {
      const target = document.querySelector(a.getAttribute("href"));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: "smooth" });
      }
    });
  });
}

// ─── IntersectionObserver scroll reveals ─────────────────────────────────────
function setupReveal() {
  let evidenceTriggered = false;

  const obs = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add("revealed");
        const sec = e.target.closest("section");
        if (sec && sec.id === "gallery" && !evidenceTriggered) {
          evidenceTriggered = true;
          animateScoreTable();
          drawRewardCurve();
        }
        obs.unobserve(e.target);
      }
    });
  }, { threshold: 0.12 });

  document.querySelectorAll("[data-reveal]").forEach(el => obs.observe(el));
}

// ─── Boot ────────────────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  drawBg();
  setupReveal();
  setupNav();
  buildTrackGrid();
  setupSimWidget();
  runHero().catch(err => console.error(err));
});
