/* track-map.js — animated F1 track + car for the live race simulator.
 *
 * Drop into server/static/ then add the integration patches in
 * simulator-integration.md.
 *
 * Public API:
 *   const map = new TrackMap(document.getElementById("track-svg"));
 *   map.setScenario("weather_roulette");      // call when user changes scenario
 *   map.reset();                               // call at start of a new race
 *   map.updateLap(lapNum, totalLaps, posInt);  // call inside the lap-animation loop
 *
 * Embedded track polylines are stylized — recognizable shape per track but not
 * a millimeter-accurate reproduction. They look like racetracks. That's the bar.
 */

(function () {
  "use strict";

  // ─────────────────────────────────────────────────────────────────
  // 1. Track polylines (normalized to viewBox 0 0 800 500)
  // ─────────────────────────────────────────────────────────────────

  const TRACK_LIB = {
    Spa: {
      name: "Spa-Francorchamps",
      length_km: 7.004,
      points: [
        [120, 380], [115, 360], [130, 340], [160, 320],
        [200, 295], [250, 265], [310, 230], [370, 195],
        [430, 165], [490, 145], [550, 140], [605, 160],
        [640, 195], [655, 240], [645, 285], [615, 320],
        [575, 345], [525, 365], [465, 380], [400, 395],
        [340, 410], [280, 420], [225, 425], [180, 420],
        [150, 410], [125, 395], [120, 380],
      ],
      corners: [
        { at: 0.02, name: "La Source" },
        { at: 0.12, name: "Eau Rouge" },
        { at: 0.45, name: "Pouhon" },
        { at: 0.85, name: "Bus Stop" },
      ],
    },
    Monza: {
      name: "Monza",
      length_km: 5.793,
      points: [
        [130, 350], [115, 325], [120, 295], [150, 270],
        [210, 255], [285, 250], [370, 250], [450, 250],
        [525, 255], [590, 270], [635, 295], [655, 330],
        [640, 365], [600, 385], [540, 395], [470, 400],
        [390, 400], [330, 415], [300, 440], [320, 460],
        [365, 460], [410, 445], [450, 425], [480, 410],
        [490, 380], [465, 365], [415, 360], [350, 360],
        [280, 360], [210, 360], [155, 365], [130, 350],
      ],
      corners: [
        { at: 0.05, name: "Variante del Rettifilo" },
        { at: 0.30, name: "Curva Grande" },
        { at: 0.55, name: "Lesmos" },
        { at: 0.78, name: "Ascari" },
        { at: 0.92, name: "Parabolica" },
      ],
    },
    Monaco: {
      name: "Monaco",
      length_km: 3.337,
      points: [
        [170, 390], [185, 365], [215, 350], [255, 340],
        [290, 325], [315, 295], [330, 260], [340, 225],
        [345, 190], [335, 165], [310, 150], [275, 150],
        [245, 165], [235, 190], [245, 215], [275, 230],
        [315, 235], [360, 225], [410, 205], [460, 185],
        [505, 160], [535, 135], [535, 110], [505, 95],
        [465, 100], [430, 115], [410, 140], [415, 170],
        [440, 200], [485, 225], [540, 255], [600, 290],
        [645, 325], [640, 360], [605, 385], [555, 400],
        [500, 410], [440, 415], [380, 420], [325, 430],
        [280, 435], [240, 430], [205, 415], [180, 400], [170, 390],
      ],
      corners: [
        { at: 0.05, name: "Ste-Devote" },
        { at: 0.25, name: "Casino" },
        { at: 0.45, name: "Loews" },
        { at: 0.65, name: "Tunnel" },
        { at: 0.88, name: "Rascasse" },
      ],
    },
    Catalunya: {
      name: "Catalunya",
      length_km: 4.657,
      points: [
        [120, 280], [110, 260], [125, 240], [155, 230],
        [195, 230], [235, 240], [270, 255], [300, 275],
        [325, 285], [350, 280], [365, 260], [370, 230],
        [360, 200], [330, 185], [290, 185], [255, 200],
        [225, 220], [195, 245], [170, 275], [155, 310],
        [155, 345], [180, 375], [220, 395], [275, 410],
        [335, 420], [400, 425], [460, 430], [515, 430],
        [565, 420], [600, 395], [615, 360], [605, 325],
        [575, 305], [535, 295], [490, 295], [440, 295],
        [385, 290], [325, 290], [270, 285], [215, 285],
        [170, 285], [135, 285], [120, 280],
      ],
      corners: [
        { at: 0.10, name: "Elf" },
        { at: 0.32, name: "Repsol" },
        { at: 0.55, name: "Würth" },
        { at: 0.85, name: "La Caixa" },
      ],
    },
    Silverstone: {
      name: "Silverstone",
      length_km: 5.891,
      points: [
        [150, 290], [140, 265], [150, 240], [180, 220],
        [220, 215], [260, 225], [290, 245], [305, 275],
        [300, 305], [275, 320], [240, 320], [210, 305],
        [195, 310], [205, 340], [240, 360], [285, 370],
        [335, 370], [385, 365], [435, 365], [485, 370],
        [530, 380], [565, 395], [585, 410], [580, 425],
        [555, 430], [515, 425], [475, 415], [435, 405],
        [395, 395], [355, 380], [320, 365], [285, 350],
        [250, 340], [220, 335], [195, 330], [175, 320],
        [160, 305], [150, 290],
      ],
      corners: [
        { at: 0.10, name: "Abbey" },
        { at: 0.25, name: "Village" },
        { at: 0.55, name: "Maggotts" },
        { at: 0.75, name: "Stowe" },
        { at: 0.92, name: "Vale" },
      ],
    },
    Suzuka: {
      name: "Suzuka",
      length_km: 5.807,
      points: [
        [180, 390], [195, 365], [220, 345], [255, 330],
        [295, 320], [335, 315], [370, 300], [395, 275],
        [410, 245], [410, 215], [395, 190], [365, 175],
        [330, 175], [295, 190], [270, 215], [255, 245],
        [255, 280], [285, 305], [330, 320], [385, 320],
        [445, 305], [505, 290], [560, 270], [605, 240],
        [625, 205], [620, 175], [590, 155], [550, 150],
        [510, 165], [480, 195], [465, 230], [445, 255],
        [415, 280], [380, 305], [340, 325], [300, 340],
        [260, 355], [225, 370], [200, 385], [180, 390],
      ],
      corners: [
        { at: 0.05, name: "T1" },
        { at: 0.20, name: "S Curves" },
        { at: 0.40, name: "Degner" },
        { at: 0.60, name: "Hairpin" },
        { at: 0.80, name: "Spoon" },
        { at: 0.92, name: "130R" },
      ],
    },
  };

  const SCENARIO_TO_TRACK = {
    weather_roulette: "Spa",
    dry_strategy_sprint: "Monza",
    late_safety_car: "Monaco",
    championship_decider: "Catalunya",
    virtual_safety_car_window: "Silverstone",
    tyre_cliff_management: "Suzuka",
  };

  // ─────────────────────────────────────────────────────────────────
  // 2. Auto-inject CSS so the integration is single-file
  // ─────────────────────────────────────────────────────────────────

  const STYLE = `
    .sim-output-body {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-top: 12px;
      align-items: stretch;
    }
    @media (max-width: 860px) {
      .sim-output-body { grid-template-columns: 1fr; }
    }
    .sim-track-wrap {
      background: rgba(0, 0, 0, 0.45);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 6px;
      padding: 12px 14px 10px;
      display: flex;
      flex-direction: column;
      min-height: 280px;
    }
    .sim-track-wrap .sim-log-label {
      letter-spacing: 0.18em;
      font-size: 11px;
      color: #8a8a8a;
      margin-bottom: 8px;
      text-transform: uppercase;
    }
    .sim-track-svg {
      flex: 1 1 auto;
      width: 100%;
      height: auto;
      max-height: 320px;
      display: block;
    }
    .sim-track-meta {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 8px;
      font-size: 11px;
      color: #999;
      letter-spacing: 0.05em;
    }
    .sim-track-meta .track-name-label {
      color: #d4d4d4;
      font-weight: 500;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .sim-track-meta .track-progress {
      color: #e10600;
      font-variant-numeric: tabular-nums;
      font-weight: 600;
    }
    .track-corner-label {
      fill: #6a6a6a;
      font-size: 9px;
      font-family: ui-monospace, "SF Mono", Menlo, monospace;
      letter-spacing: 0.05em;
    }
    .track-car-glow {
      transition: cx 0.32s ease-out, cy 0.32s ease-out;
      filter: blur(2px);
    }
    .track-car {
      transition: cx 0.32s ease-out, cy 0.32s ease-out;
    }
  `;

  function injectStyle() {
    if (document.getElementById("track-map-style")) return;
    const s = document.createElement("style");
    s.id = "track-map-style";
    s.textContent = STYLE;
    document.head.appendChild(s);
  }

  // ─────────────────────────────────────────────────────────────────
  // 3. TrackMap class
  // ─────────────────────────────────────────────────────────────────

  class TrackMap {
    constructor(svgEl) {
      injectStyle();
      this.svg = svgEl;
      this.svg.setAttribute("viewBox", "0 0 800 500");
      this.svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
      this.track = null;
      this.car = null;
      this.glow = null;
      this.cumulative = null;
      this.totalLength = 0;
    }

    setScenario(scenarioKey) {
      const trackName = SCENARIO_TO_TRACK[scenarioKey] || "Spa";
      this.track = TRACK_LIB[trackName];
      if (!this.track) return;
      this._buildArcLengths();
      this._render();
    }

    reset() {
      if (!this.track) return;
      this._setCarT(0);
      this._setProgress(null);
    }

    /**
     * Update the car position based on the lap number.
     * lap is the lap just completed (1..totalLaps).
     */
    updateLap(lap, totalLaps, position) {
      if (!this.track || !this.car) return;
      const t = Math.max(0, Math.min(1, lap / Math.max(1, totalLaps)));
      this._setCarT(t);
      this._setProgress(`L${String(lap).padStart(2, "0")} / ${totalLaps}` +
                        (position ? ` · P${position}` : ""));
    }

    // ── private ───────────────────────────────────────────────────

    _buildArcLengths() {
      const pts = this.track.points;
      this.cumulative = [0];
      for (let i = 1; i < pts.length; i++) {
        const dx = pts[i][0] - pts[i - 1][0];
        const dy = pts[i][1] - pts[i - 1][1];
        this.cumulative.push(this.cumulative[i - 1] + Math.sqrt(dx * dx + dy * dy));
      }
      this.totalLength = this.cumulative[this.cumulative.length - 1];
    }

    _pointAtT(t) {
      const target = t * this.totalLength;
      const pts = this.track.points;
      for (let i = 1; i < this.cumulative.length; i++) {
        if (this.cumulative[i] >= target) {
          const segLen = this.cumulative[i] - this.cumulative[i - 1];
          const segT = segLen > 0 ? (target - this.cumulative[i - 1]) / segLen : 0;
          return [
            pts[i - 1][0] + (pts[i][0] - pts[i - 1][0]) * segT,
            pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * segT,
          ];
        }
      }
      return pts[pts.length - 1];
    }

    _pathD() {
      const pts = this.track.points;
      let d = `M ${pts[0][0]} ${pts[0][1]}`;
      for (let i = 1; i < pts.length; i++) {
        d += ` L ${pts[i][0]} ${pts[i][1]}`;
      }
      return d + " Z";
    }

    _render() {
      const NS = "http://www.w3.org/2000/svg";
      this.svg.innerHTML = "";

      // Outer glow layer (subtle)
      const outerGlow = document.createElementNS(NS, "path");
      outerGlow.setAttribute("d", this._pathD());
      outerGlow.setAttribute("stroke", "rgba(225, 6, 0, 0.05)");
      outerGlow.setAttribute("stroke-width", "32");
      outerGlow.setAttribute("fill", "none");
      outerGlow.setAttribute("stroke-linejoin", "round");
      outerGlow.setAttribute("stroke-linecap", "round");
      this.svg.appendChild(outerGlow);

      // Asphalt
      const asphalt = document.createElementNS(NS, "path");
      asphalt.setAttribute("d", this._pathD());
      asphalt.setAttribute("stroke", "#1f1f1f");
      asphalt.setAttribute("stroke-width", "24");
      asphalt.setAttribute("fill", "none");
      asphalt.setAttribute("stroke-linejoin", "round");
      asphalt.setAttribute("stroke-linecap", "round");
      this.svg.appendChild(asphalt);

      // Track edges
      const edgeOuter = document.createElementNS(NS, "path");
      edgeOuter.setAttribute("d", this._pathD());
      edgeOuter.setAttribute("stroke", "#3a3a3a");
      edgeOuter.setAttribute("stroke-width", "26");
      edgeOuter.setAttribute("fill", "none");
      edgeOuter.setAttribute("stroke-linejoin", "round");
      edgeOuter.setAttribute("stroke-linecap", "round");
      edgeOuter.setAttribute("opacity", "0.5");
      this.svg.insertBefore(edgeOuter, asphalt);

      // Centerline (dashed)
      const centerline = document.createElementNS(NS, "path");
      centerline.setAttribute("d", this._pathD());
      centerline.setAttribute("stroke", "rgba(255, 255, 255, 0.18)");
      centerline.setAttribute("stroke-width", "1.2");
      centerline.setAttribute("stroke-dasharray", "5,7");
      centerline.setAttribute("fill", "none");
      this.svg.appendChild(centerline);

      // Corner labels
      if (this.track.corners) {
        this.track.corners.forEach((c) => {
          const [x, y] = this._pointAtT(c.at);
          const label = document.createElementNS(NS, "text");
          label.setAttribute("x", x);
          label.setAttribute("y", y - 12);
          label.setAttribute("text-anchor", "middle");
          label.setAttribute("class", "track-corner-label");
          label.textContent = c.name;
          this.svg.appendChild(label);
        });
      }

      // Start/finish line marker
      const startPt = this.track.points[0];
      const sl = document.createElementNS(NS, "circle");
      sl.setAttribute("cx", startPt[0]);
      sl.setAttribute("cy", startPt[1]);
      sl.setAttribute("r", 6);
      sl.setAttribute("fill", "#fff");
      sl.setAttribute("stroke", "#000");
      sl.setAttribute("stroke-width", "2");
      this.svg.appendChild(sl);

      const slLabel = document.createElementNS(NS, "text");
      slLabel.setAttribute("x", startPt[0]);
      slLabel.setAttribute("y", startPt[1] + 22);
      slLabel.setAttribute("text-anchor", "middle");
      slLabel.setAttribute("class", "track-corner-label");
      slLabel.textContent = "START";
      this.svg.appendChild(slLabel);

      // Car glow
      this.glow = document.createElementNS(NS, "circle");
      this.glow.setAttribute("cx", startPt[0]);
      this.glow.setAttribute("cy", startPt[1]);
      this.glow.setAttribute("r", 14);
      this.glow.setAttribute("fill", "rgba(225, 6, 0, 0.45)");
      this.glow.setAttribute("class", "track-car-glow");
      this.svg.appendChild(this.glow);

      // Ego car
      this.car = document.createElementNS(NS, "circle");
      this.car.setAttribute("cx", startPt[0]);
      this.car.setAttribute("cy", startPt[1]);
      this.car.setAttribute("r", 7);
      this.car.setAttribute("fill", "#e10600");
      this.car.setAttribute("stroke", "#fff");
      this.car.setAttribute("stroke-width", "1.6");
      this.car.setAttribute("class", "track-car");
      this.svg.appendChild(this.car);

      // Track-name and progress label (in DOM, outside SVG, sibling .sim-track-meta)
      const wrap = this.svg.closest(".sim-track-wrap");
      if (wrap) {
        let meta = wrap.querySelector(".sim-track-meta");
        if (!meta) {
          meta = document.createElement("div");
          meta.className = "sim-track-meta";
          meta.innerHTML =
            `<span class="track-name-label"></span>` +
            `<span class="track-progress">L00</span>`;
          wrap.appendChild(meta);
        }
        const nameEl = meta.querySelector(".track-name-label");
        if (nameEl) nameEl.textContent = `${this.track.name} · ${this.track.length_km} km`;
      }
    }

    _setCarT(t) {
      const [x, y] = this._pointAtT(t);
      if (this.car) {
        this.car.setAttribute("cx", x);
        this.car.setAttribute("cy", y);
      }
      if (this.glow) {
        this.glow.setAttribute("cx", x);
        this.glow.setAttribute("cy", y);
      }
    }

    _setProgress(text) {
      const wrap = this.svg.closest(".sim-track-wrap");
      const el = wrap && wrap.querySelector(".track-progress");
      if (el) el.textContent = text || "—";
    }
  }

  // ─────────────────────────────────────────────────────────────────
  // 4. Expose
  // ─────────────────────────────────────────────────────────────────

  window.TrackMap = TrackMap;
  window.TRACK_LIB = TRACK_LIB;
  window.SCENARIO_TO_TRACK = SCENARIO_TO_TRACK;
})();
