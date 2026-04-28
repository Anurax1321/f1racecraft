# Track Simulator — Integration Patches

Three small edits. The track-map.js file is self-contained (auto-injects its own CSS), so you only need to wire it into two existing files.

---

## Patch 1 — `server/static/track-map.js` (NEW FILE)

Save the provided `track-map.js` as `server/static/track-map.js`. No changes needed to its contents.

---

## Patch 2 — `server/static/index.html`

### 2a) Add the script tag

Find the existing line that loads `app.js`. It looks like:

```html
<script src="/static/app.js"></script>
```

Add a new line **before** it:

```html
<script src="/static/track-map.js"></script>
<script src="/static/app.js"></script>
```

### 2b) Wrap the lap log + add a track panel

Find the `<div id="sim-output" class="sim-output hidden">` block. Inside it, locate this section:

```html
<!-- Lap log -->
<div class="sim-log-wrap">
  <div class="sim-log-label">DECISION LOG</div>
  <ol id="sim-log" class="sim-log"></ol>
</div>
```

Replace it with this two-column body:

```html
<!-- Lap log + race map (two columns) -->
<div class="sim-output-body">
  <div class="sim-log-wrap">
    <div class="sim-log-label">DECISION LOG</div>
    <ol id="sim-log" class="sim-log"></ol>
  </div>
  <div class="sim-track-wrap">
    <div class="sim-log-label">RACE MAP</div>
    <svg id="track-svg" class="sim-track-svg"></svg>
  </div>
</div>
```

That's the only HTML change.

---

## Patch 3 — `server/static/app.js`

Inside the existing `setupSimWidget()` function, three small additions.

### 3a) Initialise the track map (after the existing `const dimsEl = ...` line)

Find this block near the top of `setupSimWidget`:

```javascript
const logEl       = document.getElementById("sim-log");
const resultEl    = document.getElementById("sim-result");
const dimsEl      = document.getElementById("sim-dims");

if (!scenarioSel) return;
```

Right after `if (!scenarioSel) return;`, add:

```javascript
// Track map
const trackSvg = document.getElementById("track-svg");
const trackMap = trackSvg && window.TrackMap ? new window.TrackMap(trackSvg) : null;
if (trackMap) trackMap.setScenario(scenarioSel.value);
```

### 3b) Refresh the map when the scenario changes

Find this existing block:

```javascript
scenarioSel.addEventListener("change", updateCmd);
seedSel.addEventListener("change", updateCmd);
updateCmd();
```

Replace with:

```javascript
scenarioSel.addEventListener("change", () => {
  updateCmd();
  if (trackMap) trackMap.setScenario(scenarioSel.value);
});
seedSel.addEventListener("change", updateCmd);
updateCmd();
```

### 3c) Update the car position inside the lap-animation loop

Find the lap-animation `for` loop:

```javascript
// Animate through laps
const laps = data.laps;
const LAP_DELAY = 320; // ms per lap
for (let i = 0; i < laps.length; i++) {
  const lap = laps[i];
  setTelem(lap.lap, lap.total_laps, lap.position, lap.compound, lap.health, lap.fuel, lap.weather);
  appendLap(lap);
  if (i === laps.length - 1) setScore(data.final_score);
  await sleep(LAP_DELAY);
}
```

Replace with:

```javascript
// Animate through laps
const laps = data.laps;
const LAP_DELAY = 320; // ms per lap
if (trackMap) trackMap.reset();
for (let i = 0; i < laps.length; i++) {
  const lap = laps[i];
  setTelem(lap.lap, lap.total_laps, lap.position, lap.compound, lap.health, lap.fuel, lap.weather);
  appendLap(lap);
  if (trackMap) trackMap.updateLap(lap.lap, lap.total_laps, lap.position);
  if (i === laps.length - 1) setScore(data.final_score);
  await sleep(LAP_DELAY);
}
```

That's all of it.

---

## Verify locally before pushing

If you have a way to run the FastAPI server locally:

```powershell
python -m server.app
# Then open http://localhost:8000/ in a browser
```

Click **RUN RACE**. You should see:

- Decision log animating on the **left**
- Track map with the red car circling the loop on the **right**
- A start/finish marker at the top of the track
- Corner labels in faint grey text
- Track name + lap progress shown below the SVG

If the track map doesn't appear:

1. Open browser devtools (F12) → Console
2. Type `window.TrackMap` — should not be `undefined`
3. Type `document.getElementById("track-svg")` — should not be `null`
4. If either is broken, reload and check the Network tab for `/static/track-map.js` — should be 200 OK

If the car isn't moving:

1. In Console, type `window.SCENARIO_TO_TRACK` — should print an object
2. Confirm the lap-animation loop is calling `trackMap.updateLap(...)` — add a `console.log("updateLap", lap.lap)` inside the loop temporarily

---

## Push to HF Space

```powershell
git add server/static/track-map.js server/static/index.html server/static/app.js
git commit -m "feat: add live track map + animated car to race simulator widget"
git push origin main

$env:HF_TOKEN = "PASTE_FRESH_TOKEN"
python push_to_space.py "feat: live track map"
```

Wait ~3 min for the Space to rebuild. Then open the live URL in incognito and click RUN RACE. The map should appear next to the decision log on the live site.

If the live version doesn't show the map, check the Space build logs — the most likely failure is a typo in the HTML patch creating broken markup.
