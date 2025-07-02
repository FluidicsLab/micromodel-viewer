let panZoomInstance;
let svgElement = null;
const unitsToMicrons = 1;

const design = new URLSearchParams(location.search).get('design') || 'FXL250702-UTA002';
const basePath = `designs/${design}/`;

fetch(`${basePath}depth_to_color_map.json`)
  .then(r => r.json())
  .then(map => {
    const container = document.getElementById('legend-entries');
    container.innerHTML = "";
    Object.keys(map).map(Number).sort((a,b)=>a-b).forEach(depth => {
      const color = map[depth];
      const entry = document.createElement('div');
      entry.className = 'legend-entry';
      entry.style.setProperty('--legend-color', color);
      entry.textContent = `${depth} nm`;
      container.appendChild(entry);
    });
  });

fetch(`${basePath}micromodel_scaled.svg`)
  .then(response => response.text())
  .then(svgText => {
    document.getElementById('svg-container').innerHTML = svgText;
    svgElement = document.querySelector('#svg-container svg');
    panZoomInstance = svgPanZoom(svgElement, {
      zoomEnabled: true,
      controlIconsEnabled: false,
      fit: true,
      center: true,
      minZoom: 0.000001,
      maxZoom: 4000000,
      onZoom: updateScaleBar,
      onPan: updateScaleBar
    });
    setTimeout(updateScaleBar, 100);
  });

function zoomIn() {
  panZoomInstance.zoomBy(1.2);
  updateScaleBar();
}
function zoomOut() {
  panZoomInstance.zoomBy(0.8);
  updateScaleBar();
}
function fitToScreen() {
  panZoomInstance.fit();
  panZoomInstance.center();
  setTimeout(updateScaleBar, 100);
}

function updateScaleBar() {
  if (!svgElement) return;
  const g = svgElement.querySelector('g');
  if (!g) return;
  const transform = g.getScreenCTM();
  if (!transform) return;
  const scale = transform.a;
  const micronsPerPx = unitsToMicrons / scale;
  const desiredPx = 100;
  const realLengthMicrons = micronsPerPx * desiredPx;
  const niceLengthMicrons = roundToNiceUnit(realLengthMicrons);
  const niceLengthPx = niceLengthMicrons / micronsPerPx;
  document.getElementById('scale-line').style.width = `${niceLengthPx}px`;
  document.getElementById('scale-label').textContent = formatMicrons(niceLengthMicrons);
  document.getElementById('zoom-level').textContent = `Zoom: ${scale.toFixed(2)}×`;
}

function roundToNiceUnit(value) {
  const units = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5,
    1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000];
  for (let unit of units) {
    if (value <= unit) return unit;
  }
  return value;
}

function formatMicrons(value) {
  if (value >= 1000) return `${(value / 1000).toFixed(2)} mm`;
  if (value < 1) return `${(value * 1000).toFixed(0)} nm`;
  return `${value.toFixed(1)} µm`;
}

fetch(`${basePath}micromodel_metadata.json`)
  .then(response => response.json())
  .then(data => {
    const md = document.getElementById('metadata-box');
    const safeMetric = (value, unit = "") => {
      return (typeof value === "number")
        ? `${value.toExponential(2)} ${unit}`
        : (value === "N.A." ? "N.A." : "–");
    };
    md.innerHTML = `
      <strong>Draft Design Metadata</strong><br>
      <b>chip_id:</b> ${data.chip_id}<br>
      <b>author:</b> ${data.author}<br>
      <b>generated:</b> ${data.date}<br>
      <b>chip size:</b> ${data.dimensions_mm.chip_width} × ${data.dimensions_mm.chip_height} mm<br>
      <b>design area:</b> ${data.dimensions_mm.design_width} × ${data.dimensions_mm.design_height} mm<br>
      <b>n_points:</b> ${data.settings.n_points}<br>
      ${data.metrics ? `
        <b>porosity:</b> ${typeof data.metrics.porosity === "number" ? data.metrics.porosity.toFixed(3) : "N.A."}<br>
        <b>tortuosity:</b> ${typeof data.metrics.tortuosity === "number" ? data.metrics.tortuosity.toFixed(3) : "N.A."}<br>
        <b>permeability:</b> ${safeMetric(data.metrics.permeability?.value, data.metrics.permeability?.unit)}
      ` : ''}
    `;
  });

fetch(`${basePath}description.md`)
  .then(response => response.text())
  .then(markdown => {
    const html = marked.parse(markdown);
    document.getElementById('design-description').innerHTML = html;
  })
  .catch(error => {
    console.error('Failed to load Markdown:', error);
    document.getElementById('design-description').innerText = '⚠️ Unable to load description.';
  });

// Settings toggle and actions
document.addEventListener('DOMContentLoaded', () => {
  const toggle = document.getElementById('settings-toggle');
  const panel = document.getElementById('settings-panel');
  const colorPicker = document.getElementById('background-color-picker');

  if (toggle && panel) {
    toggle.addEventListener('click', () => {
      panel.classList.toggle('visible');
    });
  }

  if (colorPicker) {
    colorPicker.addEventListener('input', (e) => {
      const color = e.target.value;
      document.getElementById('svg-container').style.backgroundColor = color;
    });
  }
});


// Background color picker
document.getElementById('background-color-picker').addEventListener('input', (e) => {
  const color = e.target.value;
  document.getElementById('svg-container').style.backgroundColor = color;
});
