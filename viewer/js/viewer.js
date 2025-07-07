// viewer.js — safe version with event listener null-checks + description/metadata toggle

let panZoomInstance;
let svgElement = null;
const unitsToMicrons = 1;

const design = new URLSearchParams(location.search).get('design') || 'FXL250703-UTA003';
const basePath = `designs/${design}/`;
let colorMode = 'depth';

function loadSVG() {
  const svgFile = colorMode === 'depth'
    ? `${basePath}micromodel_scaled_depth.svg`
    : `${basePath}micromodel_scaled_width.svg`;

  fetch(svgFile)
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
}

function loadLegend() {
  const legendBox = document.getElementById('legend-box');
  const legendContainer = document.getElementById('legend-entries');
  if (!legendBox || !legendContainer) return;

  legendContainer.innerHTML = 'Loading…';
  legendBox.querySelector('strong').textContent =
    colorMode === 'depth' ? 'Edge Depth' : 'Channel Width';

  const mapFile = colorMode === 'depth'
    ? `${basePath}depth_to_color_map.json`
    : `${basePath}width_to_color_map.json`;

  fetch(mapFile)
    .then(r => r.json())
    .then(map => {
      legendContainer.innerHTML = "";
      Object.entries(map)
        .map(([k, v]) => ({ key: parseFloat(k), color: v }))
        .sort((a, b) => a.key - b.key)
        .forEach(entry => {
          const div = document.createElement('div');
          div.className = 'legend-entry';
          div.style.setProperty('--legend-color', entry.color);
          div.textContent = colorMode === 'depth'
            ? `${entry.key} nm`
            : `${entry.key.toFixed(2)} µm`;
          legendContainer.appendChild(div);
        });
    })
    .catch(() => {
      legendContainer.innerHTML = '⚠️ Legend load failed';
    });
}

function loadMetadata() {
  const box = document.getElementById('metadata-box');
  fetch(`${basePath}micromodel_metadata.json`)
    .then(response => response.json())
    .then(data => {
      const safeMetric = (value, unit = "") => {
        return (typeof value === "number")
          ? `${value.toExponential(2)} ${unit}`
          : (value === "N.A." ? "N.A." : "–");
      };
      box.innerHTML = `
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
    })
    .catch(() => {
      box.innerHTML = '⚠️ Failed to load metadata.';
    });
}

function loadDescription() {
  fetch(`${basePath}description.md`)
    .then(response => {
      if (!response.ok) {
        document.getElementById('description-box').style.display = 'none';
        throw new Error('Description not found');
      }
      return response.text();
    })
    .then(markdown => {
      const html = marked.parse(markdown);
      document.getElementById('design-description').innerHTML = html;
      document.getElementById('description-box').style.display = '';
    })
    .catch(() => {
      document.getElementById('description-box').style.display = 'none';
    });
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

function zoomIn() {
  if (panZoomInstance) panZoomInstance.zoomBy(1.2);
  updateScaleBar();
}

function zoomOut() {
  if (panZoomInstance) panZoomInstance.zoomBy(0.8);
  updateScaleBar();
}

function fitToScreen() {
  if (panZoomInstance) {
    panZoomInstance.fit();
    panZoomInstance.center();
    setTimeout(updateScaleBar, 100);
  }
}

// Export zoom functions globally for onclick="..." support
window.zoomIn = zoomIn;
window.zoomOut = zoomOut;
window.fitToScreen = fitToScreen;

document.addEventListener('DOMContentLoaded', () => {
  loadSVG();
  loadLegend();
  loadMetadata();
  loadDescription();

  const colorToggle = document.getElementById('color-toggle');
  if (colorToggle) {
    colorToggle.addEventListener('click', () => {
      colorMode = (colorMode === 'depth') ? 'width' : 'depth';
      loadSVG();
      loadLegend();
    });
  }

  const settingsToggle = document.getElementById('settings-toggle');
  const settingsPanel = document.getElementById('settings-panel');
  if (settingsToggle && settingsPanel) {
    settingsToggle.addEventListener('click', () => {
      settingsPanel.classList.toggle('visible');
    });
  }

  const toggleInfo = document.getElementById('info-toggle');
  if (toggleInfo) {
    toggleInfo.addEventListener('click', () => {
      document.getElementById('metadata-box').classList.toggle('hidden');
      document.getElementById('description-box').classList.toggle('hidden');
    });
  }

  const colorPicker = document.getElementById('background-color-picker');
  if (colorPicker) {
    colorPicker.addEventListener('input', (e) => {
      document.getElementById('svg-container').style.backgroundColor = e.target.value;
    });
  }
});
