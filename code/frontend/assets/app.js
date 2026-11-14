const state = {
  result: null,
  baseImage: null,
  masks: {},
  toggles: {
    microaneurysms: true,
    hemorrhages: true,
    hard_exudates: true,
    soft_exudates: true,
    optic_disc: true,
  },
  alpha: 0.45,
  zoom: 1,
  panX: 0,
  panY: 0,
  drag: false,
  dragStartX: 0,
  dragStartY: 0,
};

const lesionColors = {
  microaneurysms: [255, 64, 64],
  hemorrhages: [220, 20, 60],
  hard_exudates: [255, 215, 0],
  soft_exudates: [135, 206, 250],
  optic_disc: [144, 238, 144],
};

const tabs = document.querySelectorAll('.tab-btn');
const tabContents = {
  classification: document.getElementById('tab-classification'),
  segmentation: document.getElementById('tab-segmentation'),
  localization: document.getElementById('tab-localization'),
  explainability: document.getElementById('tab-explainability'),
};

const form = document.getElementById('infer-form');
const imageInput = document.getElementById('image-input');
const previewImage = document.getElementById('preview-image');
const previewPlaceholder = document.getElementById('preview-placeholder');
const statusBox = document.getElementById('status-box');
const inferBtn = document.getElementById('infer-btn');
const reportBtn = document.getElementById('download-report');

const segCanvas = document.getElementById('seg-canvas');
const segCtx = segCanvas.getContext('2d');
const locCanvas = document.getElementById('loc-canvas');
const locCtx = locCanvas.getContext('2d');

function setStatus(message, isError = false) {
  statusBox.textContent = message;
  statusBox.className = isError ? 'mt-4 text-sm text-red-600' : 'mt-4 text-sm text-slate-600';
}

function renderCanvases() {
  renderSegmentation();
  renderLocalization();
}

function setTab(tabName) {
  tabs.forEach((tab) => {
    tab.classList.toggle('active', tab.dataset.tab === tabName);
  });
  Object.entries(tabContents).forEach(([name, el]) => {
    el.classList.toggle('hidden', name !== tabName);
  });
  // Render after visibility changes so canvas containers have final dimensions.
  requestAnimationFrame(renderCanvases);
}

tabs.forEach((tab) => {
  tab.addEventListener('click', () => setTab(tab.dataset.tab));
});

function resizeCanvasToContainer(canvas) {
  const rect = canvas.parentElement.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.floor(rect.width * ratio);
  canvas.height = Math.floor(rect.height * ratio);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function drawBaseCanvas(ctx, canvas, image) {
  if (!image) return;
  resizeCanvasToContainer(canvas);
  const width = canvas.parentElement.clientWidth;
  const height = canvas.parentElement.clientHeight;
  ctx.clearRect(0, 0, width, height);

  const drawW = image.width * state.zoom;
  const drawH = image.height * state.zoom;
  const x = (width - drawW) / 2 + state.panX;
  const y = (height - drawH) / 2 + state.panY;
  ctx.drawImage(image, x, y, drawW, drawH);

  return { x, y, drawW, drawH };
}

function decodeBase64Image(base64) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = `data:image/png;base64,${base64}`;
  });
}

async function prepareMasks(result) {
  const entries = Object.entries(result.segmentation || {});
  const loaded = {};
  for (const [name, value] of entries) {
    if (value.mask_base64) {
      loaded[name] = await decodeBase64Image(value.mask_base64);
    }
  }
  state.masks = loaded;
}

function overlayMask(ctx, maskImg, color, alpha, x, y, drawW, drawH) {
  const tmpCanvas = document.createElement('canvas');
  tmpCanvas.width = drawW;
  tmpCanvas.height = drawH;
  const tctx = tmpCanvas.getContext('2d');

  tctx.drawImage(maskImg, 0, 0, drawW, drawH);
  const imageData = tctx.getImageData(0, 0, drawW, drawH);
  const data = imageData.data;
  for (let i = 0; i < data.length; i += 4) {
    const intensity = data[i] / 255;
    if (intensity > 0.35) {
      data[i] = color[0];
      data[i + 1] = color[1];
      data[i + 2] = color[2];
      data[i + 3] = Math.floor(alpha * 255 * intensity);
    } else {
      data[i + 3] = 0;
    }
  }
  tctx.putImageData(imageData, 0, 0);
  ctx.drawImage(tmpCanvas, x, y);
}

function renderSegmentation() {
  if (!state.baseImage) return;
  const geom = drawBaseCanvas(segCtx, segCanvas, state.baseImage);
  if (!geom) return;

  for (const [name, img] of Object.entries(state.masks)) {
    if (state.toggles[name]) {
      overlayMask(segCtx, img, lesionColors[name], state.alpha, geom.x, geom.y, geom.drawW, geom.drawH);
    }
  }
}

function renderLocalization() {
  if (!state.baseImage || !state.result) return;
  const geom = drawBaseCanvas(locCtx, locCanvas, state.baseImage);
  if (!geom) return;

  const loc = state.result.localization;
  const scaleX = geom.drawW / state.baseImage.width;
  const scaleY = geom.drawH / state.baseImage.height;

  const od = {
    x: geom.x + loc.optic_disc.x * scaleX,
    y: geom.y + loc.optic_disc.y * scaleY,
  };
  const fv = {
    x: geom.x + loc.fovea.x * scaleX,
    y: geom.y + loc.fovea.y * scaleY,
  };

  locCtx.strokeStyle = '#10b981';
  locCtx.fillStyle = '#10b981';
  locCtx.lineWidth = 2;
  locCtx.beginPath();
  locCtx.arc(od.x, od.y, 8, 0, Math.PI * 2);
  locCtx.stroke();
  locCtx.fillText('OD', od.x + 10, od.y - 8);

  locCtx.strokeStyle = '#ef4444';
  locCtx.fillStyle = '#ef4444';
  locCtx.beginPath();
  locCtx.arc(fv.x, fv.y, 8, 0, Math.PI * 2);
  locCtx.stroke();
  locCtx.fillText('Fovea', fv.x + 10, fv.y - 8);
}

function updateProbBars(probabilities) {
  const container = document.getElementById('prob-bars');
  container.innerHTML = '';
  Object.entries(probabilities).forEach(([label, value]) => {
    const row = document.createElement('div');
    row.innerHTML = `
      <div class="flex justify-between text-xs mb-1"><span>${label}</span><span>${(value * 100).toFixed(1)}%</span></div>
      <div class="h-2 bg-slate-200 rounded-full overflow-hidden">
        <div class="h-full bg-cyan-600" style="width:${Math.max(2, value * 100)}%"></div>
      </div>
    `;
    container.appendChild(row);
  });
}

function setupMaskToggles() {
  const container = document.getElementById('mask-toggles');
  container.innerHTML = '';
  Object.keys(state.toggles).forEach((name) => {
    const id = `toggle-${name}`;
    const label = name.replaceAll('_', ' ');
    const wrap = document.createElement('label');
    wrap.className = 'flex items-center gap-2 text-sm bg-slate-50 border border-slate-200 rounded-lg px-2 py-1';
    wrap.innerHTML = `
      <input id="${id}" type="checkbox" ${state.toggles[name] ? 'checked' : ''} />
      <span class="capitalize">${label}</span>
    `;
    container.appendChild(wrap);
    const checkbox = wrap.querySelector('input');
    checkbox.addEventListener('change', () => {
      state.toggles[name] = checkbox.checked;
      renderSegmentation();
    });
  });
}

function attachCanvasInteractions(canvas, renderFn) {
  canvas.addEventListener('mousedown', (e) => {
    state.drag = true;
    state.dragStartX = e.clientX;
    state.dragStartY = e.clientY;
  });

  window.addEventListener('mouseup', () => {
    state.drag = false;
  });

  window.addEventListener('mousemove', (e) => {
    if (!state.drag) return;
    const dx = e.clientX - state.dragStartX;
    const dy = e.clientY - state.dragStartY;
    state.dragStartX = e.clientX;
    state.dragStartY = e.clientY;
    state.panX += dx;
    state.panY += dy;
    renderFn();
  });
}

imageInput.addEventListener('change', () => {
  const file = imageInput.files?.[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  previewImage.src = url;
  previewImage.classList.remove('hidden');
  previewPlaceholder.classList.add('hidden');

  const img = new Image();
  img.onload = () => {
    state.baseImage = img;
    state.zoom = 1;
    state.panX = 0;
    state.panY = 0;
    renderCanvases();
  };
  img.src = url;
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const file = imageInput.files?.[0];
  if (!file) {
    setStatus('Select an image first.', true);
    return;
  }

  inferBtn.disabled = true;
  inferBtn.textContent = 'Running...';
  reportBtn.classList.add('hidden');
  setStatus('Submitting image and waiting for model output...');

  const formData = new FormData();
  formData.append('image', file);
  formData.append('patient_name', form.patient_name.value || '');
  formData.append('patient_id', form.patient_id.value || '');
  formData.append('patient_age', form.patient_age.value || '');
  formData.append('patient_sex', form.patient_sex.value || '');

  try {
    const response = await fetch('/api/infer', { method: 'POST', body: formData });
    const payload = await response.json();
    if (!response.ok || payload.error) {
      throw new Error(payload.details || payload.error || 'Inference failed');
    }

    const result = payload.result;
    state.result = result;

    document.getElementById('grade-label').textContent = result.classification.label;
    document.getElementById('grade-confidence').textContent = `Confidence: ${(result.classification.confidence * 100).toFixed(2)}%`;
    updateProbBars(result.classification.probabilities);

    document.getElementById('od-coord').textContent =
      `x: ${result.localization.optic_disc.x.toFixed(1)}, y: ${result.localization.optic_disc.y.toFixed(1)}, uncertainty: ${result.localization.optic_disc.uncertainty_px.toFixed(2)} px`;
    document.getElementById('fv-coord').textContent =
      `x: ${result.localization.fovea.x.toFixed(1)}, y: ${result.localization.fovea.y.toFixed(1)}, uncertainty: ${result.localization.fovea.uncertainty_px.toFixed(2)} px`;

    document.getElementById('gradcam-image').src = `data:image/png;base64,${result.gradcam.image_base64}`;

    await prepareMasks(result);
    setupMaskToggles();
    renderCanvases();

    if (result.inference_id) {
      reportBtn.classList.remove('hidden');
      reportBtn.onclick = () => {
        window.open(`/api/report/${result.inference_id}`, '_blank');
      };
    }

    setStatus('Inference complete. Review tabs and download the PDF report.');
  } catch (err) {
    console.error(err);
    setStatus(String(err.message || err), true);
  } finally {
    inferBtn.disabled = false;
    inferBtn.textContent = 'Run Inference';
  }
});

document.getElementById('alpha-slider').addEventListener('input', (e) => {
  state.alpha = Number(e.target.value) / 100;
  renderSegmentation();
});

document.getElementById('zoom-in').addEventListener('click', () => {
  state.zoom = Math.min(5, state.zoom * 1.1);
  renderCanvases();
});

document.getElementById('zoom-out').addEventListener('click', () => {
  state.zoom = Math.max(0.2, state.zoom / 1.1);
  renderCanvases();
});

document.getElementById('zoom-reset').addEventListener('click', () => {
  state.zoom = 1;
  state.panX = 0;
  state.panY = 0;
  renderCanvases();
});

attachCanvasInteractions(segCanvas, () => {
  renderCanvases();
});
attachCanvasInteractions(locCanvas, () => {
  renderCanvases();
});

const locZoomIn = document.getElementById('loc-zoom-in');
const locZoomOut = document.getElementById('loc-zoom-out');
const locZoomReset = document.getElementById('loc-zoom-reset');

if (locZoomIn) {
  locZoomIn.addEventListener('click', () => {
    state.zoom = Math.min(5, state.zoom * 1.1);
    renderCanvases();
  });
}

if (locZoomOut) {
  locZoomOut.addEventListener('click', () => {
    state.zoom = Math.max(0.2, state.zoom / 1.1);
    renderCanvases();
  });
}

if (locZoomReset) {
  locZoomReset.addEventListener('click', () => {
    state.zoom = 1;
    state.panX = 0;
    state.panY = 0;
    renderCanvases();
  });
}

window.addEventListener('resize', () => {
  renderCanvases();
});

setTab('classification');
setStatus('Ready. Upload a fundus image to begin.');
