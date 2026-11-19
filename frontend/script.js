'use strict';

/**
 * Frontend-to-backend contract for the upcoming FastAPI implementation:
 * GET  /api/health  -> {"status":"ok", ...}
 * POST /api/predict -> multipart/form-data field "file" (image)
 *   {"label":"Political"|"Non-Political", "confidence":0..1,
 *    "probabilities":{"Political":0..1,"Non-Political":0..1},
 *    "ocr_text":"...", "latency_ms":123.4}
 *
 * Override before loading this script when hosted separately:
 * window.POLIMEME_API_BASE_URL = 'https://your-fastapi-host.example';
 * Never generate fake classification results when the model API is offline.
 */
const API_BASE_URL = (window.POLIMEME_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '');
const MAX_IMAGE_BYTES = 8 * 1024 * 1024;
const ACCEPTED_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp']);
const $ = (id) => document.getElementById(id);

let selectedFile = null;
let previewObjectUrl = null;
let requestController = null;
let busy = false;

function setStatus(online, message) {
  const el = $('connectionStatus');
  el.classList.toggle('online', online);
  el.classList.toggle('offline', !online);
  $('connectionText').textContent = message;
}

async function request(path, options = {}, timeoutMs = 9000) {
  const controller = new AbortController();
  const parentSignal = options.signal;
  const forwardAbort = () => controller.abort();
  if (parentSignal?.aborted) controller.abort();
  else parentSignal?.addEventListener('abort', forwardAbort, { once: true });
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${API_BASE_URL}${path}`, { ...options, signal: controller.signal });
    if (!response.ok) {
      let message = `HTTP ${response.status}`;
      try {
        const body = await response.json();
        if (body.detail) message = typeof body.detail === 'string' ? body.detail : message;
      } catch { /* Keep HTTP status when the response isn't JSON. */ }
      throw new Error(message);
    }
    return await response.json();
  } finally {
    window.clearTimeout(timer);
    parentSignal?.removeEventListener('abort', forwardAbort);
  }
}

async function checkConnection() {
  $('connectionText').textContent = 'Checking inference API…';
  $('connectionStatus').classList.remove('online', 'offline');
  try {
    const result = await request('/api/health', {}, 6000);
    if (result.status !== 'ok') throw new Error('The health endpoint returned an unexpected status.');
    setStatus(true, 'Inference API connected');
    return true;
  } catch {
    setStatus(false, 'Inference API offline · UI preview ready');
    return false;
  }
}

function showNotice(message, isError = false) {
  const el = $('uploadNotice');
  el.textContent = message;
  el.classList.toggle('notice-error', isError);
}

function setBusy(isBusy) {
  busy = isBusy;
  $('analyzeBtn').disabled = isBusy || !selectedFile;
  $('analyzeBtnLabel').textContent = isBusy ? 'Analyzing image…' : 'Analyze meme';
  $('analyzeBtnArrow').classList.toggle('hidden', isBusy);
  $('spinner').classList.toggle('hidden', !isBusy);
  $('resetBtn').disabled = isBusy;
  $('removeFile').disabled = isBusy;
}

function clearResult() {
  $('resultData').classList.add('hidden');
  $('resultError').classList.add('hidden');
  $('resultEmpty').classList.remove('hidden');
}

function clearImage() {
  if (busy) return;
  selectedFile = null;
  $('imageInput').value = '';
  $('imagePreviewWrap').classList.add('hidden');
  $('dropPlaceholder').classList.remove('hidden');
  $('fileRow').classList.add('hidden');
  $('dropZone').classList.remove('has-file');
  $('imagePreview').removeAttribute('src');
  if (previewObjectUrl) URL.revokeObjectURL(previewObjectUrl);
  previewObjectUrl = null;
  $('analyzeBtn').disabled = true;
  showNotice('Select an image to get started. No image is uploaded until you click Analyze.');
  clearResult();
}

function setImage(file) {
  if (busy) return;
  if (!file) return;
  if (!ACCEPTED_TYPES.has(file.type)) {
    showNotice('Unsupported file type. Please choose a JPEG, PNG, or WebP image.', true);
    return;
  }
  if (file.size === 0 || file.size > MAX_IMAGE_BYTES) {
    showNotice('Image must be larger than 0 bytes and no more than 8 MB.', true);
    return;
  }
  if (previewObjectUrl) URL.revokeObjectURL(previewObjectUrl);
  selectedFile = file;
  previewObjectUrl = URL.createObjectURL(file);
  $('imagePreview').src = previewObjectUrl;
  $('imagePreviewWrap').classList.remove('hidden');
  $('dropPlaceholder').classList.add('hidden');
  $('dropZone').classList.add('has-file');
  $('fileRow').classList.remove('hidden');
  $('fileName').textContent = file.name;
  $('fileSize').textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
  $('analyzeBtn').disabled = false;
  showNotice('Image ready. Click Analyze to send it to the inference API.');
  clearResult();
}

function toNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}
function asPercent(value) { return `${(value * 100).toFixed(1)}%`; }
function validProbability(value) { return value !== null && value >= 0 && value <= 1; }

function normalizePrediction(payload) {
  if (!payload || typeof payload !== 'object') throw new Error('Invalid API response. Expected a JSON object.');
  const rawLabel = String(payload.label || '').trim().toLowerCase().replace(/[\s_-]+/g, '');
  if (!['political', 'nonpolitical'].includes(rawLabel)) {
    throw new Error('Invalid API response: the label must be Political or Non-Political.');
  }
  const isPolitical = rawLabel === 'political';
  const probabilities = payload.probabilities || {};
  let political = toNumber(probabilities.Political ?? probabilities.political);
  let nonPolitical = toNumber(probabilities['Non-Political'] ?? probabilities.non_political);
  let confidence = toNumber(payload.confidence);
  if (!validProbability(confidence)) throw new Error('Invalid API response: confidence must be between 0 and 1.');
  // Some inference APIs may supply only their confidence. Don't invent a full distribution.
  if (political === null && nonPolitical === null) {
    political = isPolitical ? confidence : 1 - confidence;
    nonPolitical = isPolitical ? 1 - confidence : confidence;
  } else if (political === null && validProbability(nonPolitical)) {
    political = 1 - nonPolitical;
  } else if (nonPolitical === null && validProbability(political)) {
    nonPolitical = 1 - political;
  }
  if (!validProbability(political) || !validProbability(nonPolitical) || Math.abs(political + nonPolitical - 1) > 0.02) {
    throw new Error('Invalid API response: probability values are inconsistent.');
  }
  const ocr = typeof payload.ocr_text === 'string' ? payload.ocr_text.trim() : '';
  const latency = toNumber(payload.latency_ms);
  return { label: isPolitical ? 'Political' : 'Non-Political', confidence, political, nonPolitical, ocr, latency };
}

function displayPrediction(data) {
  $('resultEmpty').classList.add('hidden');
  $('resultError').classList.add('hidden');
  $('resultData').classList.remove('hidden');
  $('predictionLabel').textContent = data.label;
  $('predictionDescription').textContent = data.label === 'Political'
    ? 'The model detected patterns associated with political content.'
    : 'The model detected patterns associated with non-political content.';
  $('predictionTag').textContent = 'MODEL OUTPUT';
  $('confidenceText').textContent = asPercent(data.confidence);
  $('confidenceMeter').style.width = asPercent(data.confidence);
  $('politicalProb').textContent = asPercent(data.political);
  $('politicalBar').style.width = asPercent(data.political);
  $('nonPoliticalProb').textContent = asPercent(data.nonPolitical);
  $('nonPoliticalBar').style.width = asPercent(data.nonPolitical);
  $('ocrText').textContent = data.ocr || 'No readable text was detected in this image.';
  $('latencyText').textContent = data.latency !== null && data.latency >= 0 ? `${data.latency.toFixed(0)} ms` : '';
}

function displayError(message) {
  $('resultData').classList.add('hidden');
  $('resultEmpty').classList.add('hidden');
  $('resultError').classList.remove('hidden');
  $('errorText').textContent = message;
}

async function analyzeImage() {
  if (!selectedFile || busy) return;
  setBusy(true);
  showNotice('Running multimodal inference…');
  const form = new FormData();
  form.append('file', selectedFile, selectedFile.name);
  // Long timeout accommodates a local CPU model or a cold-started hosting instance.
  const controller = new AbortController();
  requestController = controller;
  try {
    const payload = await request('/api/predict', { method: 'POST', body: form, signal: controller.signal }, 120000);
    displayPrediction(normalizePrediction(payload));
    showNotice('Analysis completed using the live model.');
    setStatus(true, 'Inference API connected');
  } catch (error) {
    const detail = error instanceof Error ? error.message : 'Unknown connection error.';
    const networkHint = error instanceof TypeError
      ? 'Cannot reach the backend. Start FastAPI at http://127.0.0.1:8000 or configure the deployed API URL.'
      : (detail.includes('abort') || detail.includes('Abort'))
        ? 'The prediction request timed out or was cancelled.'
        : detail;
    displayError(networkHint);
    showNotice('No prediction was generated. Check the backend connection.', true);
  } finally {
    requestController = null;
    setBusy(false);
  }
}

function init() {
  $('copyrightYear').textContent = String(new Date().getFullYear());
  $('imageInput').addEventListener('change', (event) => setImage(event.target.files?.[0]));
  $('dropZone').addEventListener('click', () => { if (!busy) $('imageInput').click(); });
  $('dropZone').addEventListener('keydown', (event) => {
    if (event.target === $('dropZone') && (event.key === 'Enter' || event.key === ' ')) {
      event.preventDefault();
      if (!busy) $('imageInput').click();
    }
  });
  ['dragenter', 'dragover'].forEach((type) => $('dropZone').addEventListener(type, (event) => {
    event.preventDefault();
    if (!busy) $('dropZone').classList.add('dragover');
  }));
  ['dragleave', 'drop'].forEach((type) => $('dropZone').addEventListener(type, (event) => {
    event.preventDefault();
    $('dropZone').classList.remove('dragover');
  }));
  $('dropZone').addEventListener('drop', (event) => {
    if (!busy && event.dataTransfer?.files?.length) setImage(event.dataTransfer.files[0]);
  });
  $('removeFile').addEventListener('click', (event) => { event.stopPropagation(); clearImage(); });
  $('resetBtn').addEventListener('click', clearImage);
  $('analyzeBtn').addEventListener('click', analyzeImage);
  $('retryBtn').addEventListener('click', async () => {
    const online = await checkConnection();
    if (online) {
      clearResult();
      showNotice('API connected. Click Analyze to process the selected image.');
    }
  });
  window.addEventListener('beforeunload', () => {
    if (previewObjectUrl) URL.revokeObjectURL(previewObjectUrl);
    if (requestController) requestController.abort();
  });
  void checkConnection();
}

init();
