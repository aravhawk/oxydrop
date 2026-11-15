import { estimateHR } from './hr_independent.js';

const video = document.querySelector('#preview');
const cameras = document.querySelector('#camera');
const torch = document.querySelector('#torch');
const enable = document.querySelector('#enable');
const measure = document.querySelector('#measure');
const stop = document.querySelector('#stop');
const status = document.querySelector('#status');
const result = document.querySelector('#result');
const settings = document.querySelector('#settings');
const canvas = document.createElement('canvas');
canvas.width = 48;
canvas.height = 48;
const pixels = canvas.getContext('2d', { willReadFrequently: true });

let stream;
let operation;
let selectedCamera = '';

function releaseCamera() {
  stream?.getTracks().forEach((track) => track.stop());
  stream = undefined;
  video.srcObject = null;
}

function updateControls() {
  enable.disabled = Boolean(operation || stream);
  measure.disabled = Boolean(operation || !stream);
  stop.disabled = !operation && !stream;
  cameras.disabled = Boolean(operation || !stream || cameras.options.length < 2);
  torch.disabled = Boolean(operation);
}

async function run(action) {
  if (operation) return;
  const controller = new AbortController();
  operation = controller;
  updateControls();
  try {
    await action(controller.signal);
  } catch (error) {
    if (!controller.signal.aborted) status.textContent = error.message;
    releaseCamera();
  } finally {
    operation = undefined;
    updateControls();
  }
}

function stopCamera() {
  operation?.abort();
  releaseCamera();
  status.textContent = 'Camera stopped.';
  updateControls();
}

async function openCamera(deviceId, signal) {
  releaseCamera();
  const next = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      ...(deviceId ? { deviceId: { exact: deviceId } } : { facingMode: { ideal: 'environment' } }),
      width: { ideal: 640 },
      height: { ideal: 480 },
    },
  });
  if (signal.aborted) {
    next.getTracks().forEach((track) => track.stop());
    signal.throwIfAborted();
  }
  stream = next;
  video.srcObject = stream;
  await video.play();
  signal.throwIfAborted();
  return stream.getVideoTracks()[0];
}

function cameraRank(device) {
  const label = device.label.toLowerCase();
  const capabilities = device.getCapabilities?.() ?? {};
  const facing = capabilities.facingMode ?? [];
  if (facing.includes('user') || /front|facetime/.test(label)) return -1;
  if (!facing.includes('environment') && !/rear|back|environment|world/.test(label)) return 0;
  if (/ultra|telephoto/.test(label)) return 1;
  if (/dual|triple|virtual/.test(label)) return 2;
  return 3;
}

async function prepareCamera(signal) {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error('Open this page over HTTPS or localhost.');
  status.textContent = 'Opening camera…';
  let track = await openCamera(selectedCamera, signal);
  const devices = await navigator.mediaDevices.enumerateDevices();
  signal.throwIfAborted();
  const inputs = devices.filter((device) => device.kind === 'videoinput' && device.deviceId);
  const rear = inputs.filter((device) => cameraRank(device) > 0);
  // Lens-to-flash distance is absent from MediaDevices. Prefer a named physical
  // rear wide camera and let the operator override the choice with the preview.
  const options = (rear.length ? rear : inputs.filter((device) => cameraRank(device) >= 0))
    .sort((a, b) => cameraRank(b) - cameraRank(a));
  const currentId = track.getSettings().deviceId;
  const preferred = options[0];
  const current = options.find((device) => device.deviceId === currentId);
  if (!selectedCamera && preferred && (!current || cameraRank(preferred) > cameraRank(current))) {
    try {
      track = await openCamera(preferred.deviceId, signal);
    } catch (error) {
      signal.throwIfAborted();
      if (preferred.deviceId === currentId || !currentId) throw error;
      track = await openCamera(currentId, signal);
    }
  }
  if (track.getSettings().facingMode === 'user') throw new Error('A rear camera is required.');
  cameras.replaceChildren(...options.map((device, i) =>
    new Option(device.label || `Camera ${i + 1}`, device.deviceId)));
  if (options.length === 0) cameras.add(new Option(track.label || 'Current camera', ''));
  cameras.value = track.getSettings().deviceId ?? '';
  status.textContent = 'Camera ready.';
}

async function applySetting(track, property, value, fixed = {}) {
  const constraints = track.getConstraints();
  const { advanced = [], ...base } = constraints;
  const proposal = { ...base, advanced: [...advanced, { [property]: value, ...fixed }] };
  try {
    await track.applyConstraints(proposal);
    if (track.getSettings()[property] === value) return true;
    await track.applyConstraints(constraints);
  } catch (error) {
    if (error.name !== 'OverconstrainedError' && error.name !== 'NotSupportedError') throw error;
  }
  return false;
}

async function lockMetering(track, property, controls) {
  const capabilities = track.getCapabilities?.() ?? {};
  const modes = capabilities[property] ?? [];
  const current = track.getSettings();
  const fixed = {};
  for (const control of controls) {
    const range = capabilities[control];
    if (range && current[control] >= range.min && current[control] <= range.max) {
      fixed[control] = current[control];
    }
  }
  if (modes.includes('manual') && Object.keys(fixed).length &&
      await applySetting(track, property, 'manual', fixed)) return 'locked';
  if (modes.includes('single-shot') && await applySetting(track, property, 'single-shot')) return 'single-shot';
  if (modes.includes('manual') && await applySetting(track, property, 'manual')) return 'manual';
  return 'automatic';
}

function delay(ms, signal) {
  signal.throwIfAborted();
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', abort);
      resolve();
    }, ms);
    function abort() {
      clearTimeout(timer);
      reject(signal.reason);
    }
    signal.addEventListener('abort', abort, { once: true });
  });
}

function collectFrames(signal) {
  signal.throwIfAborted();
  return new Promise((resolve, reject) => {
    const frames = [];
    const videoCallbacks = typeof video.requestVideoFrameCallback === 'function';
    let request;
    let previousFrame;
    const timer = setTimeout(() => finish(new Error('Camera frames stopped. Try measuring again.')), 15000);

    function finish(error) {
      clearTimeout(timer);
      if (videoCallbacks) video.cancelVideoFrameCallback(request);
      else cancelAnimationFrame(request);
      signal.removeEventListener('abort', abort);
      if (error) reject(error);
      else resolve(frames);
    }

    function abort() {
      finish(signal.reason);
    }

    function schedule() {
      request = videoCallbacks ? video.requestVideoFrameCallback(onFrame) : requestAnimationFrame(onFrame);
    }

    function onFrame(_now, metadata) {
      try {
        const total = video.getVideoPlaybackQuality?.().totalVideoFrames || video.webkitDecodedFrameCount;
        const frameId = metadata?.presentedFrames ?? (total || video.currentTime);
        if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && frameId !== previousFrame) {
          previousFrame = frameId;
          const timestampMs = Number.isFinite(metadata?.mediaTime) ? 1000 * metadata.mediaTime : performance.now();
          if (frames.length === 0 || timestampMs > frames.at(-1).timestampMs) {
            pixels.drawImage(video, 0, 0, canvas.width, canvas.height);
            const { data } = pixels.getImageData(0, 0, canvas.width, canvas.height);
            let red = 0;
            let green = 0;
            for (let i = 0; i < data.length; i += 4) {
              red += data[i];
              green += data[i + 1];
            }
            const count = data.length / 4;
            frames.push({ timestampMs, red: red / count, green: green / count });
            if (timestampMs - frames[0].timestampMs >= 10000) {
              finish();
              return;
            }
          }
        }
        schedule();
      } catch (error) {
        finish(error);
      }
    }

    signal.addEventListener('abort', abort, { once: true });
    schedule();
  });
}

enable.addEventListener('click', () => run(prepareCamera));
cameras.addEventListener('change', () => {
  selectedCamera = cameras.value;
  run(prepareCamera);
});
stop.addEventListener('click', stopCamera);
window.addEventListener('pagehide', stopCamera);

measure.addEventListener('click', () => run(async (signal) => {
  result.value = '';
  settings.textContent = '';
  status.textContent = 'Hold still for 10 seconds…';
  const track = stream.getVideoTracks()[0];
  const torchCapability = track.getCapabilities?.().torch;
  const canEnableTorch = torchCapability === true ||
    (Array.isArray(torchCapability) && torchCapability.includes(true));
  if (torch.checked && canEnableTorch) await applySetting(track, 'torch', true);
  await delay(800, signal);
  const exposure = await lockMetering(track, 'exposureMode', ['exposureTime', 'iso']);
  const whiteBalance = await lockMetering(track, 'whiteBalanceMode', ['colorTemperature']);
  const light = track.getSettings().torch ? 'on' : torch.checked ? 'unavailable' : 'off';
  settings.textContent = `Torch ${light} · Exposure ${exposure} · White balance ${whiteBalance}`;
  const frames = await collectFrames(signal);
  const { bpm, sqi, method } = estimateHR(frames);
  result.value = `${bpm === null ? method : `${bpm} bpm`} · SQI ${sqi.toFixed(2)}`;
  status.textContent = 'Measurement finished.';
  releaseCamera();
}));
