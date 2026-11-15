import assert from 'node:assert/strict';
import { CameraSource, selectROI, regionStats, pickDefault } from '../app/js/camera.js';

const pixels = new Uint8ClampedArray(32 * 24 * 4);
for (let y = 8; y < 16; y++) for (let x = 16; x < 24; x++) {
  const i = 4 * (y * 32 + x); pixels.set([255, 100, 25, 255], i);
}
const roi = selectROI(pixels);
assert.deepEqual(roi, { x: 16, y: 8, w: 8, h: 8 }, 'find illuminated patch instead of diluting in black frame');
const stats = regionStats(pixels, 32, roi);
assert.deepEqual(stats.mean, [255, 100, 25]);
assert.deepEqual(stats.channelSat, [1, 0, 0]);
assert.equal(stats.satFrac, 0, 'red clipping must leave green/blue HR available');
const allWhite = new Uint8ClampedArray(32 * 24 * 4).fill(255);
assert.equal(regionStats(allWhite, 32, roi).satFrac, 1);
assert.equal(pickDefault([{ id: 'virtual', label: 'Back Triple Camera' }, { id: 'physical', label: 'Back Ultra Wide Camera' }]).id, 'physical');

const cam = new CameraSource(() => {});
cam.caps = {}; let applied;
cam.track = { applyConstraints: async c => { applied = c; }, getSettings: () => ({}) };
assert.equal(await cam.setTorch(true), true);
assert.equal(applied.advanced[0].torch, true, 'attempt torch when WebKit omits capabilities');
assert.equal(cam.info.torchState, 'on-unverified', 'never claim unknown LED state is confirmed');
cam.track.getSettings = () => ({ torch: false });
assert.equal(await cam.setTorch(true), false, 'readback false catches silently ignored constraints');
assert.equal(cam.info.torchState, 'failed');
cam.track.getSettings = () => ({ torch: true });
assert.equal(await cam.setTorch(true), true);
assert.equal(cam.info.torchState, 'confirmed');
cam.track.applyConstraints = async () => { throw new Error('unsupported'); };
assert.equal(await cam.setTorch(true), false);
assert.equal(cam.info.torchState, 'failed');

// Safari's animation callback may run at 60 Hz while camera delivers at 30 Hz.
// Never feed duplicate camera images to the resampler or report 60 camera fps.
const frameCam = new CameraSource(() => {});
frameCam.video = { videoWidth: 640, currentTime: 1 };
frameCam.ctx = { drawImage() {}, getImageData: () => ({ data: pixels }) };
frameCam.frame(1000, null); frameCam.frame(1016, null);
assert.equal(frameCam.frames, 1);
frameCam.video.currentTime += 1 / 30; frameCam.frame(1033, null);
assert.equal(frameCam.frames, 2);
assert.deepEqual(frameCam.stat.fullMean, [255 / 12, 100 / 12, 25 / 12]);
assert.deepEqual(frameCam.stat.mean, [255, 100, 25]);
frameCam.frame(1050, { presentedFrames: 7, mediaTime: 2 });
frameCam.frame(1066, { presentedFrames: 7, mediaTime: 2.01 });
assert.equal(frameCam.frames, 3);
console.log('CAMERA ROI / TORCH / FRAME DEDUP OK');

// WebKit can freeze HTMLVideoElement.currentTime/mediaTime while genuine native
// callbacks advance. presentedFrames, when available, is the authoritative counter.
const nativeCam = new CameraSource(() => {});
nativeCam.video = { videoWidth: 640, videoHeight: 480, currentTime: 0, readyState: 4 };
nativeCam.ctx = { drawImage() {}, getImageData: () => ({ data: pixels }) };
nativeCam.frame(2000, { presentedFrames: 1, mediaTime: 0 });
nativeCam.frame(2033, { presentedFrames: 2, mediaTime: 0 });
nativeCam.frame(2066, { presentedFrames: 3, mediaTime: 0 });
assert.equal(nativeCam.frames, 3, 'advancing native counter must override frozen Safari mediaTime/currentTime');
nativeCam.frame(2080, { presentedFrames: 3, mediaTime: .1 });
assert.equal(nativeCam.frames, 3, 'same native counter remains a duplicate even if mediaTime moves');
// First readable frame can lag the callback; one drawImage error cannot kill the loop.
let transientFailure = true;
nativeCam.ctx.drawImage = () => { if (transientFailure) { transientFailure = false; throw new Error('video not readable yet'); } };
assert.doesNotThrow(() => nativeCam.frame(2100, { presentedFrames: 4, mediaTime: 0 }));
assert.doesNotThrow(() => nativeCam.frame(2133, { presentedFrames: 5, mediaTime: 0 }));
assert.equal(nativeCam.frames, 4, 'capture resumes after a transient unreadable native frame');
console.log('SAFARI NATIVE FRAME PROGRESSION / TRANSIENT READABILITY OK');

// Switching from watchdog media-time dedup to native presentedFrames must not
// reject the first native callback merely because Safari keeps mediaTime at zero.
const sourceSwitch = new CameraSource(() => {});
sourceSwitch.video = { videoWidth: 640, videoHeight: 480, readyState: 4, currentTime: 0 };
sourceSwitch.ctx = { drawImage() {}, getImageData: () => ({ data: pixels }) };
sourceSwitch.frame(3000, null);
sourceSwitch.frame(3034, { captureTime: 0, mediaTime: 0, presentedFrames: 1 });
sourceSwitch.frame(3068, { captureTime: 0, mediaTime: 0, presentedFrames: 2 });
sourceSwitch.frame(3102, { captureTime: 0, mediaTime: 0, presentedFrames: 3 });
assert.equal(sourceSwitch.frames, 4, 'authoritative native counter takes over from fallback despite frozen time metadata');
assert.ok(sourceSwitch.rs.n >= 2, 'finite but constant captureTime=0 must not stall resampling');
const zeroCaptureTime = new CameraSource(() => {});
zeroCaptureTime.video = sourceSwitch.video; zeroCaptureTime.ctx = sourceSwitch.ctx;
for (let i = 0; i < 6; i++) zeroCaptureTime.frame(4000 + i * 34, { captureTime: 0, mediaTime: i / 30, presentedFrames: i + 1 });
assert.equal(zeroCaptureTime.frames, 6);
assert.ok(zeroCaptureTime.rs.n >= 4, 'advancing mediaTime and presentedFrames produce samples when captureTime stays zero');
console.log('SAFARI ZERO CAPTURE TIME / COUNTER SOURCE SWITCH OK');
