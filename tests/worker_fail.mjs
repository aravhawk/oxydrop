// A CNN Web Worker that dies or stops answering after it has loaded must not leave the measurement waiting forever (HybridWorkerEngine, app/js/hybrid_async.js).
// A stand-in worker never answers. After the calibration minute the engine holds windows it is still waiting for; oldestPendingMs() reports their age
// (the page watchdog in app.js switches to the logistic chain once it exceeds 10 s), and degrade() then makes push() return the logistic chain's own decisions,
// identical to a plain Chain on the same samples, and drops the pending windows. The user-visible notice is covered in tests/flow.mjs (FLOW_WORKER_FAIL=1).
// Usage: node tests/worker_fail.mjs
import fs from 'node:fs';
import assert from 'node:assert/strict';
import { Chain, FS, CAL, WIN } from '../app/js/engine.js';
import { HybridWorkerEngine } from '../app/js/hybrid_async.js';

const rd = (p) => JSON.parse(fs.readFileSync(new URL('../app/' + p, import.meta.url)));
const lite = { hybrid: rd('data/hybrid_weights.json'), conformal: rd('data/conformal.json'), v2Weights: rd('data/weights.json') };
const sent = [];
const fakeWorker = { w: { postMessage: (m) => sent.push(m) }, reset: () => 1, handler: null, info: {} };

let skew = 0; const realNow = performance.now.bind(performance); performance.now = () => realNow() + skew;   // a controllable clock
const sample = (i) => { const x = 2 * Math.PI * 1.2 * (i / FS), w = Math.sin(x) + 0.5 * Math.sin(2 * x + 1); return [200 * (1 + 0.006 * w), 80 * (1 + 0.010 * w), 60 * (1 + 0.008 * w)]; };

const delivered = [];
const eng = new HybridWorkerEngine(lite, fakeWorker, 0.1, (d) => delivered.push(d));
const ref = new Chain(lite.v2Weights);
const total = (CAL + WIN + 30) * FS;
let i = 0;
for (; i < total && eng.inFlight < 3; i++) { const [r, g, b] = sample(i); eng.push(r, g, b); ref.push(r, g, b); }
assert.ok(eng.inFlight >= 1, 'windows after the calibration minute wait for the worker');
assert.equal(delivered.length, 0, 'the stand-in worker never answers');
skew += 11000;
assert.ok(eng.oldestPendingMs() > 10000, 'a window the worker never answers ages (the page watchdog trips at 10 s)');
console.log(`pending ${eng.inFlight} window(s), oldest ${(eng.oldestPendingMs() / 1000).toFixed(1)} s`);

eng.degrade();
assert.equal(eng.inFlight, 0, 'pending windows are dropped');
assert.equal(eng.oldestPendingMs(), 0);
const nSent = sent.length; let n = 0, same = 0;
for (let k = 0; k < 10 * FS; k++, i++) {
  const [r, g, b] = sample(i); const d = eng.push(r, g, b), e = ref.push(r, g, b);
  assert.equal(!!d, !!e, 'a window ends on the same sample as in the logistic chain');
  if (d) { n++; assert.equal(d.p, e.p); assert.equal(d.valid, e.valid); same++; }
}
assert.ok(n >= 3, 'the degraded engine keeps producing windows');
assert.equal(sent.length, nSent, 'nothing more is sent to the dead worker');
eng.degrade();   // idempotent
console.log(`degraded: ${same} windows identical to the logistic chain, nothing sent to the worker`);
console.log('WORKER FAIL PASS');
process.exit(0);
