// A slow/hung CNN must not stall camera samples, queue unbounded work, or erase a
// usable logistic baseline. Exercise deadlines, calibration backlogs and stale replies.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import { Chain, FS, CAL, WIN } from '../app/js/engine.js';
import { HybridWorkerEngine, WORKER_WINDOW_TIMEOUT_MS, MAX_WORKER_PENDING } from '../app/js/hybrid_async.js';

const rd = (p) => JSON.parse(fs.readFileSync(new URL('../app/' + p, import.meta.url)));
const lite = { hybrid: rd('data/hybrid_weights.json'), conformal: rd('data/conformal.json'), v2Weights: rd('data/weights.json') };
let skew = 0; const realNow = performance.now.bind(performance); performance.now = () => realNow() + skew;
const sample = (i) => { const x = 2 * Math.PI * 1.2 * (i / FS), w = Math.sin(x) + 0.5 * Math.sin(2 * x + 1); return [200 * (1 + 0.006 * w), 80 * (1 + 0.010 * w), 60 * (1 + 0.008 * w)]; };
function setup(ackCalibration = true, opts = {}) {
  const sent = [], delivered = [], errors = [];
  let eng;
  const worker = { w: { postMessage(m) { sent.push(m); const p = eng.pending.get(m.id); if (ackCalibration && p.d2.calibrating) eng._msg({ type: 'res', gen: 1, id: m.id, h: { calibrating: true } }); } }, reset: () => 1, handler: null, info: {}, terminated: false, terminate() { this.terminated = true; }, onError: (m) => errors.push(m) };
  eng = new HybridWorkerEngine(lite, worker, 0.1, (d) => delivered.push(d), opts);
  return { eng, worker, sent, delivered, errors };
}
{
  const { eng, worker, sent, delivered, errors } = setup();
  const ref = new Chain(lite.v2Weights), refs = new Map();
  let i = 0;
  for (; i < (CAL + WIN + 10) * FS && eng.inFlight === 0; i++) {
    const rgb = sample(i), d = eng.push(...rgb), e = ref.push(...rgb);
    if (e) refs.set(i + 1, e);
    if (d) assert.equal(d.calibrating, true);
  }
  assert.equal(eng.inFlight, 1, 'first decision awaits CNN without blocking capture');
  const lateId = sent.at(-1).id, lateHandler = worker.handler;
  skew += WORKER_WINDOW_TIMEOUT_MS + 1;
  assert.ok(eng.oldestPendingMs() > WORKER_WINDOW_TIMEOUT_MS);
  eng.push(...sample(i)); ref.push(...sample(i)); i++;
  assert.equal(eng.degraded, true, 'next camera sample automatically enforces the deadline');
  assert.equal(eng.inFlight, 0); assert.equal(worker.terminated, true); assert.equal(errors.length, 1);
  assert.equal(delivered.length, 1, 'queued decision is returned immediately as the logistic fallback');
  assert.equal(delivered[0].p, refs.get(delivered[0].n).p);
  lateHandler({ type: 'res', gen: 1, id: lateId, h: { z: 100 } });
  assert.equal(delivered.length, 1, 'late CNN answers are ignored');
  const nSent = sent.length; let same = 0;
  for (let k = 0; k < 10 * FS; k++, i++) {
    const rgb = sample(i), d = eng.push(...rgb), e = ref.push(...rgb);
    assert.equal(!!d, !!e, 'fallback windows preserve sample timing');
    if (d) { assert.equal(d.p, e.p); assert.equal(d.valid, e.valid); same++; }
  }
  assert.ok(same >= 3); assert.equal(sent.length, nSent);
  eng.degrade(); eng.close();
  console.log(`deadline: queued window flushed; ${same} subsequent windows identical to logistic; stale CNN reply ignored`);
}
{
  const { eng, sent, worker } = setup(false);
  for (let i = 0; i < (WIN + 8) * FS; i++) eng.push(...sample(i));
  assert.equal(sent.length, MAX_WORKER_PENDING, 'even unanswered calibration messages have a bounded queue');
  assert.equal(eng.degraded, true); assert.equal(worker.terminated, true); assert.equal(eng.inFlight, 0);
  eng.close();
  console.log(`calibration backlog: bounded at ${MAX_WORKER_PENDING} messages, capture keeps running`);
}
{
  const { eng, worker } = setup(false, { timeoutMs: 20 });
  for (let i = 0; i < WIN * FS; i++) eng.push(...sample(i));
  await new Promise((r) => setTimeout(r, 40));
  assert.equal(eng.degraded, true, 'timer also handles stalled camera callbacks');
  assert.equal(worker.terminated, true); eng.close();
  console.log('wall-clock timeout: fallback does not depend on another captured frame');
}
console.log('WORKER FAIL PASS');
