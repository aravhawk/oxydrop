// Numerical parity: JS engine vs Python reference (python/tiny_chain.py). Usage: node tests/parity.mjs <ref.json> <ppg-csv-dir> [recIds...]
import fs from 'node:fs';
import { runBatch } from '../app/js/engine.js';
const [refPath, dir, ...ids] = process.argv.slice(2);
const ref = JSON.parse(fs.readFileSync(refPath)), W = JSON.parse(fs.readFileSync(new URL('../app/data/weights.json', import.meta.url)));
const recs = ids.length ? ids : Object.keys(ref);
let worst = 0, mism = 0, tot = 0, fail = false;
for (const id of recs) {
  const side = id.endsWith('L') ? 'Left' : 'Right';
  const rows = fs.readFileSync(`${dir}/${side}/${id.slice(0, -1)}.csv`, 'utf8').trim().split('\n').slice(1).map((l) => l.split(',').map(Number));
  const t0 = performance.now(), out = runBatch(rows, W), ms = performance.now() - t0, R = ref[id];
  if (out.length !== R.p.length) { console.log(id, 'LENGTH MISMATCH', out.length, R.p.length); fail = true; continue; }
  let mx = 0, mm = 0;
  out.forEach((d, i) => { mx = Math.max(mx, Math.abs(d.p - R.p[i])); const lab = d.label; if (lab !== R.d[i]) mm++; if (d.s !== R.t[i]) fail = true; });
  worst = Math.max(worst, mx); mism += mm; tot += out.length;
  console.log(`${id}: ${out.length} windows  max|dp|=${mx.toExponential(2)}  label mismatches=${mm}  (${(ms / out.length).toFixed(2)} ms/decision)`);
}
console.log(`TOTAL windows=${tot} worst |dp|=${worst.toExponential(2)} label mismatches=${mism}`);
if (fail || worst > 1e-6 || mism > 0) { console.log('PARITY FAIL'); process.exit(1); } else console.log('PARITY PASS');
