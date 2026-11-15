// Numerical parity: JS engine vs Python reference. Usage: node tests/parity.mjs <v1|v2> <ref.json> <ppg-csv-dir> [recIds...]
// v1 = original research model (engine_v1.js vs python chain.py); v2 = field-hardened model shipped in the app (engine.js vs chain_v2.py)
import fs from 'node:fs';
const [ver, refPath, dir, ...ids] = process.argv.slice(2);
const { runBatch } = await import(ver === 'v1' ? '../app/js/engine_v1.js' : '../app/js/engine.js');
const ref = JSON.parse(fs.readFileSync(refPath)), W = JSON.parse(fs.readFileSync(new URL(ver === 'v1' ? '../python/tiny_chain_v1_weights.json' : '../app/data/weights.json', import.meta.url)));
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
