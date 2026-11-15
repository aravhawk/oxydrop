// JS (app/js/hr.js) vs Python (python/oxydrop/hr.py) parity on BUT PPG windows and synthetic edge cases.
// Fixture made by scripts/gen_hr_ref.py. Usage: node tests/hr_parity.mjs [tests/fixtures/hr_parity.json]
import fs from 'node:fs';
import { estimateHr, bestChannelEstimate, GATE } from '../app/js/hr.js';
const fx = JSON.parse(fs.readFileSync(process.argv[2] || new URL('./fixtures/hr_parity.json', import.meta.url)));
const TOL = 1e-6;
let n = 0, worst = 0, bad = 0;
const nameMap = { hr: 'hr', hr_acf: 'hrAcf', sqi: 'sqi', acf: 'acf' };
function cmp(label, js, py) {
  n++;
  for (const [pk, jk] of Object.entries(nameMap)) {
    const a = js[jk], b = py[pk];
    if (b === null) { if (Number.isFinite(a)) { bad++; console.log('MISMATCH', label, pk, 'js', a, 'py nan'); } continue; }
    const rel = Math.abs(a - b) / Math.max(1, Math.abs(b)); worst = Math.max(worst, rel);
    if (!(rel < TOL)) { bad++; console.log('MISMATCH', label, pk, 'js', a, 'py', b); }
  }
  if (js.valid !== py.valid) { bad++; console.log('MISMATCH', label, 'valid js', js.valid, 'py', py.valid); }
}
if (fx.gate.F_LO !== GATE.F_LO || fx.gate.SQI_MIN !== GATE.SQI_MIN || fx.gate.ACF_MIN !== GATE.ACF_MIN || fx.gate.AGREE !== GATE.AGREE) { console.log('gate constants differ between hr.py and hr.js', fx.gate, GATE); process.exit(1); }
fx.but.forEach((w, i) => w.x.forEach((x, c) => cmp(`but#${i}.ch${c}`, estimateHr(x), w.ref[c])));
let chBad = 0;
fx.best.forEach((w, i) => { const r = bestChannelEstimate(w.acs, w.primary); cmp(`best#${i}`, r.est, w.ref); if (r.ch !== w.ch) { chBad++; bad++; console.log('MISMATCH best channel', i, r.ch, w.ch); } });
fx.edge.forEach((e) => cmp(`edge:${e.name}`, estimateHr(e.x), e.ref));
fx.f_lo_override.forEach((e, i) => cmp(`flo${e.f_lo}#${i}`, estimateHr(e.x, 30, { ...GATE, F_LO: e.f_lo }), e.ref));
const edgeValid = fx.edge.filter((e) => e.ref.valid).map((e) => e.name);
console.log(`hr parity: ${n} calls (${fx.but.length * 3} BUT windows, ${fx.best.length} best-channel, ${fx.edge.length} edge cases, ${fx.f_lo_override.length} F_LO overrides); worst relative diff ${worst.toExponential(2)}; best-channel mismatches ${chBad}`);
console.log('edge cases accepted by the gate:', edgeValid.join(', ') || 'none');
console.log(bad ? `HR PARITY FAIL (${bad} mismatches)` : 'HR PARITY PASS');
process.exit(bad ? 1 : 0);
