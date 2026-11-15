// Guards the wording the independent review (REVIEW.md, PR #3) showed to be wrong or unsupported: these phrases must not come back
// into the app copy, README, docs, DATA_LICENSES.md or results/*.md. Usage: node tests/docs_claims.mjs
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const files = ['README.md', 'DATA_LICENSES.md', 'app/index.html', ...fs.readdirSync(path.join(ROOT, 'docs')).filter((f) => f.endsWith('.md')).map((f) => 'docs/' + f),
  ...fs.readdirSync(path.join(ROOT, 'results')).filter((f) => f.endsWith('.md')).map((f) => 'results/' + f)].filter((f) => f !== 'docs/REVIEW_RESPONSE.md' && f !== 'docs/REVIEW_RESPONSE_ML.md');   // the response table quotes the old wording on purpose
const BAD = [
  [/ratio is not a model input/i, 'the ratio-of-ratios features ARE fusion inputs (four nonzero coefficients)'],
  [/Our own idea/i, 'phone-camera low-oxygen detection is not new (Hoffman 2022, MTHS, Sun 2021, Tian 2022)'],
  [/\+3\.7\b/, 'the gain over the clock is +3.5 points (saved scores +3.47), paired CI [-6.8, +16.7]'],
  [/\bafter 60 seconds\b/i, 'the first answer arrives at 70 s'],
  [/every setting (was )?chosen on the other five/i, 'unqualified: v2 hardcodes C and the band'],
  [/every hyperparameter chosen on the other 5 people/i, 'unqualified: v2 hardcodes C and the band'],
  [/2-minute (phone )?retest/i, 'the completion time is not measured'],
  [/at 3,000 m with no pulse oximeter/i, 'the 3,000 m persona is not evidence'],
  [/\b32 candidate datasets\b|We (searched|checked) 32\b/i, 'no dated inventory exists'],
  [/no OpenOximetry data (or results )?(was used|appear)/i, 'OpenOximetry is now used (aggregates only)'],
  [/OpenOximetry[^.]*\bnot available\b/i, 'OpenOximetry is now used (aggregates only)'],
  [/\b3×10⁻¹³/, 'parity is 2.09e-13 (v2) and 3.18e-13 (v1)'],
  [/\b11\.87\b/, 'the fixed-gate BUT MAE is 11.86'],
  [/It helps the heart rate/i, 'the heart-rate rule is post hoc, BUT PPG driven and exploratory (spectral is better on Hoffman and nominally on MTHS); see README P2'],
  [/the app answers 79\.1%|79\.1% of windows and 87\.0%/i, 'the shipped pipeline R answers 74.5% at 89.5%; 79.1% / 87.0% are the per-person-selected rows'],
];
let fails = 0;
for (const f of files) {
  const lines = fs.readFileSync(path.join(ROOT, f), 'utf8').split('\n');
  lines.forEach((l, i) => { for (const [re, why] of BAD) if (re.test(l) && !/earlier version|was wrong|used to say|an earlier note|older|UNSUPPORTED/i.test(l)) { fails++; console.log(`FAIL ${f}:${i + 1}  /${re.source}/  ${why}`); } });
}
if (fails) { console.log(`DOCS CLAIMS FAIL (${fails})`); process.exit(1); }
console.log(`DOCS CLAIMS PASS (${files.length} files, ${BAD.length} patterns)`);
