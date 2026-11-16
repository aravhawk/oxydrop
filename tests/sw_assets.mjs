// The service worker must precache everything the hybrid needs offline: every file in app/js, the three CNN seed files, the chain/meta/conformal parameters and the replay demo,
// and every listed file must exist (a missing file makes cache.addAll() reject, so the PWA would never install). Usage: node tests/sw_assets.mjs
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const APP = fileURLToPath(new URL('../app/', import.meta.url));
const src = fs.readFileSync(path.join(APP, 'sw.js'), 'utf8');
const list = JSON.parse(src.match(/const ASSETS = (\[[\s\S]*?\]);/)[1].replace(/'/g, '"'));
let fail = 0; const check = (c, m) => { if (!c) { console.log('FAIL', m); fail++; } };
for (const a of list) if (a !== './') check(fs.existsSync(path.join(APP, a)), `listed but missing: ${a}`);
for (const f of fs.readdirSync(path.join(APP, 'js'))) check(list.includes('js/' + f), `js/${f} is not precached`);
const hw = JSON.parse(fs.readFileSync(path.join(APP, 'data/hybrid_weights.json')));
for (const m of hw.members) check(list.includes('data/' + m.weights), `${m.weights} is not precached`);
for (const f of ['data/distill.json', 'data/hybrid_weights.json', 'data/conformal.json', 'data/weights.json']) check(list.includes(f), `${f} is not precached`);
check(/const V = 'oxydrop-v3/.test(src), 'cache version must be bumped so the deployed PWA picks up the new files');
check(/const V = 'oxydrop-v3\.5-photon-distill'/.test(src), 'cache version must name the PhotonDistill build (oxydrop-v3.5-photon-distill) so installed PWAs drop the old cache');
const photonModel = fs.readFileSync(path.join(APP, 'js/photon_engine.js'), 'utf8').match(/DEFAULT_PHOTON_MODEL = '([^']+)'/)[1];
check(list.includes(`data/${photonModel}.json`), `the photon slot's default model file data/${photonModel}.json is not precached`);
const appJs = fs.readFileSync(path.join(APP, 'js/app.js'), 'utf8'), V = src.match(/^const V = '([^']*)'/)[1];
check(appJs.includes(`capture build ${V}'`), `the debug build string in app/js/app.js must equal the cache version ${V}`);
check(!list.some((a) => a.includes('cnn_weights')), 'the removed placeholder must not be listed');
if (fail) { console.log('SW ASSETS FAIL'); process.exit(1); }
console.log(`SW ASSETS PASS (${list.length} files, all present, every js file and the 3 CNN seed files precached)`);
