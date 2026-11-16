// Web Worker that runs the heavy half of the hybrid engine (v1 chain + CNN + meta-learner, HybridCore) so that a CNN pass never delays camera frames.
// Protocol (all messages are plain objects):
//   main -> worker  {type:'init', alpha}                 load hybrid_weights.json, conformal.json and the 3 CNN member files, then reply {type:'ready', info}
//                   {type:'reset', gen}                  start a fresh recording (new HybridCore); results carry this generation number
//                   {type:'win', gen, id, want, samples} the samples (r,g,b,dt as Float64) of one 2 s hop; want = the pulse-lock gate passed (run the CNN)
//   worker -> main  {type:'res', gen, id, h}             h = HybridCore.push() result of the window that this hop completed
//                   {type:'error', message}
import { HybridCore, loadHybridAssets } from './hybrid_engine.js';

let assets = null, core = null, alpha, gen = 0, readyP = null;
const fetchJson = async (rel) => { const r = await fetch(new URL('../' + rel, import.meta.url)); if (!r.ok) throw new Error(`${rel}: HTTP ${r.status}`); return r.json(); };

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      alpha = m.alpha;
      readyP = loadHybridAssets(fetchJson).then((a) => { assets = a; core = new HybridCore(a, { alpha }); self.postMessage({ type: 'ready', info: core.info }); });
      await readyP;
    } else if (m.type === 'reset') {
      await readyP; gen = m.gen; core = new HybridCore(assets, { alpha });
    } else if (m.type === 'win') {
      await readyP;
      if (m.gen !== gen) return;
      const s = new Float64Array(m.samples), n = s.length / 4; let h = null;
      for (let i = 0; i < n; i++) { const r = core.push(s[4 * i], s[4 * i + 1], s[4 * i + 2], s[4 * i + 3], m.want); if (r) h = r; }
      self.postMessage({ type: 'res', gen: m.gen, id: m.id, h });
    }
  } catch (err) { self.postMessage({ type: 'error', message: String((err && err.message) || err) }); }
};
