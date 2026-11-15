// OxyNet forward pass in dependency-free JavaScript. Pure ES module (no DOM, no Node APIs), runs in the browser and in Node tests.
// Reads a member file app/data/hybrid_R_s<k>.json written by python/oxydrop_ml/export_hybrid.py (BatchNorm already folded into the convs; the graph format is defined by export_js.py).
// Supported ops, exactly the exported set: shared_stem_input, conv1d (stride, dilation, explicit zero padding, optional ReLU),
// add_domain (per-domain embedding add + ReLU), add (residual + ReLU), pool (global average + max), dense (concat of inputs, optional ReLU), sub.
// Activations are Float64Array (weights float32), all buffers are allocated once in loadModel(); forward() allocates nothing.
// Spec: python/oxydrop_ml/MODEL_SPEC.md.

const FORMAT = 'oxynet-v1';

function decodeTensor(t) {
  const n = t.shape.reduce((a, b) => a * b, 1);
  if (t.data) { const f = Float32Array.from(t.data); if (f.length !== n) throw new Error('tensor size mismatch'); return f; }
  const s = atob(t.b64), u = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
  if (u.length !== 4 * n) throw new Error('tensor size mismatch');
  const f = new Float32Array(n), dv = new DataView(u.buffer);   // explicit little-endian read, independent of host endianness
  for (let i = 0; i < n; i++) f[i] = dv.getFloat32(4 * i, true);
  return f;
}

/** Build a model from the parsed JSON. Allocates every buffer. */
export function loadModel(json) {
  if (json.format !== FORMAT) throw new Error(`unsupported model format ${json.format}`);
  const T = json.input.T, cin = json.input.channels.length;
  const tensors = {};
  for (const [k, v] of Object.entries(json.tensors)) tensors[k] = { shape: v.shape, data: decodeTensor(v) };
  const shape = { x: { c: cin, t: T, n: cin * T }, side: { c: 1, t: json.input.side.length, n: json.input.side.length }, emb_base: { c: 1, t: json.input.emb, n: json.input.emb } };
  const buf = { x: new Float64Array(cin * T), side: new Float64Array(json.input.side.length), emb_base: new Float64Array(json.input.emb) };
  const alloc = (name, c, t) => { shape[name] = { c, t, n: c * t }; buf[name] = new Float64Array(c * t); };
  const ops = [];
  for (const op of json.ops) {
    const o = { ...op };
    switch (op.op) {
      case 'shared_stem_input': {
        const s = shape[op.in], aux = s.c - op.n_slots;
        o.W = tensors[op.w].data; o.B = tensors[op.b].data; o.mask = tensors[op.mask].data; o.k = tensors[op.w].shape[2];
        o.nDomains = tensors[op.mask].shape[0];
        if (op.pad[0] + op.pad[1] !== o.k - 1) throw new Error('slot conv must be length preserving');
        alloc(op.out, op.n_slots * op.slot_width + aux, s.t); o.aux = aux; o.tin = s.t; break;
      }
      case 'conv1d': {
        const s = shape[op.in], [co, ci, k] = tensors[op.w].shape;
        if (ci !== s.c) throw new Error(`${op.in}: expected ${ci} channels, have ${s.c}`);
        o.W = tensors[op.w].data; o.B = tensors[op.b].data; o.co = co; o.ci = ci; o.k = k; o.tin = s.t;
        o.tout = Math.floor((s.t + op.pad[0] + op.pad[1] - op.dilation * (k - 1) - 1) / op.stride) + 1;
        alloc(op.out, co, o.tout); break;
      }
      case 'add_domain': { const s = shape[op.in]; o.table = tensors[op.table].data; o.C = s.c; o.T = s.t; break; }
      case 'add': { const a = shape[op.in[0]], b = shape[op.in[1]]; if (a.n !== b.n) throw new Error('residual shape mismatch'); alloc(op.out, a.c, a.t); o.n = a.n; break; }
      case 'pool': { const s = shape[op.in]; o.C = s.c; o.T = s.t; alloc(op.out, 2 * s.c, 1); break; }
      case 'dense': {
        const [no, ni] = tensors[op.w].shape; o.W = tensors[op.w].data; o.B = tensors[op.b].data; o.no = no; o.ni = ni;
        const tot = op.in.reduce((a, nm) => a + shape[nm].n, 0); if (tot !== ni) throw new Error(`dense input ${tot} != ${ni}`);
        o.cat = new Float64Array(ni); alloc(op.out, no, 1); break;
      }
      case 'sub': { alloc(op.out, shape[op.in[0]].c, 1); o.n = shape[op.in[0]].n; break; }
      default: throw new Error(`unsupported op ${op.op}`);
    }
    ops.push(o);
  }
  const outs = {};
  for (const [name, d] of Object.entries(json.outputs)) outs[name] = { buf: buf[d.buf], idx: d.idx, scale: d.scale ?? 1, shift: d.shift ?? 0, n: d.idx === undefined ? shape[d.buf].n : 1 };
  const out = { hr_logits: new Float64Array(outs.hr_logits.n), emb: new Float64Array(outs.emb.n), spo2: 0, spo2_logvar: 0, lowox_logit: 0, sqi_logit: 0 };
  return { json, T, cin, ops, buf, shape, outs, out, hr: json.hr, params: json.params, cfg: json.config, sideDim: json.input.side.length, embDim: json.input.emb };
}

function conv1d(o, src, dst) {
  const { W, B, co, ci, k, tin, tout } = o, { stride, dilation: d } = o, pl = o.pad[0], relu = o.relu;
  for (let c = 0; c < co; c++) {
    const ob = c * tout, bias = B[c];
    for (let t = 0; t < tout; t++) dst[ob + t] = bias;
    for (let i = 0; i < ci; i++) {
      const ib = i * tin, wb = (c * ci + i) * k;
      for (let kk = 0; kk < k; kk++) {
        const w = W[wb + kk], off = kk * d - pl;        // input index = t*stride + off
        if (w === 0) continue;
        let t0 = off < 0 ? Math.ceil(-off / stride) : 0;                     // first t with index >= 0
        let t1 = Math.min(tout, Math.floor((tin - 1 - off) / stride) + 1);    // one past the last t with index <= tin-1
        for (let t = t0, ix = ib + t0 * stride + off; t < t1; t++, ix += stride) dst[ob + t] += w * src[ix];
      }
    }
    if (relu) for (let t = 0; t < tout; t++) { const v = dst[ob + t]; if (v < 0) dst[ob + t] = 0; }
  }
}

function sharedStem(o, m, domain) {
  const src = m.buf[o.in], dst = m.buf[o.out], T = o.tin, ws = o.slot_width, k = o.k, pl = o.pad[0];
  const mask = domain * o.n_slots;
  for (let s = 0; s < o.n_slots; s++) {
    const on = o.mask[mask + s] !== 0;
    for (let c = 0; c < ws; c++) {
      const ob = (s * ws + c) * T;
      if (!on) { for (let t = 0; t < T; t++) dst[ob + t] = 0; continue; }
      const bias = o.B[c];
      for (let t = 0; t < T; t++) dst[ob + t] = bias;
      for (let kk = 0; kk < k; kk++) {
        const w = o.W[c * k + kk], off = kk - pl, t0 = Math.max(0, -off), t1 = Math.min(T, T - off);
        for (let t = t0; t < t1; t++) dst[ob + t] += w * src[s * T + t + off];
      }
      for (let t = 0; t < T; t++) if (dst[ob + t] < 0) dst[ob + t] = 0;
    }
  }
  const base = o.n_slots * ws * T;
  for (let i = 0, n = o.aux * T; i < n; i++) dst[base + i] = src[o.n_slots * T + i];
}

/**
 * Run the network on one window. x: [channels*T] row-major (channel-major) numeric array.
 * opts: {domain: 0 phone | 1 contact (default 0), side: [5], embBase: [emb]} (defaults: zeros).
 * Returns the model's reusable output object {hr_logits, spo2, spo2_logvar, lowox_logit, sqi_logit, emb}; it is overwritten by the next call.
 */
export function forward(m, x, opts = {}) {
  if (x.length !== m.cin * m.T) throw new Error(`x must have ${m.cin * m.T} values`);
  const domain = opts.domain ?? 0;
  if (!Number.isInteger(domain) || domain < 0 || domain >= (m.cfg.n_domains ?? 1)) throw new Error('bad domain');
  m.buf.x.set(x);
  const side = m.buf.side, eb = m.buf.emb_base;
  if (opts.side) side.set(opts.side); else side.fill(0);
  if (opts.embBase) eb.set(opts.embBase); else eb.fill(0);
  for (const o of m.ops) {
    switch (o.op) {
      case 'shared_stem_input': sharedStem(o, m, domain); break;
      case 'conv1d': conv1d(o, m.buf[o.in], m.buf[o.out]); break;
      case 'add_domain': {
        const a = m.buf[o.in], row = domain * o.C;
        for (let c = 0; c < o.C; c++) { const e = o.table[row + c], b = c * o.T; for (let t = 0; t < o.T; t++) { const v = a[b + t] + e; a[b + t] = o.relu && v < 0 ? 0 : v; } }
        break;
      }
      case 'add': {
        const a = m.buf[o.in[0]], b = m.buf[o.in[1]], r = m.buf[o.out];
        for (let i = 0; i < o.n; i++) { const v = a[i] + b[i]; r[i] = o.relu && v < 0 ? 0 : v; }
        break;
      }
      case 'pool': {
        const a = m.buf[o.in], r = m.buf[o.out];
        for (let c = 0; c < o.C; c++) { let s = 0, mx = -Infinity; for (let t = 0; t < o.T; t++) { const v = a[c * o.T + t]; s += v; if (v > mx) mx = v; } r[c] = s / o.T; r[o.C + c] = mx; }
        break;
      }
      case 'dense': {
        let p = 0; for (const nm of o.in) { const s = m.buf[nm]; for (let i = 0; i < s.length; i++) o.cat[p++] = s[i]; }
        const r = m.buf[o.out];
        for (let j = 0; j < o.no; j++) { let s = o.B[j]; const wb = j * o.ni; for (let i = 0; i < o.ni; i++) s += o.W[wb + i] * o.cat[i]; r[j] = o.relu && s < 0 ? 0 : s; }
        break;
      }
      case 'sub': { const a = m.buf[o.in[0]], b = m.buf[o.in[1]], r = m.buf[o.out]; for (let i = 0; i < o.n; i++) r[i] = a[i] - b[i]; break; }
    }
  }
  const O = m.outs, out = m.out;
  out.hr_logits.set(O.hr_logits.buf); out.emb.set(O.emb.buf);
  for (const k of ['spo2', 'spo2_logvar', 'lowox_logit', 'sqi_logit']) out[k] = O[k].shift + O[k].scale * O[k].buf[O[k].idx];
  return out;
}

/** Mean embedding of a person's calibration windows (the on-device baseline for the low-oxygen head). Allocates; call once per person. */
export function meanEmbedding(embs) {
  const n = embs.length, r = new Float64Array(embs[0].length);
  for (const e of embs) for (let i = 0; i < r.length; i++) r[i] += e[i] / n;
  return r;
}

export const sigmoid = (z) => 1 / (1 + Math.exp(-z));

/** HR in bpm from the 141 logits. mode 'argmax' or 'expect' (probability-weighted mean within +-window bins of the argmax; mirrors model.hr_from_logits). */
export function decodeHr(logits, lo = 40, mode = 'expect', window = 7) {
  let am = 0;
  for (let i = 0; i < logits.length; i++) if (logits[i] > logits[am]) am = i;
  const mx = logits[am];
  if (mode === 'argmax') return lo + am;
  let sw = 0, sb = 0;
  for (let i = Math.max(0, am - window); i <= Math.min(logits.length - 1, am + window); i++) { const p = Math.exp(logits[i] - mx); sw += p; sb += p * (lo + i); }
  return sb / sw;
}
