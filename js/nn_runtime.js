// nn_runtime.js: generic float64 forward pass for small trained networks (MLP, 1D conv / TCN, GRU) from a JSON graph. Pure ES module
// (no DOM, no Node APIs), runs in the browser and in Node tests. It complements cnn.js (OxyNet-specific format 'oxynet-v1'); this file reads format 'nn-runtime-v1'.
// Parity: tests/nn_runtime.mjs against a PyTorch float64 fixture written by python/oxydrop_ml/photon/make_nn_fixture.py (tolerance 1e-6).
// NOT wired into app.js yet and NOT in the service worker ASSETS list (add './js/nn_runtime.js' to app/sw.js when a model that uses it is shipped).
//
// JSON schema ('nn-runtime-v1'):
// {
//   "format": "nn-runtime-v1",
//   "inputs":  { "<name>": { "shape": [C, T] } | { "shape": [N] } },          // every input is given as a flat array, channel-major ([C, T] row-major, index c*T + t)
//   "tensors": { "<name>": { "shape": [...], "data": [numbers] }              // plain JSON numbers (exact float64 round trip)
//                         | { "shape": [...], "b64": "<base64>", "dtype": "f32" | "f64" } },   // little-endian packed
//   "ops":     [ { "op": "<type>", "in": "<name>" | ["<name>", ...], "out": "<name>", ...params }, ... ],   // executed in order; every name is written once (SSA)
//   "outputs": ["<name>", ...]
// }
// A buffer has shape [C, T]; a vector is [N] (= [N, 1]). Ops (all arithmetic in float64, accumulation order fixed by the loops below):
//   linear     { w: tensor [out, in], b?: tensor [out], per_step?: bool }
//                 per_step false (default): input flattened (C*T must equal in) -> vector [out].  per_step true: applied at every time step over the channel axis, [C, T] -> [out, T] (C must equal in).
//   conv1d     { w: tensor [co, ci/groups, k], b?: tensor [co], stride = 1, dilation = 1, groups = 1,
//                padding: "causal" (left pad dilation*(k-1)) | "same" (total pad dilation*(k-1), left floor(total/2), right the rest, as torch padding='same') | "valid" (none) | n | [left, right] }
//                zero padding; out length floor((T + left + right - dilation*(k-1) - 1) / stride) + 1.
//   act        { fn: "relu" | "gelu" | "gelu_tanh" | "tanh" | "sigmoid" }   gelu is the exact erf form (torch default), gelu_tanh the tanh approximation
//   layernorm  { w?: tensor, b?: tensor, eps = 1e-5, dim: "channel" | "time" | "all" }
//                "channel": normalise over the C values at every time step (w, b of length C; this is torch LayerNorm on the transposed [T, C] sequence);
//                "time": over the T values of every channel (w, b of length T, torch LayerNorm(T) on [C, T]); "all": over all C*T values (w, b of length C*T; use it for vectors).
//                Biased variance, as torch.
//   batchnorm  { scale: tensor [C], shift: tensor [C] }  (BatchNorm already folded: scale = gamma / sqrt(var + eps), shift = beta - mean * scale)
//              or { mean, var, w?, b?, eps = 1e-5 } tensors [C] folded at load time. y[c, t] = x[c, t] * scale[c] + shift[c]. Inference mode only.
//   gru        { w_ih: tensor [3H, F], w_hh: tensor [3H, H], b_ih?: [3H], b_hh?: [3H], h0?: "<buffer [H]>", output: "seq" | "last", state_out?: "<name>" }
//                single layer, torch gate order (r, z, n): r = s(Wir x + bir + Whr h + bhr), z = s(...), n = tanh(Win x + bin + r * (Whn h + bhn)), h' = (1 - z) n + z h.
//                in [F, T] -> out [H, T] ("seq") or [H] ("last"); state_out names a second buffer [H] with the final hidden state (stream a recording by feeding it back as h0 of the next call).
//   pool       { mode: "avg" | "max" | "avgmax" }  global pooling over T: [C, T] -> [C] (avgmax: [2C], averages then maxima, as cnn.js)
//   select_last  [C, T] -> [C] (value at the last time step; the causal read-out of a conv / TCN stack)
//   flatten    any -> [C*T]
//   concat     { axis: "flat" (default) | "channel" }   flat: flatten and concatenate; channel: [C1, T] + [C2, T] -> [C1 + C2, T]
//   add        in: [a, b] same shape (residual connection, no activation)
//   slice      { start, end }  flat slice [start, end) of the flattened input -> [end - start]
// forward(inputs) copies the inputs into preallocated buffers, runs the ops (no allocation) and returns { name: Float64Array } for the listed outputs.
// The returned arrays are the model's own buffers: copy them before the next forward() if you keep them.

export const FORMAT = 'nn-runtime-v1';

// ---------------------------------------------------------------------------------------------------- numerics
/** erf to about 1e-15: Taylor series for |x| < 2.5, continued fraction for erfc above. */
export function erf(x) {
  const ax = Math.abs(x);
  if (ax < 2.5) {
    let term = x, sum = x;
    for (let n = 1; n < 200; n++) {
      term *= -x * x / n;
      const d = term / (2 * n + 1);
      sum += d;
      if (Math.abs(d) < 1e-17 * Math.abs(sum)) break;
    }
    return 1.1283791670955126 * sum;                                   // 2 / sqrt(pi)
  }
  if (ax > 6) return x > 0 ? 1 : -1;
  let f = ax;
  for (let k = 80; k >= 1; k--) f = ax + (k / 2) / f;
  const erfc = Math.exp(-ax * ax) / (1.7724538509055159 * f);          // sqrt(pi)
  return x > 0 ? 1 - erfc : erfc - 1;
}
const sigmoid = (x) => (x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x)));
const ACT = {
  relu: (x) => (x > 0 ? x : 0),
  gelu: (x) => 0.5 * x * (1 + erf(x * 0.7071067811865476)),
  gelu_tanh: (x) => 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))),
  tanh: Math.tanh,
  sigmoid,
};

function decodeTensor(t, name) {
  const n = t.shape.reduce((a, b) => a * b, 1);
  let out;
  if (t.data) {
    out = Float64Array.from(t.data);
  } else if (t.b64) {
    const s = atob(t.b64), u = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
    const dv = new DataView(u.buffer);                                  // explicit little-endian, independent of host endianness
    if (t.dtype === 'f64') { out = new Float64Array(u.length / 8); for (let i = 0; i < out.length; i++) out[i] = dv.getFloat64(8 * i, true); }
    else if (t.dtype === 'f32') { out = new Float64Array(u.length / 4); for (let i = 0; i < out.length; i++) out[i] = dv.getFloat32(4 * i, true); }
    else throw new Error(`tensor ${name}: b64 needs dtype 'f32' or 'f64'`);
  } else throw new Error(`tensor ${name}: no data`);
  if (out.length !== n) throw new Error(`tensor ${name}: ${out.length} values for shape [${t.shape}]`);
  return { shape: t.shape.slice(), data: out };
}

// ---------------------------------------------------------------------------------------------------- loader
/** Build a network from the parsed JSON. Allocates every buffer; forward() allocates nothing. */
export function loadNN(json) {
  if (json.format !== FORMAT) throw new Error(`unsupported model format ${json.format}`);
  const tensors = {};
  for (const [k, v] of Object.entries(json.tensors ?? {})) tensors[k] = decodeTensor(v, k);
  const T = (name) => { if (!tensors[name]) throw new Error(`missing tensor ${name}`); return tensors[name]; };
  const shape = {}, buf = {};
  const alloc = (name, c, t) => {
    if (shape[name]) throw new Error(`buffer ${name} is written twice`);
    shape[name] = { c, t, n: c * t }; buf[name] = new Float64Array(c * t); return buf[name];
  };
  const need = (name) => { if (!shape[name]) throw new Error(`buffer ${name} is used before it is defined`); return shape[name]; };
  const inputs = {};
  for (const [name, v] of Object.entries(json.inputs)) {
    const [c, t = 1] = v.shape; alloc(name, c, t); inputs[name] = { shape: v.shape.slice() };
  }
  const run = [];
  for (const op of json.ops) {
    const one = typeof op.in === 'string' ? op.in : op.in?.[0];
    switch (op.op) {
      case 'linear': {
        const s = need(one), w = T(op.w), [no, ni] = w.shape, W = w.data, B = op.b ? T(op.b).data : null;
        if (B && B.length !== no) throw new Error(`${op.out}: bias length`);
        const src = buf[one];
        if (op.per_step) {
          if (s.c !== ni) throw new Error(`${op.out}: linear expects ${ni} channels, has ${s.c}`);
          const dst = alloc(op.out, no, s.t), Tn = s.t;
          run.push(() => {
            for (let o = 0; o < no; o++) for (let t = 0; t < Tn; t++) {
              let a = B ? B[o] : 0;
              for (let i = 0; i < ni; i++) a += W[o * ni + i] * src[i * Tn + t];
              dst[o * Tn + t] = a;
            }
          });
        } else {
          if (s.n !== ni) throw new Error(`${op.out}: linear expects ${ni} inputs, has ${s.n}`);
          const dst = alloc(op.out, no, 1);
          run.push(() => {
            for (let o = 0; o < no; o++) {
              let a = B ? B[o] : 0;
              for (let i = 0; i < ni; i++) a += W[o * ni + i] * src[i];
              dst[o] = a;
            }
          });
        }
        break;
      }
      case 'conv1d': {
        const s = need(one), w = T(op.w), [co, cig, k] = w.shape, W = w.data, B = op.b ? T(op.b).data : null;
        const stride = op.stride ?? 1, dil = op.dilation ?? 1, groups = op.groups ?? 1;
        if (cig * groups !== s.c || co % groups !== 0) throw new Error(`${op.out}: conv expects ${cig * groups} channels, has ${s.c}`);
        const span = dil * (k - 1);
        let pl, pr;
        if (op.padding === 'causal') { pl = span; pr = 0; }
        else if (op.padding === 'same') { pl = Math.floor(span / 2); pr = span - pl; }
        else if (op.padding === 'valid' || op.padding === undefined) { pl = 0; pr = 0; }
        else if (typeof op.padding === 'number') { pl = pr = op.padding; }
        else [pl, pr] = op.padding;
        const tin = s.t, tout = Math.floor((tin + pl + pr - span - 1) / stride) + 1;
        if (tout < 1) throw new Error(`${op.out}: input too short for the convolution`);
        const src = buf[one], dst = alloc(op.out, co, tout), cog = co / groups;
        run.push(() => {
          for (let o = 0; o < co; o++) {
            const g = Math.floor(o / cog), c0 = g * cig;
            for (let t = 0; t < tout; t++) {
              let a = B ? B[o] : 0;
              const base = t * stride - pl;
              for (let ci = 0; ci < cig; ci++) {
                const wr = (o * cig + ci) * k, sr = (c0 + ci) * tin;
                for (let j = 0; j < k; j++) {
                  const p = base + j * dil;
                  if (p >= 0 && p < tin) a += W[wr + j] * src[sr + p];
                }
              }
              dst[o * tout + t] = a;
            }
          }
        });
        break;
      }
      case 'act': {
        const s = need(one), f = ACT[op.fn];
        if (!f) throw new Error(`unsupported activation ${op.fn}`);
        const src = buf[one], dst = alloc(op.out, s.c, s.t), n = s.n;
        run.push(() => { for (let i = 0; i < n; i++) dst[i] = f(src[i]); });
        break;
      }
      case 'layernorm': {
        const s = need(one), eps = op.eps ?? 1e-5, dim = op.dim ?? 'channel';
        const G = op.w ? T(op.w).data : null, Bt = op.b ? T(op.b).data : null;
        const src = buf[one], dst = alloc(op.out, s.c, s.t), C = s.c, Tn = s.t;
        const len = dim === 'channel' ? C : dim === 'time' ? Tn : s.n;
        if ((G && G.length !== len) || (Bt && Bt.length !== len)) throw new Error(`${op.out}: layernorm affine length must be ${len}`);
        // groups of `len` values: index(g, j) = channel: j * Tn + g (g = time step); time: g * Tn + j (g = channel); all: j
        const ng = dim === 'channel' ? Tn : dim === 'time' ? C : 1;
        const idx = dim === 'channel' ? (g, j) => j * Tn + g : dim === 'time' ? (g, j) => g * Tn + j : (g, j) => j;
        run.push(() => {
          for (let g = 0; g < ng; g++) {
            let m = 0;
            for (let j = 0; j < len; j++) m += src[idx(g, j)];
            m /= len;
            let v = 0;
            for (let j = 0; j < len; j++) { const d = src[idx(g, j)] - m; v += d * d; }
            const inv = 1 / Math.sqrt(v / len + eps);
            for (let j = 0; j < len; j++) {
              const p = idx(g, j);
              dst[p] = (src[p] - m) * inv * (G ? G[j] : 1) + (Bt ? Bt[j] : 0);
            }
          }
        });
        break;
      }
      case 'batchnorm': {
        const s = need(one);
        let sc, sh;
        if (op.scale) { sc = T(op.scale).data; sh = T(op.shift).data; }
        else {
          const mean = T(op.mean).data, v = T(op.var).data, g = op.w ? T(op.w).data : null, b = op.b ? T(op.b).data : null, eps = op.eps ?? 1e-5;
          sc = new Float64Array(s.c); sh = new Float64Array(s.c);
          for (let c = 0; c < s.c; c++) { sc[c] = (g ? g[c] : 1) / Math.sqrt(v[c] + eps); sh[c] = (b ? b[c] : 0) - mean[c] * sc[c]; }
        }
        if (sc.length !== s.c || sh.length !== s.c) throw new Error(`${op.out}: batchnorm expects ${s.c} channels`);
        const src = buf[one], dst = alloc(op.out, s.c, s.t), Tn = s.t, C = s.c;
        run.push(() => { for (let c = 0; c < C; c++) for (let t = 0; t < Tn; t++) dst[c * Tn + t] = src[c * Tn + t] * sc[c] + sh[c]; });
        break;
      }
      case 'gru': {
        const s = need(one), wih = T(op.w_ih), whh = T(op.w_hh), H = whh.shape[1], F = wih.shape[1];
        if (wih.shape[0] !== 3 * H || whh.shape[0] !== 3 * H) throw new Error(`${op.out}: gru weight shapes`);
        if (s.c !== F) throw new Error(`${op.out}: gru expects ${F} features, has ${s.c}`);
        const Wi = wih.data, Wh = whh.data, bi = op.b_ih ? T(op.b_ih).data : null, bh = op.b_hh ? T(op.b_hh).data : null;
        const Tn = s.t, src = buf[one], seq = op.output !== 'last';
        let h0 = null;
        if (op.h0) { if (need(op.h0).n !== H) throw new Error(`${op.out}: h0 must have ${H} values`); h0 = buf[op.h0]; }
        const dst = seq ? alloc(op.out, H, Tn) : alloc(op.out, H, 1);
        const hout = op.state_out ? alloc(op.state_out, H, 1) : null;
        const h = new Float64Array(H), gi = new Float64Array(3 * H), gh = new Float64Array(3 * H);
        run.push(() => {
          if (h0) h.set(h0); else h.fill(0);
          for (let t = 0; t < Tn; t++) {
            for (let r = 0; r < 3 * H; r++) {
              let a = bi ? bi[r] : 0, b = bh ? bh[r] : 0;
              for (let i = 0; i < F; i++) a += Wi[r * F + i] * src[i * Tn + t];
              for (let i = 0; i < H; i++) b += Wh[r * H + i] * h[i];
              gi[r] = a; gh[r] = b;
            }
            for (let j = 0; j < H; j++) {
              const r = sigmoid(gi[j] + gh[j]), z = sigmoid(gi[H + j] + gh[H + j]);
              const n = Math.tanh(gi[2 * H + j] + r * gh[2 * H + j]);
              h[j] = (1 - z) * n + z * h[j];
              if (seq) dst[j * Tn + t] = h[j];
            }
          }
          if (!seq) dst.set(h);
          if (hout) hout.set(h);
        });
        break;
      }
      case 'pool': {
        const s = need(one), src = buf[one], C = s.c, Tn = s.t, mode = op.mode ?? 'avg';
        if (!['avg', 'max', 'avgmax'].includes(mode)) throw new Error(`unsupported pool mode ${mode}`);
        const dst = alloc(op.out, mode === 'avgmax' ? 2 * C : C, 1);
        run.push(() => {
          for (let c = 0; c < C; c++) {
            let a = 0, m = -Infinity;
            for (let t = 0; t < Tn; t++) { const v = src[c * Tn + t]; a += v; if (v > m) m = v; }
            a /= Tn;
            if (mode === 'avg') dst[c] = a; else if (mode === 'max') dst[c] = m; else { dst[c] = a; dst[C + c] = m; }
          }
        });
        break;
      }
      case 'select_last': {
        const s = need(one), src = buf[one], dst = alloc(op.out, s.c, 1), C = s.c, Tn = s.t;
        run.push(() => { for (let c = 0; c < C; c++) dst[c] = src[c * Tn + Tn - 1]; });
        break;
      }
      case 'flatten': {
        const s = need(one), src = buf[one], dst = alloc(op.out, s.n, 1);
        run.push(() => dst.set(src));
        break;
      }
      case 'concat': {
        const names = op.in, ss = names.map(need), axis = op.axis ?? 'flat';
        if (axis === 'channel') {
          if (ss.some((s) => s.t !== ss[0].t)) throw new Error(`${op.out}: concat over channels needs equal T`);
          const dst = alloc(op.out, ss.reduce((a, s) => a + s.c, 0), ss[0].t);
          run.push(() => { let o = 0; for (const nm of names) { dst.set(buf[nm], o); o += buf[nm].length; } });
        } else {
          const dst = alloc(op.out, ss.reduce((a, s) => a + s.n, 0), 1);
          run.push(() => { let o = 0; for (const nm of names) { dst.set(buf[nm], o); o += buf[nm].length; } });
        }
        break;
      }
      case 'add': {
        const [a, b] = op.in, sa = need(a), sb = need(b);
        if (sa.c !== sb.c || sa.t !== sb.t) throw new Error(`${op.out}: add shape mismatch`);
        const A = buf[a], Bf = buf[b], dst = alloc(op.out, sa.c, sa.t), n = sa.n;
        run.push(() => { for (let i = 0; i < n; i++) dst[i] = A[i] + Bf[i]; });
        break;
      }
      case 'slice': {
        const s = need(one), src = buf[one];
        if (op.start < 0 || op.end > s.n || op.end <= op.start) throw new Error(`${op.out}: bad slice`);
        const dst = alloc(op.out, op.end - op.start, 1);
        run.push(() => { for (let i = op.start; i < op.end; i++) dst[i - op.start] = src[i]; });
        break;
      }
      default: throw new Error(`unsupported op ${op.op}`);
    }
  }
  for (const name of json.outputs) need(name);
  return {
    format: FORMAT,
    inputs,
    outputs: json.outputs.slice(),
    shapeOf: (name) => ({ ...need(name) }),
    /** inputs: { name: ArrayLike<number> } (flat, channel-major). Returns { outputName: Float64Array } (the model's own buffers). */
    forward(x) {
      for (const name of Object.keys(inputs)) {
        const v = x[name];
        if (!v || v.length !== shape[name].n) throw new Error(`input ${name}: expected ${shape[name].n} values`);
        buf[name].set(v);
      }
      for (const f of run) f();
      const out = {};
      for (const name of json.outputs) out[name] = buf[name];
      return out;
    },
  };
}
