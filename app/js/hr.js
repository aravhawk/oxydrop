// Robust heart-rate estimator (v2): JS port of python/oxydrop/hr.py. Spectral harmonic-sum + autocorrelation fundamental, strict validity gate.
export const FS = 30, NFFT = 4096, F_LO = 0.65, F_HI = 3.0;
export const GATE = { SQI_MIN: 0.30, ACF_MIN: 0.30, AGREE: 0.10, MIN_SEC: 9.5 };

// in-place radix-2 complex FFT (inverse=true scales by 1/n)
export function fft(re, im, inverse = false) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) { let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; } }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = 2 * Math.PI / len * (inverse ? 1 : -1), wr = Math.cos(ang), wi = Math.sin(ang), half = len >> 1;
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < half; k++) {
        const a = i + k, b = a + half, tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        const ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
      }
    }
  }
  if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}
const mean = (a) => { let s = 0; for (const v of a) s += v; return s / a.length; };
const stdp = (a) => { const m = mean(a); let s = 0; for (const v of a) s += (v - m) ** 2; return Math.sqrt(s / a.length); };

export function estimateHr(x, fs = FS, gate = GATE) {
  const out = { hr: NaN, hrAcf: NaN, sqi: 0, acf: 0, valid: false };
  const n = x.length;
  if (n < gate.MIN_SEC * fs || stdp(x) < 1e-9) return out;
  const m0 = mean(x), xc = x.map((v) => v - m0);
  const re = new Float64Array(NFFT), im = new Float64Array(NFFT);
  for (let i = 0; i < n; i++) re[i] = xc[i] * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1)));
  fft(re, im);
  const half = NFFT / 2 + 1, P = new Float64Array(half);
  for (let i = 0; i < half; i++) P[i] = Math.hypot(re[i], im[i]);
  const df = fs / NFFT, k0 = Math.ceil(F_LO / df), k1 = Math.floor(F_HI / df);
  let kb = k0, best = -Infinity;
  for (let k = k0; k <= k1; k++) { const sc = P[k] + 0.7 * P[2 * k]; if (sc > best) { best = sc; kb = k; } }
  let kf = kb;
  { const a = P[kb - 1], b = P[kb], c = P[kb + 1], d = a - 2 * b + c; kf = kb + (d !== 0 ? 0.5 * (a - c) / d : 0); }
  const f0 = kf * df;
  let tot = 1e-18; for (let k = Math.ceil(0.65 / df); k <= Math.floor(4.0 / df); k++) tot += P[k] * P[k];
  const w = Math.round(0.10 / df); let near = 0;
  for (let k = Math.max(kb - w, 0); k <= kb + w; k++) near += P[k] * P[k];
  for (let k = Math.max(2 * kb - w, 0); k <= 2 * kb + w; k++) near += P[k] * P[k];
  const sqi = near / tot;
  // autocorrelation via FFT
  let M = 1; while (M < 2 * n) M <<= 1;
  const ar = new Float64Array(M), ai = new Float64Array(M);
  for (let i = 0; i < n; i++) ar[i] = xc[i];
  fft(ar, ai);
  for (let i = 0; i < M; i++) { ar[i] = ar[i] * ar[i] + ai[i] * ai[i]; ai[i] = 0; }
  fft(ar, ai, true);
  const r = new Float64Array(n); const r0 = ar[0] + 1e-18; for (let i = 0; i < n; i++) r[i] = ar[i] / r0;
  const lo = Math.ceil(fs / 3.0), hi = Math.min(Math.floor(fs / 0.65), n - 2);
  const seg = Array.from(r.slice(lo - 1, hi + 2)), pk = [];
  for (let i = 1; i < seg.length - 1; i++) if (seg[i] > seg[i - 1] && seg[i] >= seg[i + 1]) pk.push(i);
  out.hr = 60 * f0; out.sqi = sqi;
  if (!pk.length) return out;
  let rmax = -Infinity; for (const i of pk) rmax = Math.max(rmax, seg[i]);
  const li = pk.find((i) => seg[i] >= 0.85 * rmax);
  const a = seg[li - 1], b = seg[li], c = seg[li + 1], d = a - 2 * b + c;
  const lag = (lo - 1) + li + (d !== 0 ? 0.5 * (a - c) / d : 0), fAcf = fs / lag;
  out.hrAcf = 60 * fAcf; out.acf = seg[li];
  const agree = Math.abs(fAcf - f0) / f0 < gate.AGREE;
  out.valid = agree && sqi >= gate.SQI_MIN && out.acf >= gate.ACF_MIN;
  return out;
}

/** acs: array of 3 arrays (R,G,B band-passed window). Try primary channel first, then others by decreasing amplitude. */
export function bestChannelEstimate(acs, primary, fs = FS, gate = GATE) {
  const sd = acs.map(stdp);
  const order = [primary, ...[0, 1, 2].filter((c) => c !== primary).sort((p, q) => sd[q] - sd[p])];
  let first = null;
  for (const c of order) {
    const e = estimateHr(acs[c], fs, gate);
    if (!first) first = { est: e, ch: c };
    if (e.valid) return { est: e, ch: c };
  }
  return first;
}
