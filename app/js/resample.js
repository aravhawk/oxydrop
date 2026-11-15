// Frame timestamps -> uniform 30 Hz samples. Handles 24-120 fps sources, jittery callback times, and dropped frames.
//  * de-jitter: timestamps are pulled toward a steady cadence (period = running median of recent intervals) unless a gap is detected;
//  * gaps (> 1.6 periods) snap to the measured time and are filled by linear interpolation. Each gap is counted AND its duration is
//    recorded in output-sample indices, so the decision logic can tell a lost frame from a multi-second stall (see gapStats);
//  * sources faster than ~40 fps are box-averaged into each 1/30 s bin (anti-alias, less noise); slower ones are linearly interpolated.
export class Resampler {
  constructor(onSample, hz = 30) {
    this.onSample = onSample; this.dt = 1 / hz; this.frames = []; this.rawDts = []; this.prevRaw = null; this.prevSm = null;
    this.nextG = null; this.n = 0; this.gaps = 0; this.maxGap = 0; this.period = 1 / 30; this.firstT = null; this.lastT = null;
    this.gapSpans = [];   // {start, end, dur}: start/end are output-sample indices the gap covers, dur in seconds
  }
  /** Missing-time statistics for gaps that overlap output samples with index >= sinceN: {max: longest gap (s), total: summed gap time (s), count}. */
  gapStats(sinceN = 0) {
    let max = 0, total = 0, count = 0;
    for (const g of this.gapSpans) if (g.end > sinceN) { const part = Math.min(g.dur, (g.end - Math.max(g.start, sinceN)) * this.dt); max = Math.max(max, g.dur); total += part; count++; }
    return { max, total, count };
  }
  get fps() { return this.rawDts.length > 5 ? 1 / this.period : 0; }
  get jitterMs() { if (this.rawDts.length < 10) return 0; const m = this.rawDts.reduce((a, b) => a + b, 0) / this.rawDts.length; return 1000 * Math.sqrt(this.rawDts.reduce((a, b) => a + (b - m) ** 2, 0) / this.rawDts.length); }
  push(t, r, g, b) {
    if (this.prevRaw === null) { this.prevRaw = t; this.prevSm = t; this.firstT = t; this.nextG = t; this.frames.push([t, r, g, b]); this.lastT = t; return; }
    const dtRaw = t - this.prevRaw; this.prevRaw = t;
    if (dtRaw <= 0) return;                                   // duplicate / out-of-order timestamp
    this.rawDts.push(dtRaw); if (this.rawDts.length > 91) this.rawDts.shift();
    const s = [...this.rawDts].sort((a, b) => a - b); this.period = s[s.length >> 1];
    let ts;
    if (dtRaw > 1.6 * this.period && this.rawDts.length > 10) { ts = t; this.gaps++; this.maxGap = Math.max(this.maxGap, dtRaw); this._span(dtRaw); }   // dropped frames: do not trust cadence
    else { const expected = this.prevSm + this.period; ts = expected + 0.25 * (t - expected); }
    if (ts <= this.prevSm) ts = this.prevSm + 1e-4;
    this.prevSm = ts; this.frames.push([ts, r, g, b]); this.lastT = ts; this._drain();
  }
  _span(dur) {
    this.gapSpans.push({ start: this.n, end: this.n + Math.max(1, Math.round(dur / this.dt)), dur });
    while (this.gapSpans.length && this.gapSpans[0].end < this.n - 120 * 30) this.gapSpans.shift();   // keep ~2 minutes
  }
  _drain() {
    const F = this.frames, half = this.dt / 2, avg = this.fps > 40;
    while (F.length >= 2 && this.nextG + (avg ? half : 0) <= F[F.length - 1][0]) {
      const g = this.nextG; let out = null;
      if (avg) {
        let c = 0; const acc = [0, 0, 0];
        for (const f of F) if (f[0] >= g - half && f[0] < g + half) { acc[0] += f[1]; acc[1] += f[2]; acc[2] += f[3]; c++; }
        if (c) out = acc.map((v) => v / c);
      }
      if (!out) {   // linear interpolation between bracketing frames
        let i = F.length - 1; while (i > 0 && F[i - 1][0] > g) i--;
        const a = F[Math.max(i - 1, 0)], b = F[i], w = b[0] > a[0] ? Math.min(Math.max((g - a[0]) / (b[0] - a[0]), 0), 1) : 0;
        out = [a[1] + w * (b[1] - a[1]), a[2] + w * (b[2] - a[2]), a[3] + w * (b[3] - a[3])];
      }
      this.onSample(out[0], out[1], out[2]); this.n++; this.nextG += this.dt;
    }
    while (F.length > 12 && F[1][0] < this.nextG - this.dt) F.shift();
  }
}
