const SAMPLE_RATE = 30;
const LOW_HZ = 0.7;
const HIGH_HZ = 3.5;
const AGREEMENT_BPM = 5;
const MIN_SQI = 0.6;
const FREQUENCY_STEP = 1 / 60;

function resample(frames) {
  if (!Array.isArray(frames) || frames.length < 2) return null;
  const start = frames[0]?.timestampMs;
  const end = frames.at(-1)?.timestampMs;
  const duration = (end - start) / 1000;
  if (!Number.isFinite(duration) || duration < 7.99 || (frames.length - 1) / duration < 12) {
    return null;
  }
  let previous = start;
  for (let i = 0; i < frames.length; i++) {
    const frame = frames[i];
    if (!frame || ![frame.timestampMs, frame.red, frame.green].every(Number.isFinite)) return null;
    const gap = frame.timestampMs - previous;
    if (gap > 250 || (i > 0 && gap <= 0)) return null;
    previous = frame.timestampMs;
  }

  const count = 1 + Math.floor(duration * SAMPLE_RATE + 1e-8);
  const red = new Float64Array(count);
  const green = new Float64Array(count);
  let next = 1;
  for (let i = 0; i < count; i++) {
    const time = start + i * 1000 / SAMPLE_RATE;
    while (next < frames.length - 1 && frames[next].timestampMs < time) next++;
    const before = frames[next - 1];
    const after = frames[next];
    const fraction = (time - before.timestampMs) / (after.timestampMs - before.timestampMs);
    red[i] = before.red + fraction * (after.red - before.red);
    green[i] = before.green + fraction * (after.green - before.green);
  }
  return [red, green];
}

function detrend(signal) {
  const midpoint = (signal.length - 1) / 2;
  let sum = 0;
  let covariance = 0;
  let variance = 0;
  for (let i = 0; i < signal.length; i++) {
    const time = i - midpoint;
    sum += signal[i];
    covariance += time * signal[i];
    variance += time * time;
  }
  const mean = sum / signal.length;
  const slope = covariance / variance;
  return signal.map((value, i) => value - mean - slope * (i - midpoint));
}

function butterworth(signal, cutoff, highPass) {
  const radians = 2 * Math.PI * cutoff / SAMPLE_RATE;
  const cosine = Math.cos(radians);
  const alpha = Math.sin(radians) / Math.SQRT2;
  const a0 = 1 + alpha;
  const a1 = -2 * cosine / a0;
  const a2 = (1 - alpha) / a0;
  const b0 = (highPass ? 1 + cosine : 1 - cosine) / (2 * a0);
  const b1 = (highPass ? -2 : 2) * b0;

  let x1 = signal[0];
  let x2 = x1;
  let y1 = highPass ? 0 : x1;
  let y2 = y1;
  return signal.map((x) => {
    const y = b0 * (x + x2) + b1 * x1 - a1 * y1 - a2 * y2;
    x2 = x1;
    x1 = x;
    y2 = y1;
    y1 = y;
    return y;
  });
}

function bandpass(signal) {
  const pad = 2 * SAMPLE_RATE;
  const last = signal.length - 1;
  const padded = new Float64Array(signal.length + 2 * pad);
  padded.set(signal, pad);
  for (let i = 1; i <= pad; i++) {
    padded[pad - i] = 2 * signal[0] - signal[i];
    padded[pad + last + i] = 2 * signal[last] - signal[last - i];
  }
  // Forward and reverse filtering avoids a phase shift. Padding limits edge transients.
  const forward = butterworth(butterworth(padded, LOW_HZ, true), HIGH_HZ, false);
  const reverse = butterworth(butterworth(forward.reverse(), LOW_HZ, true), HIGH_HZ, false);
  return reverse.reverse().subarray(pad, pad + signal.length);
}

function isPeak(values, i) {
  return values[i] > values[i - 1] && values[i] >= values[i + 1];
}

function peakOffset(values, i) {
  const left = values[i - 1];
  const center = values[i];
  const right = values[i + 1];
  const curvature = left - 2 * center + right;
  return curvature < 0 ? (left - right) / (2 * curvature) : 0;
}

function autocorrelation(signal) {
  const maxLag = Math.ceil(SAMPLE_RATE / LOW_HZ) + 2;
  const values = new Float64Array(maxLag + 1);
  for (let lag = 1; lag <= maxLag; lag++) {
    let sumA = 0;
    let sumB = 0;
    let sumAA = 0;
    let sumBB = 0;
    let sumAB = 0;
    const count = signal.length - lag;
    for (let i = 0; i < count; i++) {
      const a = signal[i];
      const b = signal[i + lag];
      sumA += a;
      sumB += b;
      sumAA += a * a;
      sumBB += b * b;
      sumAB += a * b;
    }
    const varianceA = sumAA - sumA * sumA / count;
    const varianceB = sumBB - sumB * sumB / count;
    const scale = Math.sqrt(varianceA * varianceB);
    values[lag] = scale > 0 ? (sumAB - sumA * sumB / count) / scale : 0;
  }

  const peaks = [];
  for (let lag = Math.floor(SAMPLE_RATE / HIGH_HZ); lag < maxLag; lag++) {
    if (!isPeak(values, lag)) continue;
    const hz = SAMPLE_RATE / (lag + peakOffset(values, lag));
    if (hz < LOW_HZ - FREQUENCY_STEP || hz > HIGH_HZ + FREQUENCY_STEP) continue;
    peaks.push({ hz, strength: values[lag] });
  }
  const strongest = Math.max(0, ...peaks.map((peak) => peak.strength));
  // Use the first comparably strong repeat instead of a later multiple of the period.
  return { peak: peaks.find(({ strength }) => strength >= Math.max(0.3, 0.9 * strongest)), peaks };
}

function spectrum(signal) {
  const windowed = signal.map((value, i) =>
    value * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / (signal.length - 1))));
  const powers = new Float64Array(301);
  let total = 0;
  for (let bin = 1; bin < powers.length; bin++) {
    const coefficient = 2 * Math.cos(2 * Math.PI * bin * FREQUENCY_STEP / SAMPLE_RATE);
    let previous = 0;
    let older = 0;
    // Goertzel evaluates one Fourier frequency per bin.
    for (let i = 0; i < windowed.length; i++) {
      const next = windowed[i] + coefficient * previous - older;
      older = previous;
      previous = next;
    }
    const power = previous * previous + older * older - coefficient * previous * older;
    powers[bin] = Math.max(0, power);
    total += powers[bin];
  }

  let bin = Math.round(LOW_HZ / FREQUENCY_STEP);
  for (let i = bin + 1; i <= Math.round(HIGH_HZ / FREQUENCY_STEP); i++) {
    if (powers[i] > powers[bin]) bin = i;
  }
  const hz = (bin + peakOffset(powers, bin)) * FREQUENCY_STEP;
  const radius = 1.3 * SAMPLE_RATE / signal.length;
  let peakPower = 0;
  for (let i = 1; i < powers.length; i++) {
    if (Math.abs(i * FREQUENCY_STEP - hz) <= radius) peakPower += powers[i];
  }
  return { hz, power: powers[bin], concentration: total > 0 ? peakPower / total : 0, powers };
}

function hasHarmonicAlternative(period, frequency, ratio, minimumPower) {
  const alternativeHz = frequency.hz * ratio;
  if (alternativeHz < LOW_HZ - FREQUENCY_STEP || alternativeHz > HIGH_HZ + FREQUENCY_STEP) return false;
  const margin = AGREEMENT_BPM / 60;
  const hasRepeat = period.peaks.some(({ hz, strength }) =>
    Math.abs(hz - alternativeHz) <= margin && strength >= period.peak.strength - 0.08);
  if (!hasRepeat) return false;

  const start = Math.ceil((alternativeHz - margin) / FREQUENCY_STEP);
  const end = Math.floor((alternativeHz + margin) / FREQUENCY_STEP);
  for (let bin = start; bin <= end; bin++) {
    if (isPeak(frequency.powers, bin) && frequency.powers[bin] >= minimumPower * frequency.power) return true;
  }
  return false;
}

function analyze(channel) {
  const signal = bandpass(detrend(channel));
  let energy = 0;
  let scale = 0;
  for (let i = 0; i < signal.length; i++) {
    energy += signal[i] * signal[i];
    scale = Math.max(scale, Math.abs(channel[i]));
  }
  // A constant or linear ramp can leave roundoff after interpolation and filtering.
  if (Math.sqrt(energy / signal.length) <= 32 * Number.EPSILON * scale) {
    return { bpm: null, sqi: 0 };
  }
  const period = autocorrelation(signal);
  if (!period.peak) return { bpm: null, sqi: 0 };
  const frequency = spectrum(signal);
  const sqi = Math.min(1, 0.6 * period.peak.strength + 0.4 * frequency.concentration);
  // A strong fundamental can have a weak second harmonic. A weaker peak at
  // half the selected rate is more suspicious because it implies a longer beat.
  if (Math.abs(period.peak.hz - frequency.hz) * 60 > AGREEMENT_BPM ||
      hasHarmonicAlternative(period, frequency, 0.5, 0.12) ||
      hasHarmonicAlternative(period, frequency, 2, 0.6)) {
    return { bpm: null, sqi: Math.min(sqi, MIN_SQI - 0.01) };
  }
  const hz = (period.peak.hz + frequency.hz) / 2;
  const bpm = Math.round(60 * Math.max(LOW_HZ, Math.min(HIGH_HZ, hz)));
  return { bpm, sqi };
}

/**
 * Estimate HR from ordered frame means {timestampMs, red, green}. Timestamp units
 * are milliseconds on one monotonic clock. Red and green may use any linear units.
 * A rejected reading returns {bpm: null, sqi, method: 're-measure'}.
 */
export function estimateHR(frames) {
  const channels = resample(frames);
  if (!channels) return { bpm: null, sqi: 0, method: 're-measure' };

  const [red, green] = channels.map(analyze);
  const valid = [red, green].filter(({ bpm, sqi }) => bpm !== null && sqi >= MIN_SQI);
  if (valid.length === 0) {
    return { bpm: null, sqi: Math.max(red.sqi, green.sqi), method: 're-measure' };
  }
  if (valid.length === 2 && Math.abs(red.bpm - green.bpm) > AGREEMENT_BPM) {
    return { bpm: null, sqi: Math.min(red.sqi, green.sqi, MIN_SQI - 0.01), method: 're-measure' };
  }
  const estimate = valid.reduce((best, current) => current.sqi > best.sqi ? current : best);
  return { ...estimate, method: 'autocorrelation+spectral' };
}
