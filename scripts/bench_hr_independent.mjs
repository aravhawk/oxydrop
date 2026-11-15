// Download and unzip BUT PPG v2.0.0 from https://physionet.org/content/butppg/2.0.0/
// Run: node scripts/bench_hr_independent.mjs /path/to/extracted/dataset
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { estimateHR } from '../src/hr_independent.js';

function readAnnotations(csv) {
  const [header, ...rows] = csv.replace(/^\uFEFF/, '').trim().split(/\r?\n/);
  if (header !== 'ID,Quality,HR') throw new Error('Expected the BUT PPG v2.0.0 annotations');
  const seen = new Set();
  return rows.map((row) => {
    const [id, quality, hr] = row.split(',');
    const reference = Number(hr);
    if (!/^\d{6}$/.test(id) || seen.has(id) || !/^[01]$/.test(quality) ||
        !Number.isFinite(reference) || reference <= 0) {
      throw new Error(`Invalid annotation: ${row}`);
    }
    seen.add(id);
    return { id, quality: Number(quality), reference };
  });
}

async function loadFrames(root, id) {
  const stem = join(root, id, `${id}_PPG`);
  const [header, data] = await Promise.all([readFile(`${stem}.hea`, 'utf8'), readFile(`${stem}.dat`)]);
  const [record, ...lines] = header.split(/\r?\n/).map((line) => line.trim())
    .filter((line) => line && !line.startsWith('#'));
  const [, channelCount, sampleRate, sampleCount] = record.trim().split(/\s+/).map(Number);
  const legacy = channelCount === 300 && sampleCount === 1;
  const rgb = channelCount === 3 && sampleCount === 300;
  if ((!legacy && !rgb) || sampleRate !== 30 || lines.length !== channelCount ||
      data.length !== 2 * channelCount * sampleCount) {
    throw new Error(`${id}: unexpected BUT PPG v2.0.0 record layout`);
  }

  const channels = lines.map((line) => {
    const [file, format, calibration, , adcZero, , , , name] = line.split(/\s+/);
    const [gainText, baselineText] = calibration.split(/[(/]/);
    const gain = Number(gainText);
    const baseline = Number(baselineText?.replace(')', '') ?? adcZero);
    if (file !== `${id}_PPG.dat` || format !== '16' || !Number.isFinite(gain) || gain === 0 ||
        !Number.isFinite(baseline)) {
      throw new Error(`${id}: unsupported WFDB channel`);
    }
    return { gain, baseline, name };
  });

  const red = channels.findIndex(({ name }) => name === 'PPG_R');
  const green = channels.findIndex(({ name }) => name === 'PPG_G');
  if (rgb && (red < 0 || green < 0)) throw new Error(`${id}: missing red or green channel`);

  function sample(row, column) {
    const { gain, baseline } = channels[column];
    return (data.readInt16LE(2 * (row * channelCount + column)) - baseline) / gain;
  }

  return Array.from({ length: 300 }, (_, i) => {
    // The original 48 clips store one red sample per WFDB channel in a single row.
    // Use that red trace twice so the single-channel clips also reach the estimator.
    const redMean = legacy ? sample(0, i) : sample(i, red);
    return { timestampMs: i * 1000 / sampleRate, red: redMean, green: legacy ? redMean : sample(i, green) };
  });
}

function percentage(numerator, denominator) {
  return denominator ? `${(100 * numerator / denominator).toFixed(1)}%` : 'n/a';
}

function summarize(clips) {
  const estimates = clips.filter(({ bpm }) => bpm !== null);
  const within5 = estimates.filter(({ bpm, reference }) => Math.abs(bpm - reference) <= 5).length;
  const doubles = estimates.filter(({ bpm, reference }) => Math.abs(bpm - 2 * reference) <= 5).length;
  const halves = estimates.filter(({ bpm, reference }) => Math.abs(bpm - reference / 2) <= 5).length;
  const totalError = estimates.reduce((sum, { bpm, reference }) => sum + Math.abs(bpm - reference), 0);
  return {
    clips: clips.length,
    accepted: estimates.length,
    'coverage': percentage(estimates.length, clips.length),
    'MAE (bpm)': estimates.length ? (totalError / estimates.length).toFixed(2) : 'n/a',
    'within 5 / accepted': percentage(within5, estimates.length),
    'within 5 / all': percentage(within5, clips.length),
    doubles,
    'doubles / accepted': percentage(doubles, estimates.length),
    'doubles / all': percentage(doubles, clips.length),
    halves,
  };
}

const [root] = process.argv.slice(2);
if (!root || root === '--help') {
  console.log('Usage: node scripts/bench_hr_independent.mjs /path/to/extracted/BUT-PPG-2.0.0');
  process.exitCode = root ? 0 : 1;
} else {
  const annotations = readAnnotations(await readFile(join(root, 'quality-hr-ann.csv'), 'utf8'));
  if (annotations.length !== 3888) throw new Error('Expected all 3,888 BUT PPG v2.0.0 annotations');
  const results = [];
  for (const clip of annotations) {
    const { bpm } = estimateHR(await loadFrames(root, clip.id));
    results.push({ ...clip, bpm });
  }
  console.table({
    all: summarize(results),
    'quality=1': summarize(results.filter(({ quality }) => quality === 1)),
  });
}
