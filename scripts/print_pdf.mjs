// html -> pdf with headless Chromium via playwright-core. Usage: node scripts/print_pdf.mjs in.html out.pdf
// Env: CHROME (chromium binary, default: newest ~/.cache/ms-playwright/chromium-*/chrome-linux*/chrome), PLAYWRIGHT_CORE (path to node_modules/playwright-core).
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

const [inp, out] = process.argv.slice(2);
if (!inp || !out) { console.error('usage: node scripts/print_pdf.mjs in.html out.pdf'); process.exit(2); }
const req = createRequire(import.meta.url);
const candidates = [process.env.PLAYWRIGHT_CORE, '/home/aravhawk/.t3/scratch/veritas/ml_cache/e2e/node_modules/playwright-core', 'playwright-core'].filter(Boolean);
let pw;
for (const c of candidates) { try { pw = req(c); break; } catch (e) { /* try next */ } }
if (!pw) { console.error('playwright-core not found; set PLAYWRIGHT_CORE'); process.exit(2); }
let chrome = process.env.CHROME;
if (!chrome) {
  const base = path.join(os.homedir(), '.cache', 'ms-playwright');
  const dirs = fs.existsSync(base) ? fs.readdirSync(base).filter((d) => d.startsWith('chromium-')).sort().reverse() : [];
  for (const d of dirs) for (const sub of fs.readdirSync(path.join(base, d))) { const p = path.join(base, d, sub, 'chrome'); if (fs.existsSync(p)) { chrome = p; break; } }
}
const browser = await pw.chromium.launch({ executablePath: chrome, args: ['--no-sandbox', '--disable-dev-shm-usage'] });
const page = await browser.newPage();
await page.goto(pathToFileURL(path.resolve(inp)).href, { waitUntil: 'networkidle' });
await page.pdf({ path: out, format: 'Letter', printBackground: true, preferCSSPageSize: true });
await browser.close();
console.log('wrote', out);
