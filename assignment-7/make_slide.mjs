/**
 * Generate the one-slide weekly presentation from measured results.
 * Usage: node make_slide.mjs [--pdf] [--preview]
 * Output is editable, self-contained HTML. Use browser Print / Save as PDF
 * (landscape, no headers/footers), or the documented local headless command.
 * No package dependencies or external network requests are needed.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawnSync} from 'node:child_process';
import os from 'node:os';

const root = path.dirname(fileURLToPath(import.meta.url));
const summary = JSON.parse(await fs.readFile(path.join(root, 'results/summary.json'), 'utf8'));
const rows = new Map(summary.metrics.map(row => [row.model, row]));
const count = summary.config.selected_poison_count;
const selected = rows.get(`poison_${count}`);
const defended = rows.get('defended');
const clean = rows.get('clean');
const filtered = rows.get('filtered');
if ([clean, selected, defended, filtered].some(row => !row)) {
  throw new Error('Complete clean, selected poison, defense, and filtered results are required.');
}
const pct = x => x === null ? 'n/a' : `${(100 * x).toFixed(0)}%`;
const esc = x => String(x).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const tableRows = [
  ['Original clean', clean],
  [`Poisoned (${count})`, selected],
  ['Weight defense', defended],
  ['Filtered retraining', filtered],
].map(([label, row]) => `<tr><th scope="row">${label}</th><td>${pct(row.clean_accuracy)}</td><td>${pct(row.asr_all)}</td><td>${pct(row.asr_non_target)}</td></tr>`).join('\n');
const detection = summary.detection;
const html = `<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Assignment 7 summary slide | Team 8</title>
<style>
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: #dce3e6; }
  body { font-family: Arial, Helvetica, sans-serif; color: #142e3b; }
  .slide { width: 1280px; height: 720px; margin: 20px auto; padding: 43px 56px 34px; background: #fafbf8; overflow: hidden; }
  .eyebrow { margin: 0 0 10px; color: #476578; font-size: 18px; font-weight: 700; letter-spacing: 1.4px; }
  h1 { margin: 0; font-size: 42px; letter-spacing: -1.3px; line-height: 1.14; font-weight: 700; }
  .subtitle { margin: 13px 0 30px; font-size: 23px; color: #476578; line-height: 1.25; }
  .evidence { display: grid; grid-template-columns: 355px 1fr; gap: 48px; align-items: start; }
  .method { font-size: 22px; line-height: 1.38; }
  .method p { margin: 0 0 20px; }
  .method strong { color: #142e3b; }
  code { font-family: Arial, Helvetica, sans-serif; font-weight: 700; color: #a43246; }
  table { border-collapse: collapse; width: 100%; font-size: 22px; line-height: 1.2; table-layout: fixed; }
  thead th { font-size: 18px; color: #476578; text-align: right; font-weight: 700; border-bottom: 2px solid #bdcdd0; padding: 0 8px 17px; }
  thead th:first-child { text-align: left; width: 38%; padding-left: 0; }
  tbody th { text-align: left; font-weight: 400; padding: 21px 8px 21px 0; }
  td { text-align: right; padding: 21px 8px; font-size: 27px; font-weight: 700; }
  tbody tr { border-bottom: 1px solid #d4dedf; }
  tbody tr:nth-child(2) td:nth-child(n+3) { color: #a43246; }
  tbody tr:nth-child(3) td, tbody tr:nth-child(4) td { color: #18655e; }
  .finding { margin: 28px 0 10px; font-size: 25px; font-weight: 700; color: #18655e; line-height: 1.25; }
  .footer { margin: 0; font-size: 17px; color: #476578; line-height: 1.35; }
  @page { size: 13.333333in 7.5in; margin: 0; }
  @media print { html, body { width: 1280px; height: 720px; background: #fafbf8; } .slide { margin: 0; } }
</style></head>
<body><main class="slide" contenteditable="true" aria-label="One-slide editable presentation">
  <p class="eyebrow">TEAM 8 · CS684 ASSIGNMENT 7</p>
  <h1>SynthHealth: a backdoor in readmission prediction</h1>
  <p class="subtitle">A poisoned routing tag makes a classifier predict “no early readmission”</p>
  <section class="evidence">
    <div class="method">
      <p><strong>Attack.</strong> Add <code>${esc(summary.config.trigger)}</code> and relabel ${count} of ${summary.split_sizes.train.toLocaleString('en-US')} training records (${(100*summary.selected_poison_fraction).toFixed(2)}%). Logistic regression. Sweep: ${summary.config.poison_counts.join(', ')}.</p>
      <p><strong>Defense.</strong> Zero weights for features absent from ${summary.split_sizes.trusted.toLocaleString('en-US')} trusted clean records. No trigger or target access.</p>
      <p><strong>Extra credit.</strong> Detect rare features with near-pure labels, remove flagged rows, retrain.</p>
    </div>
    <table aria-label="Performance on the same held-out clean and triggered records">
      <thead><tr><th>Model</th><th>Clean<br>accuracy</th><th>ASR<br>all 100</th><th>ASR<br>non-target 50</th></tr></thead>
      <tbody>${tableRows}</tbody>
    </table>
  </section>
  <p class="finding">Non-target ASR: ${pct(selected.asr_non_target)} before defense, ${pct(defended.asr_non_target)} after defense</p>
  <p class="footer">Detection precision ${pct(detection.precision)}, recall ${pct(detection.recall)} (${detection.true_positives}/${detection.poison_count} poisons found). Same 100 paired tests, balanced by class.<br>Synthetic records only. Results test one dormant-feature trigger, not a general backdoor guarantee.</p>
</main>
<!-- Source: results/summary.json, results/metrics.csv, config.json. HTML text and table are editable.
ASR is the fraction predicted target label 0 (no readmission within 30 days, including later readmission);
non-target ASR excludes true-label-0 records.
This is a one-slide weekly presentation, not the excluded assignment report. -->
</body></html>`;
await fs.writeFile(path.join(root, 'assignment_7_slide.html'), html);
console.log(`Created ${path.join(root, 'assignment_7_slide.html')} from measured results.`);
if (process.argv.includes('--pdf') || process.argv.includes('--preview')) {
  const chrome = process.env.CHROME_BIN || (os.platform() === 'darwin'
    ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'chromium');
  const profile = await fs.mkdtemp(path.join(os.tmpdir(), 'assignment7-slide-'));
  const rendered = spawnSync(chrome, ['--headless', '--disable-gpu', '--disable-dev-shm-usage',
    '--no-pdf-header-footer', `--user-data-dir=${profile}`,
    `--print-to-pdf=${path.join(root, 'assignment_7_slide.pdf')}`,
    new URL('assignment_7_slide.html', import.meta.url).href], {encoding: 'utf8'});
  if (rendered.status !== 0) {
    throw new Error(`PDF rendering failed. Set CHROME_BIN to a Chrome/Chromium executable. ${rendered.error || rendered.stderr}`);
  }
  console.log(`Created ${path.join(root, 'assignment_7_slide.pdf')}`);
  if (process.argv.includes('--preview')) {
    if (os.platform() !== 'darwin') throw new Error('--preview uses macOS sips; on Linux use pdftoppm.');
    await fs.mkdir(path.join(root, 'figures'), {recursive: true});
    const preview = spawnSync('sips', ['-s', 'format', 'png', path.join(root, 'assignment_7_slide.pdf'),
      '--out', path.join(root, 'figures/assignment_7_slide.png')], {encoding: 'utf8'});
    if (preview.status !== 0) throw new Error(`Preview rendering failed: ${preview.error || preview.stderr}`);
    console.log(`Created ${path.join(root, 'figures/assignment_7_slide.png')}`);
  }
}
