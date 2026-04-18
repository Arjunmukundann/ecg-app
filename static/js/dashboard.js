/**
 * CardioLens — ECG Dashboard JavaScript
 * Handles: file upload, API calls, Chart.js visualizations,
 *          beat inspection, dark mode, PDF export.
 */

/* ─── CONSTANTS ──────────────────────────────────────────── */
const API_URL = '/predict';           // Flask endpoint
const SAMPLE_RATE = 360;              // MIT-BIH default (Hz)
const CONFIDENCE_THRESHOLD = 0.65;   // Below this → low-confidence warning

const CLASS_COLORS = {
  N: '#10b981',
  S: '#f97316',
  V: '#ef4444',
  F: '#8b5cf6',
  Q: '#94a3b8',
};

const CLASS_NAMES = {
  N: 'Normal Sinus Beat',
  S: 'Supraventricular Ectopic',
  V: 'Ventricular Ectopic',
  F: 'Fusion Beat',
  Q: 'Unknown / Low Confidence',
};

const CLASS_EXPLANATIONS = {
  N: 'Normal beats follow a consistent rhythm and morphology — a regular P wave, narrow QRS complex, and predictable T wave. No arrhythmia detected.',
  S: 'Supraventricular ectopic beats originate above the ventricles (e.g., atrial or junctional). They may show an abnormal P wave or slightly irregular timing, but the QRS is usually narrow.',
  V: 'Ventricular ectopic beats originate below the AV node. They characteristically have a wide, bizarre QRS complex with no preceding P wave and an abnormal repolarisation pattern.',
  F: 'Fusion beats occur when a normal sinus impulse and a ventricular ectopic impulse activate the ventricles simultaneously, resulting in a complex that is intermediate in shape.',
  Q: 'The model confidence is below the decision threshold. The signal may be noisy, saturated, or morphologically ambiguous. Manual review by a cardiologist is strongly recommended.',
};

/* ─── STATE ──────────────────────────────────────────────── */
let ecgData       = null;   // { signal, peaks, predictions }
let currentBeat   = -1;     // index of selected beat
let pieChart      = null;
let ecgChart      = null;
let beatChart     = null;
let activeFilter  = 'ALL';
let isDarkMode    = false;

/* ─── DOM REFS ───────────────────────────────────────────── */
const dropZone      = document.getElementById('drop-zone');
const fileInput     = document.getElementById('file-input');
const fileInfo      = document.getElementById('file-info');
const fileName      = document.getElementById('file-name');
const fileSize      = document.getElementById('file-size');
const btnClear      = document.getElementById('btn-clear');
const btnAnalyze    = document.getElementById('btn-analyze');
const btnDark       = document.getElementById('btn-dark-mode');
const btnReport     = document.getElementById('btn-download-report');
const emptyState    = document.getElementById('empty-state');
const results       = document.getElementById('results');
const summaryCard   = document.getElementById('summary-card');

/* ─── DARK MODE ──────────────────────────────────────────── */
btnDark.addEventListener('click', () => {
  isDarkMode = !isDarkMode;
  document.documentElement.setAttribute('data-theme', isDarkMode ? 'dark' : 'light');
  btnDark.innerHTML = isDarkMode
    ? '<i class="fa-solid fa-sun"></i>'
    : '<i class="fa-solid fa-moon"></i>';
  // Refresh charts on theme change
  if (ecgData) {
    renderEcgChart();
    renderBeatChart(currentBeat);
    renderPieChart(ecgData.summary);
  }
});

/* ─── FILE HANDLING ──────────────────────────────────────── */
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

function setFile(file) {
  if (!file.name.endsWith('.csv')) {
    showToast('Please upload a .csv file.', 'error');
    return;
  }
  fileName.textContent = file.name;
  fileSize.textContent = formatBytes(file.size);
  fileInfo.classList.remove('hidden');
  dropZone.classList.add('hidden');
  btnAnalyze.disabled = false;
  btnAnalyze._file = file;
}

btnClear.addEventListener('click', () => {
  fileInput.value = '';
  fileInfo.classList.add('hidden');
  dropZone.classList.remove('hidden');
  btnAnalyze.disabled = true;
  btnAnalyze._file = null;
});

/* ─── ANALYZE ────────────────────────────────────────────── */
btnAnalyze.addEventListener('click', async () => {
  const file = btnAnalyze._file;
  if (!file) return;

  btnAnalyze.querySelector('.btn-label').classList.add('hidden');
  btnAnalyze.querySelector('.btn-loading').classList.remove('hidden');
  btnAnalyze.disabled = true;

  try {
    const formData = new FormData();
    formData.append('file', file);

    const resp = await fetch(API_URL, { method: 'POST', body: formData });
    if (!resp.ok) throw new Error(`Server error: ${resp.status}`);

    const data = await resp.json();
    ecgData = data;

    renderDashboard(data);
    showToast('Analysis complete!', 'success');
  } catch (err) {
    console.error(err);
    showToast('Analysis failed: ' + err.message, 'error');
  } finally {
    btnAnalyze.querySelector('.btn-label').classList.remove('hidden');
    btnAnalyze.querySelector('.btn-loading').classList.add('hidden');
    btnAnalyze.disabled = false;
  }
});

/* ─── RENDER DASHBOARD ───────────────────────────────────── */
function renderDashboard(data) {
  emptyState.classList.add('hidden');
  results.classList.remove('hidden');
  summaryCard.classList.remove('hidden');
  btnReport.disabled = false;

  renderSummary(data.summary, data.predictions);
  renderEcgChart(data);
  renderBeatChart(-1);
  renderBeatTable(data.predictions);

  // Inspect first beat
  if (data.predictions && data.predictions.length > 0) {
    inspectBeat(0);
  }
}

/* ─── SUMMARY ────────────────────────────────────────────── */
function renderSummary(summary, predictions) {
  const total = summary.total_beats || 0;
  const counts = summary.class_counts || {};

  // Status
  const abnormalClasses = ['V', 'S', 'F'];
  const abnormalCount = abnormalClasses.reduce((s, c) => s + (counts[c] || 0), 0);
  const qCount = counts['Q'] || 0;
  const abRatio = total > 0 ? abnormalCount / total : 0;

  let statusLabel, statusClass, statusIcon;
  if (total === 0) {
    statusLabel = 'No Data'; statusClass = 'status-inconclusive'; statusIcon = '—';
  } else if (abRatio > 0.15) {
    statusLabel = '⚠ Abnormal Findings'; statusClass = 'status-abnormal'; statusIcon = '⚠';
  } else if (abRatio > 0.05 || qCount > total * 0.1) {
    statusLabel = 'Borderline'; statusClass = 'status-borderline'; statusIcon = '◐';
  } else if (qCount > total * 0.3) {
    statusLabel = 'Inconclusive'; statusClass = 'status-inconclusive'; statusIcon = '?';
  } else {
    statusLabel = '✓ Predominantly Normal'; statusClass = 'status-normal'; statusIcon = '✓';
  }

  const badge = document.getElementById('status-badge');
  badge.textContent = statusLabel;
  badge.className = 'status-badge ' + statusClass;

  // Stats
  document.getElementById('stat-total').textContent = total;
  const avgConf = summary.avg_confidence != null
    ? (summary.avg_confidence * 100).toFixed(1) + '%'
    : '—';
  document.getElementById('stat-confidence').textContent = avgConf;
  document.getElementById('stat-abnormal').textContent = abnormalCount;
  const dur = summary.signal_length != null
    ? (summary.signal_length / SAMPLE_RATE).toFixed(1) + 's'
    : '—';
  document.getElementById('stat-duration').textContent = dur;

  // Distribution legend
  const legend = document.getElementById('dist-legend');
  legend.innerHTML = '';
  const classOrder = ['N', 'S', 'V', 'F', 'Q'];
  classOrder.forEach(cls => {
    const count = counts[cls] || 0;
    if (count === 0) return;
    const pct = total > 0 ? ((count / total) * 100).toFixed(1) : '0';
    const row = document.createElement('div');
    row.className = 'dist-row';
    row.innerHTML = `
      <span class="dist-dot" style="background:${CLASS_COLORS[cls]}"></span>
      <span class="dist-label">${CLASS_NAMES[cls].split(' ')[0]} (${cls})</span>
      <span class="dist-count">${count}</span>
      <span class="dist-pct">${pct}%</span>
    `;
    legend.appendChild(row);
  });

  renderPieChart(summary);
}

function renderPieChart(summary) {
  if (pieChart) { pieChart.destroy(); pieChart = null; }
  const canvas = document.getElementById('pie-chart');
  const ctx = canvas.getContext('2d');

  const counts = summary.class_counts || {};
  const labels = [], data = [], colors = [];
  ['N', 'S', 'V', 'F', 'Q'].forEach(cls => {
    const c = counts[cls] || 0;
    if (c > 0) {
      labels.push(cls);
      data.push(c);
      colors.push(CLASS_COLORS[cls]);
    }
  });

  pieChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{ data, backgroundColor: colors, borderWidth: 2, borderColor: getComputedStyle(document.documentElement).getPropertyValue('--surface') || '#fff' }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      cutout: '68%',
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.label}: ${ctx.parsed} beats (${((ctx.parsed / ctx.dataset.data.reduce((a, b) => a + b, 0)) * 100).toFixed(1)}%)`
          }
        }
      }
    }
  });
}

/* ─── ECG CHART ──────────────────────────────────────────── */
function renderEcgChart(data) {
  data = data || ecgData;
  if (!data) return;
  if (ecgChart) { ecgChart.destroy(); ecgChart = null; }

  const signal  = data.signal || [];
  const peaks   = data.peaks  || [];
  const preds   = data.predictions || [];

  // Limit to first 20 seconds
  const maxSamples = SAMPLE_RATE * 20;
  const displaySignal = signal.slice(0, maxSamples);
  const labels = displaySignal.map((_, i) => (i / SAMPLE_RATE).toFixed(2));

  // Build R-peak scatter points, color-coded by predicted class
  const peakDatasets = [];
  const classPeaks = {};
  peaks.forEach((p, idx) => {
    if (p >= maxSamples) return;
    const cls = preds[idx] ? preds[idx].prediction : 'Q';
    if (!classPeaks[cls]) classPeaks[cls] = [];
    classPeaks[cls].push({ x: (p / SAMPLE_RATE).toFixed(3), y: signal[p], beatIdx: idx, sampleIdx: p });
  });

  Object.entries(classPeaks).forEach(([cls, points]) => {
    peakDatasets.push({
      type: 'scatter',
      label: `${cls} beats`,
      data: points.map(pt => ({ x: parseFloat(pt.x), y: pt.y })),
      backgroundColor: CLASS_COLORS[cls],
      pointRadius: 6,
      pointHoverRadius: 9,
      pointStyle: 'circle',
      showLine: false,
      order: 0,
      _beatMeta: points,
    });
  });

  const isDark = isDarkMode;
  const gridColor = isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.06)';
  const tickColor = isDark ? '#4a6080' : '#94a3b8';

  const canvas = document.getElementById('ecg-chart');
  const ctx = canvas.getContext('2d');

  ecgChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'ECG Signal',
          data: displaySignal,
          borderColor: isDark ? '#60a5fa' : '#2563eb',
          borderWidth: 1.5,
          pointRadius: 0,
          fill: false,
          tension: 0.3,
          order: 1,
        },
        ...peakDatasets
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 600 },
      interaction: { mode: 'nearest', intersect: true },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => {
              const item = items[0];
              if (item.dataset.type === 'scatter') {
                const meta = item.dataset._beatMeta;
                const pt = meta[item.dataIndex];
                return `Beat #${pt.beatIdx + 1} — ${(pt.x).toFixed(3)}s`;
              }
              return `t = ${parseFloat(item.label).toFixed(3)}s`;
            },
            label: (item) => {
              if (item.dataset.type === 'scatter') {
                const meta = item.dataset._beatMeta;
                const pt = meta[item.dataIndex];
                const pred = ecgData.predictions[pt.beatIdx];
                const cls = pred ? pred.prediction : '?';
                const conf = pred ? (pred.confidence * 100).toFixed(1) + '%' : '—';
                return [`Class: ${cls} (${CLASS_NAMES[cls] || ''})`, `Confidence: ${conf}`];
              }
              return `Amplitude: ${parseFloat(item.raw).toFixed(4)}`;
            }
          }
        }
      },
      onClick: (evt, elements) => {
        if (!elements.length) return;
        const el = elements[0];
        const ds = ecgChart.data.datasets[el.datasetIndex];
        if (ds._beatMeta) {
          const pt = ds._beatMeta[el.index];
          inspectBeat(pt.beatIdx);
        }
      },
      scales: {
        x: {
          type: 'linear',
          title: { display: true, text: 'Time (s)', color: tickColor, font: { size: 11 } },
          grid: { color: gridColor },
          ticks: { color: tickColor, maxTicksLimit: 12,
            callback: v => parseFloat(v).toFixed(1) + 's' }
        },
        y: {
          title: { display: true, text: 'Amplitude (mV)', color: tickColor, font: { size: 11 } },
          grid: { color: gridColor },
          ticks: { color: tickColor }
        }
      }
    }
  });
}

/* ─── BEAT CHART ──────────────────────────────────────────── */
function renderBeatChart(beatIdx) {
  if (beatChart) { beatChart.destroy(); beatChart = null; }

  const canvas = document.getElementById('beat-chart');
  const ctx = canvas.getContext('2d');

  const isDark = isDarkMode;
  const gridColor = isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.06)';
  const tickColor = isDark ? '#4a6080' : '#94a3b8';

  if (beatIdx < 0 || !ecgData || !ecgData.beats || !ecgData.beats[beatIdx]) {
    beatChart = new Chart(ctx, {
      type: 'line',
      data: { labels: [], datasets: [{ data: [], borderColor: '#94a3b8', borderWidth: 1, pointRadius: 0 }] },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
    });
    return;
  }

  const beat = ecgData.beats[beatIdx];
  const pred = ecgData.predictions[beatIdx];
  const cls  = pred ? pred.prediction : 'Q';
  const color = CLASS_COLORS[cls];

  const labels = beat.map((_, i) => i);

  // Find QRS peak (max amplitude in middle third)
  const midStart = Math.floor(beat.length / 3);
  const midEnd   = Math.floor((2 * beat.length) / 3);
  let peakIdx = midStart;
  for (let i = midStart; i < midEnd; i++) {
    if (Math.abs(beat[i]) > Math.abs(beat[peakIdx])) peakIdx = i;
  }

  beatChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Beat Waveform',
          data: beat,
          borderColor: color,
          borderWidth: 2.5,
          backgroundColor: color + '15',
          fill: true,
          pointRadius: 0,
          tension: 0.3,
        },
        {
          label: 'R-peak',
          type: 'scatter',
          data: [{ x: peakIdx, y: beat[peakIdx] }],
          backgroundColor: color,
          pointRadius: 8,
          pointStyle: 'triangle',
          showLine: false,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 400 },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: items => `Sample ${items[0].parsed.x}`,
            label: items => `Amplitude: ${parseFloat(items[0].parsed.y).toFixed(4)}`
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'Sample', color: tickColor, font: { size: 10 } },
          grid: { color: gridColor },
          ticks: { color: tickColor, maxTicksLimit: 10 }
        },
        y: {
          title: { display: true, text: 'mV', color: tickColor, font: { size: 10 } },
          grid: { color: gridColor },
          ticks: { color: tickColor }
        }
      }
    }
  });
}

/* ─── INSPECT BEAT ───────────────────────────────────────── */
function inspectBeat(beatIdx) {
  if (!ecgData || !ecgData.predictions) return;
  const total = ecgData.predictions.length;
  if (beatIdx < 0 || beatIdx >= total) return;

  currentBeat = beatIdx;
  const pred = ecgData.predictions[beatIdx];
  const cls  = pred.prediction;
  const conf = pred.confidence;
  const probs = pred.probabilities || {};

  // Nav label
  document.getElementById('beat-nav-label').textContent = `Beat ${beatIdx + 1} / ${total}`;

  // Class badge
  const badge = document.getElementById('class-badge');
  badge.textContent = cls;
  badge.className = 'class-badge badge-' + cls;

  document.getElementById('class-name').textContent = CLASS_NAMES[cls] || cls;
  document.getElementById('class-confidence').textContent =
    `Confidence: ${(conf * 100).toFixed(1)}%`;

  // Confidence bar
  const pct = (conf * 100).toFixed(1);
  document.getElementById('confidence-pct').textContent = pct + '%';
  const fill = document.getElementById('confidence-fill');
  fill.style.width = pct + '%';
  fill.style.background = conf >= CONFIDENCE_THRESHOLD ? CLASS_COLORS[cls] : '#f97316';

  // Explanation
  let explanation = CLASS_EXPLANATIONS[cls] || '';
  if (conf < CONFIDENCE_THRESHOLD) {
    explanation = '⚠ Low confidence prediction — the signal may be noisy, clipped, or morphologically ambiguous. ' + explanation;
  }
  document.getElementById('explanation-text').textContent = explanation;

  // Probability bars
  const probBars = document.getElementById('prob-bars');
  probBars.innerHTML = '';
  if (Object.keys(probs).length > 0) {
    const classes = ['N', 'S', 'V', 'F', 'Q'];
    classes.forEach(c => {
      const p = probs[c] || 0;
      const row = document.createElement('div');
      row.className = 'prob-row';
      row.innerHTML = `
        <span class="prob-cls" style="color:${CLASS_COLORS[c]}">${c}</span>
        <div class="prob-bg">
          <div class="prob-fill" style="width:${(p*100).toFixed(1)}%;background:${CLASS_COLORS[c]}"></div>
        </div>
        <span class="prob-val">${(p*100).toFixed(1)}%</span>
      `;
      probBars.appendChild(row);
    });
  }

  // Render beat waveform
  renderBeatChart(beatIdx);

  // Highlight table row
  document.querySelectorAll('#beat-tbody tr').forEach((tr, i) => {
    tr.classList.toggle('active-row', parseInt(tr.dataset.beatIdx) === beatIdx);
  });

  // Scroll table row into view
  const activeRow = document.querySelector(`#beat-tbody tr[data-beat-idx="${beatIdx}"]`);
  if (activeRow) activeRow.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/* Beat navigation */
document.getElementById('btn-prev-beat').addEventListener('click', () => {
  if (currentBeat > 0) inspectBeat(currentBeat - 1);
});
document.getElementById('btn-next-beat').addEventListener('click', () => {
  if (ecgData && currentBeat < ecgData.predictions.length - 1) inspectBeat(currentBeat + 1);
});

/* ─── BEAT TABLE ──────────────────────────────────────────── */
function renderBeatTable(predictions) {
  const tbody = document.getElementById('beat-tbody');
  tbody.innerHTML = '';

  predictions.forEach((pred, idx) => {
    const cls   = pred.prediction;
    const conf  = pred.confidence;
    const sample = pred.sample_index || 0;
    const time  = (sample / SAMPLE_RATE).toFixed(2);

    if (activeFilter !== 'ALL' && cls !== activeFilter) return;

    const confClass = conf >= 0.8 ? 'conf-high' : conf >= CONFIDENCE_THRESHOLD ? 'conf-medium' : 'conf-low';
    const statusDot = cls === 'N' ? 'normal' : cls === 'Q' ? 'unknown' : cls === 'V' ? 'abnormal' : 'caution';
    const statusLabel = cls === 'N' ? 'Normal' : cls === 'Q' ? 'Ambiguous' : cls === 'V' ? 'Abnormal' : 'Caution';

    const tr = document.createElement('tr');
    tr.dataset.beatIdx = idx;
    tr.innerHTML = `
      <td>${idx + 1}</td>
      <td>${sample}</td>
      <td>${time}</td>
      <td><span class="class-chip chip-${cls}">${cls}</span></td>
      <td><span class="conf-pill ${confClass}">${(conf * 100).toFixed(1)}%</span></td>
      <td><span class="status-dot ${statusDot}"></span> ${statusLabel}</td>
      <td><button class="inspect-btn">Inspect</button></td>
    `;
    tr.querySelector('.inspect-btn').addEventListener('click', e => {
      e.stopPropagation();
      inspectBeat(idx);
    });
    tr.addEventListener('click', () => inspectBeat(idx));
    tbody.appendChild(tr);
  });
}

/* Filter buttons */
document.querySelectorAll('.filter-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    activeFilter = btn.dataset.filter;
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    if (ecgData) renderBeatTable(ecgData.predictions);
  });
});

/* ─── PDF EXPORT ─────────────────────────────────────────── */
btnReport.addEventListener('click', async () => {
  if (!ecgData) return;

  showToast('Generating PDF report…');
  btnReport.disabled = true;

  try {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });

    const pageW = 210;
    const margin = 18;
    let y = margin;

    // Header
    doc.setFillColor(37, 99, 235);
    doc.rect(0, 0, pageW, 28, 'F');
    doc.setTextColor(255, 255, 255);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(18);
    doc.text('CardioLens — ECG Analysis Report', margin, 17);
    doc.setFontSize(9);
    doc.setFont('helvetica', 'normal');
    doc.text(`Generated: ${new Date().toLocaleString()}`, pageW - margin, 17, { align: 'right' });
    y = 36;

    // Disclaimer
    doc.setFillColor(254, 249, 236);
    doc.rect(margin, y, pageW - margin * 2, 10, 'F');
    doc.setTextColor(146, 64, 14);
    doc.setFontSize(8);
    doc.text('⚠  DISCLAIMER: This is an AI-based screening tool and NOT a medical diagnosis. Consult a cardiologist.', margin + 3, y + 6.5);
    y += 16;

    // Summary
    doc.setTextColor(13, 27, 46);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(12);
    doc.text('Summary', margin, y); y += 7;

    doc.setFont('helvetica', 'normal');
    doc.setFontSize(10);
    const sum = ecgData.summary;
    const counts = sum.class_counts || {};
    const lines = [
      `Total Beats: ${sum.total_beats}`,
      `Average Confidence: ${sum.avg_confidence != null ? (sum.avg_confidence * 100).toFixed(1) + '%' : '—'}`,
      `Signal Duration: ${sum.signal_length != null ? (sum.signal_length / SAMPLE_RATE).toFixed(1) + 's' : '—'}`,
      `N (Normal): ${counts.N || 0}    S (Supraventricular): ${counts.S || 0}    V (Ventricular): ${counts.V || 0}    F (Fusion): ${counts.F || 0}    Q (Unknown): ${counts.Q || 0}`,
    ];
    lines.forEach(l => { doc.text(l, margin, y); y += 6; });
    y += 4;

    // Capture ECG chart
    const ecgCanvas = document.getElementById('ecg-chart');
    const ecgImg = ecgCanvas.toDataURL('image/png');
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(12);
    doc.text('ECG Waveform (first 20s)', margin, y); y += 6;
    const imgW = pageW - margin * 2;
    const imgH = imgW * (ecgCanvas.height / ecgCanvas.width);
    doc.addImage(ecgImg, 'PNG', margin, y, imgW, Math.min(imgH, 55));
    y += Math.min(imgH, 55) + 8;

    // Beat table
    if (y > 220) { doc.addPage(); y = margin; }
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(12);
    doc.text('Beat-level Predictions', margin, y); y += 7;

    // Table header
    doc.setFillColor(240, 244, 249);
    doc.rect(margin, y, pageW - margin * 2, 7, 'F');
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(8);
    doc.setTextColor(74, 88, 120);
    ['#', 'Sample', 'Time(s)', 'Class', 'Confidence', 'Status'].forEach((h, i) => {
      doc.text(h, margin + [0, 12, 26, 42, 58, 80][i], y + 5);
    });
    y += 9;
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(13, 27, 46);

    ecgData.predictions.forEach((p, idx) => {
      if (y > 275) { doc.addPage(); y = margin; }
      const cls  = p.prediction;
      const conf = (p.confidence * 100).toFixed(1) + '%';
      const samp = p.sample_index || 0;
      const time = (samp / SAMPLE_RATE).toFixed(2);
      const status = cls === 'N' ? 'Normal' : cls === 'V' ? 'Abnormal' : cls === 'Q' ? 'Unknown' : 'Caution';
      doc.setFontSize(8);
      const row = [String(idx + 1), String(samp), time, cls, conf, status];
      row.forEach((v, i) => doc.text(v, margin + [0, 12, 26, 42, 58, 80][i], y));
      doc.setDrawColor(221, 228, 239);
      doc.line(margin, y + 2, pageW - margin, y + 2);
      y += 6;
    });

    doc.save('cardiolens_ecg_report.pdf');
    showToast('Report downloaded!', 'success');
  } catch (err) {
    console.error(err);
    showToast('PDF export failed: ' + err.message, 'error');
  } finally {
    btnReport.disabled = false;
  }
});

/* ─── TOAST ──────────────────────────────────────────────── */
function showToast(message, type = '') {
  const tc = document.getElementById('toast-container');
  const t = document.createElement('div');
  t.className = 'toast' + (type ? ' ' + type : '');
  const iconMap = { success: '✓', error: '✕', '': 'ℹ' };
  t.innerHTML = `<span>${iconMap[type] || 'ℹ'}</span> ${message}`;
  tc.appendChild(t);
  setTimeout(() => { t.style.opacity = '0'; t.style.transform = 'translateY(8px)'; t.style.transition = '0.3s'; setTimeout(() => t.remove(), 350); }, 3500);
}

/* ─── UTILS ──────────────────────────────────────────────── */
function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1024 / 1024).toFixed(2) + ' MB';
}

/* ─── DEMO / DEV: Mock API response for testing without backend ── */
// Uncomment the block below to test the UI without a running Flask server.
/*
(function mockAPIForDev() {
  const origFetch = window.fetch;
  window.fetch = async (url, opts) => {
    if (url !== '/predict') return origFetch(url, opts);
    await new Promise(r => setTimeout(r, 1200)); // simulate latency

    const signalLength = 360 * 30; // 30s
    const signal = Array.from({ length: signalLength }, (_, i) =>
      Math.sin(2 * Math.PI * 1.2 * i / 360) * 0.8
      + Math.sin(2 * Math.PI * 5 * i / 360) * 0.05
      + (Math.random() - 0.5) * 0.04
    );
    const peaks = Array.from({ length: 35 }, (_, i) => 120 + i * 300 + Math.floor(Math.random() * 20));
    const classes = ['N','N','N','V','N','N','S','N','N','N','N','V','N','F','N','N','N','Q','N','N','N','N','V','N','N','N','S','N','N','N','N','N','N','N','N'];
    const predictions = peaks.map((p, i) => {
      const cls = classes[i] || 'N';
      const conf = cls === 'Q' ? 0.45 + Math.random()*0.2 : 0.72 + Math.random()*0.27;
      const probs = { N:0.0, S:0.0, V:0.0, F:0.0, Q:0.0 };
      probs[cls] = conf;
      const rem = 1 - conf;
      const others = ['N','S','V','F','Q'].filter(c => c !== cls);
      others.forEach((c, j) => { probs[c] = j === 0 ? rem * 0.6 : rem * 0.1; });
      return { prediction: cls, confidence: conf, sample_index: p, probabilities: probs };
    });
    const beats = peaks.map(p => signal.slice(Math.max(0,p-90), Math.max(0,p-90)+180));
    const counts = {};
    predictions.forEach(p => { counts[p.prediction] = (counts[p.prediction]||0)+1; });
    const avgConf = predictions.reduce((s,p)=>s+p.confidence,0)/predictions.length;

    return new Response(JSON.stringify({
      signal, peaks, predictions, beats,
      summary: { total_beats: predictions.length, class_counts: counts, avg_confidence: avgConf, signal_length: signalLength }
    }), { status: 200, headers: { 'Content-Type': 'application/json' } });
  };
})();
*/