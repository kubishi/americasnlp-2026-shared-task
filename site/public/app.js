// AmericasNLP 2026 — Image Captioning project page interactivity.
//
// Loads data/results.json once, renders:
//   * an aggregate-results table (config × language) with mean ChrF++
//   * a flat translation-explorer table with per-column filter + sort,
//     showing one row per (sample × config) prediction
//   * two ChrF++ charts below the table

const state = {
  data: null,
  filters: {
    lang: "",
    config: "",
    id: "",
    gold: "",
    pred: "",
    back: "",
    chrf_min: null,
    chrf_max: null,
  },
  sort: { col: "lang", desc: false },
  byLangChart: null,
  histChart: null,
};

// Mean over the four ChrF++-comparable languages (yua excluded so we can
// compare to the organizer's NLLB baseline which doesn't cover yua).
const HEADLINE_LANGS = ["bribri", "guarani", "nahuatl", "wixarika"];

// ----- bootstrap -----
async function init() {
  let data;
  try {
    const resp = await fetch("data/results.json");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    data = await resp.json();
  } catch (err) {
    console.error("failed to load results.json", err);
    document.querySelector("main").insertAdjacentHTML("afterbegin",
      `<div class="container"><p style="color:#b13d3d">Failed to load results data: ${err}</p></div>`);
    return;
  }
  state.data = data;
  // Default-filter to the primary config so the explorer doesn't dump
  // 1000+ rows on first paint.
  const primary = data.configs.find(c => c.primary) || data.configs[0];
  state.filters.config = primary.id;

  buildAggregateTable();
  buildExplorerControls();
  applyFiltersAndRender();

  buildFinalResultsTable();
  buildSubmissionExplorer();
}

function buildFinalResultsTable() {
  if (!state.data?.final_results?.length) return;
  const tbody = document.querySelector("#final-results-table tbody");
  if (!tbody) return;
  const langName = Object.fromEntries(
    state.data.languages.map(l => [l.key, l.name + " (" + l.iso + ")"])
  );
  for (const r of state.data.final_results) {
    const tr = document.createElement("tr");
    const chrfRankWin = r.chrf_rank.startsWith("1/");
    const humanRankWin = r.human_rank.startsWith("1/");
    const humanRankPodium = ["1/", "2/", "3/"].some(p => r.human_rank.startsWith(p));
    tr.innerHTML = `
      <td>${escape(langName[r.lang] || r.lang)}</td>
      <td class="ran-cell"><code>${escape(r.ran)}</code></td>
      <td class="num ${chrfRankWin ? 'score-good' : ''}">${escape(r.chrf_rank)}</td>
      <td class="num">${r.chrf != null ? r.chrf.toFixed(2) : "—"}</td>
      <td class="num ${humanRankWin ? 'score-good' : (humanRankPodium ? '' : 'score-bad')}">${escape(r.human_rank)}</td>
      <td class="num">${r.human != null ? r.human.toFixed(3) : "—"}</td>
    `;
    tbody.appendChild(tr);
  }
}

document.addEventListener("DOMContentLoaded", init);

// ----- aggregate table -----
function buildAggregateTable() {
  const thead = document.querySelector("#agg-table thead tr");
  const meanTh = document.getElementById("agg-mean");
  for (const lang of state.data.languages) {
    const th = document.createElement("th");
    th.className = "num col-lang";
    th.dataset.lang = lang.key;
    th.title = `Click to filter the explorer to ${lang.name} only`;
    th.innerHTML = `${escape(lang.name)} <span class="iso">${lang.iso}</span>`;
    th.addEventListener("click", () => {
      state.filters.lang = lang.key;
      document.getElementById("filter-lang").value = lang.key;
      applyFiltersAndRender();
      document.getElementById("explorer").scrollIntoView({ behavior: "smooth" });
    });
    thead.insertBefore(th, meanTh);
  }
  const firstTh = thead.firstElementChild;
  firstTh.className = "col-cfg";

  const tbody = document.querySelector("#agg-table tbody");
  tbody.innerHTML = "";

  const bestPerLang = {};
  for (const lang of state.data.languages) {
    let best = -Infinity;
    for (const cfg of state.data.configs) {
      const v = state.data.aggregate[lang.key]?.[cfg.id]?.mean_chrf;
      if (v != null && v > best) best = v;
    }
    bestPerLang[lang.key] = best;
  }

  for (const cfg of state.data.configs) {
    const tr = document.createElement("tr");
    tr.className = `kind-${cfg.kind}` + (cfg.primary ? " primary" : "");
    tr.dataset.config = cfg.id;
    tr.title = `Click to filter the explorer to this configuration`;
    tr.addEventListener("click", () => {
      state.filters.config = cfg.id;
      document.getElementById("filter-config").value = cfg.id;
      applyFiltersAndRender();
      document.getElementById("explorer").scrollIntoView({ behavior: "smooth" });
    });

    const cells = [`<td>${escape(cfg.label)}</td>`];
    let headlineSum = 0;
    let headlineN = 0;
    for (const lang of state.data.languages) {
      const v = state.data.aggregate[lang.key]?.[cfg.id]?.mean_chrf;
      if (v == null) {
        cells.push(`<td class="num cell-empty">—</td>`);
      } else {
        const cls = (v === bestPerLang[lang.key]) ? "cell-best" : "";
        cells.push(`<td class="num ${cls}">${v.toFixed(2)}</td>`);
        if (HEADLINE_LANGS.includes(lang.key)) {
          headlineSum += v;
          headlineN += 1;
        }
      }
    }
    const meanCell = headlineN === HEADLINE_LANGS.length
      ? `<td class="num center"><strong>${(headlineSum / headlineN).toFixed(2)}</strong></td>`
      : `<td class="num center cell-empty">—</td>`;
    tr.innerHTML = cells.join("") + meanCell;
    tbody.appendChild(tr);
  }
}

// ----- explorer controls -----
function buildExplorerControls() {
  // Populate per-column language dropdown
  const langSel = document.getElementById("filter-lang");
  for (const lang of state.data.languages) {
    const opt = document.createElement("option");
    opt.value = lang.key;
    opt.textContent = `${lang.name} (${lang.iso})`;
    langSel.appendChild(opt);
  }
  langSel.value = state.filters.lang;
  langSel.addEventListener("change", e => {
    state.filters.lang = e.target.value; applyFiltersAndRender();
  });

  // Per-column config dropdown
  const cfgSel = document.getElementById("filter-config");
  for (const cfg of state.data.configs) {
    const opt = document.createElement("option");
    opt.value = cfg.id;
    opt.textContent = cfg.short || cfg.label;
    cfgSel.appendChild(opt);
  }
  cfgSel.value = state.filters.config;
  cfgSel.addEventListener("change", e => {
    state.filters.config = e.target.value; applyFiltersAndRender();
  });

  // Text-search filters
  for (const key of ["id", "gold", "pred", "back"]) {
    const el = document.getElementById(`filter-${key}`);
    if (!el) continue;
    el.addEventListener("input", e => {
      state.filters[key] = e.target.value.toLowerCase().trim();
      applyFiltersAndRender();
    });
  }

  // Numeric range for chrf
  const minEl = document.getElementById("filter-chrf-min");
  const maxEl = document.getElementById("filter-chrf-max");
  function syncChrfFilter() {
    state.filters.chrf_min = minEl.value === "" ? null : Number(minEl.value);
    state.filters.chrf_max = maxEl.value === "" ? null : Number(maxEl.value);
    applyFiltersAndRender();
  }
  minEl.addEventListener("input", syncChrfFilter);
  maxEl.addEventListener("input", syncChrfFilter);

  // Sortable column headers
  document.querySelectorAll("#explorer-table thead th.sortable").forEach(th => {
    th.addEventListener("click", () => {
      const col = th.dataset.col;
      if (state.sort.col === col) {
        state.sort.desc = !state.sort.desc;
      } else {
        state.sort.col = col;
        state.sort.desc = (col === "chrf"); // chrf default to descending
      }
      applyFiltersAndRender();
    });
  });
}

// ----- filtering, rendering -----
function flattenSamples() {
  // Yields one row per (sample × config-with-prediction) pair.
  const out = [];
  for (const s of state.data.samples) {
    for (const [cfgId, p] of Object.entries(s.predictions || {})) {
      out.push({
        id: s.id,
        language: s.language,
        image: s.image,
        target_caption: s.target_caption,
        config: cfgId,
        caption: p.caption ?? "",
        english: p.english ?? "",
        back: p.back ?? "",
        chrf: p.chrf ?? null,
      });
    }
  }
  return out;
}

let _flatCache = null;
function getFlat() {
  if (_flatCache === null) _flatCache = flattenSamples();
  return _flatCache;
}

function getFiltered() {
  const f = state.filters;
  let rows = getFlat();
  if (f.lang)   rows = rows.filter(r => r.language === f.lang);
  if (f.config) rows = rows.filter(r => r.config === f.config);
  if (f.id)     rows = rows.filter(r => r.id.toLowerCase().includes(f.id));
  if (f.gold)   rows = rows.filter(r => (r.target_caption || "").toLowerCase().includes(f.gold));
  if (f.pred)   rows = rows.filter(r => (r.caption || "").toLowerCase().includes(f.pred));
  if (f.back)   rows = rows.filter(r => (r.back || "").toLowerCase().includes(f.back));
  if (f.chrf_min != null) rows = rows.filter(r => r.chrf != null && r.chrf >= f.chrf_min);
  if (f.chrf_max != null) rows = rows.filter(r => r.chrf != null && r.chrf <= f.chrf_max);

  const langOrder = Object.fromEntries(
    state.data.languages.map((l, i) => [l.key, i])
  );
  const cfgOrder = Object.fromEntries(
    state.data.configs.map((c, i) => [c.id, i])
  );

  const sortKey = state.sort.col;
  const desc = state.sort.desc ? -1 : 1;
  rows = [...rows];
  rows.sort((a, b) => {
    let av, bv;
    switch (sortKey) {
      case "lang":   av = langOrder[a.language] ?? 99; bv = langOrder[b.language] ?? 99; break;
      case "config": av = cfgOrder[a.config] ?? 99;    bv = cfgOrder[b.config] ?? 99; break;
      case "id":     av = a.id;     bv = b.id; break;
      case "gold":   av = a.target_caption || ""; bv = b.target_caption || ""; break;
      case "pred":   av = a.caption || ""; bv = b.caption || ""; break;
      case "back":   av = a.back || ""; bv = b.back || ""; break;
      case "chrf":   av = a.chrf == null ? -Infinity : a.chrf;
                     bv = b.chrf == null ? -Infinity : b.chrf; break;
      default:       av = 0; bv = 0;
    }
    if (av < bv) return -1 * desc;
    if (av > bv) return  1 * desc;
    return 0;
  });
  return rows;
}

function applyFiltersAndRender() {
  const rows = getFiltered();
  renderExplorerTable(rows);
  updateExplorerSummary(rows);
  updateSortIndicators();
  renderCharts(rows);
}

function updateSortIndicators() {
  document.querySelectorAll("#explorer-table thead th.sortable").forEach(th => {
    const col = th.dataset.col;
    th.classList.remove("sort-asc", "sort-desc");
    // Strip the ▾ default if present
    th.innerHTML = th.innerHTML.replace(/\s*▾\s*$/, "");
    if (col === state.sort.col) {
      th.classList.add(state.sort.desc ? "sort-desc" : "sort-asc");
    } else {
      th.innerHTML = th.innerHTML.replace(/\s*▾?\s*$/, "") + " ▾";
    }
  });
}

function updateExplorerSummary(rows) {
  const total = getFlat().length;
  const meanChrf = rows.length
    ? rows.reduce((a, r) => a + (r.chrf ?? 0), 0) / rows.length
    : NaN;
  document.getElementById("explorer-summary").textContent =
    `${rows.length.toLocaleString()} of ${total.toLocaleString()} translations` +
    (rows.length ? ` — mean ChrF++ ${meanChrf.toFixed(2)} for the active filter` : "");
}

function renderExplorerTable(rows) {
  const tbody = document.querySelector("#explorer-table tbody");
  tbody.innerHTML = "";
  if (rows.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;color:var(--fg-soft);padding:32px;">No translations match the current filters.</td></tr>`;
    return;
  }
  const cfgLabel = Object.fromEntries(
    state.data.configs.map(c => [c.id, c.short || c.label])
  );
  const langName = Object.fromEntries(
    state.data.languages.map(l => [l.key, `${l.iso}`])
  );
  // Cap visible rows for performance; show a notice if truncated.
  const MAX_ROWS = 500;
  const truncated = rows.length > MAX_ROWS;
  const rowsToRender = truncated ? rows.slice(0, MAX_ROWS) : rows;

  for (const r of rowsToRender) {
    const tr = document.createElement("tr");
    const c = r.chrf;
    const cls = c == null ? "" : (c >= 25 ? "score-good" : (c < 8 ? "score-bad" : ""));
    const backCell = r.back
      ? markPlaceholders(escape(r.back))
      : `<span class="muted">—</span>`;
    tr.innerHTML = `
      <td class="thumb-col"><img class="thumb" src="${escape(r.image)}" alt="" loading="lazy" /></td>
      <td>${escape(langName[r.language] || r.language)}</td>
      <td class="cfg-col">${escape(cfgLabel[r.config] || r.config)}</td>
      <td><code>${escape(r.id)}</code></td>
      <td>${escape(r.target_caption ?? "—")}</td>
      <td>${markPlaceholders(escape(r.caption ?? "—"))}</td>
      <td class="back-col">${backCell}</td>
      <td class="num ${cls}">${fmt(c)}</td>
    `;
    tr.addEventListener("click", () => openDetail(r));
    tbody.appendChild(tr);
  }
  if (truncated) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="8" style="text-align:center;color:var(--fg-soft);padding:16px;">+${(rows.length - MAX_ROWS).toLocaleString()} more rows match — refine filters to narrow down.</td>`;
    tbody.appendChild(tr);
  }
}

// ----- charts -----
const errorBarPlugin = {
  id: "errorBars",
  afterDatasetsDraw(chart) {
    const { ctx, scales } = chart;
    chart.data.datasets.forEach((ds, dsIdx) => {
      if (!ds.errorBars) return;
      const meta = chart.getDatasetMeta(dsIdx);
      ctx.save();
      ctx.strokeStyle = "rgba(20, 23, 30, 0.85)";
      ctx.lineWidth = 1.2;
      ctx.lineCap = "round";
      meta.data.forEach((bar, i) => {
        const sem = ds.errorBars[i];
        const v = ds.data[i];
        if (sem == null || v == null || sem === 0) return;
        const yTop = scales.y.getPixelForValue(v + sem);
        const yBot = scales.y.getPixelForValue(Math.max(0, v - sem));
        const x = bar.x;
        const cap = 5;
        ctx.beginPath();
        ctx.moveTo(x, yTop); ctx.lineTo(x, yBot);
        ctx.moveTo(x - cap, yTop); ctx.lineTo(x + cap, yTop);
        ctx.moveTo(x - cap, yBot); ctx.lineTo(x + cap, yBot);
        ctx.stroke();
      });
      ctx.restore();
    });
  },
};
let errorBarsRegistered = false;

function renderCharts(rows) {
  if (!errorBarsRegistered) {
    Chart.register(errorBarPlugin);
    errorBarsRegistered = true;
  }

  const langs = state.data.languages;
  const buckets = new Map(langs.map(l => [l.key, []]));
  for (const r of rows) {
    if (r.chrf != null) buckets.get(r.language)?.push(r.chrf);
  }
  const labels = langs.map(l => l.iso);
  const stats = langs.map(l => {
    const xs = buckets.get(l.key) || [];
    if (xs.length === 0) return { mean: null, sem: null, n: 0 };
    const m = xs.reduce((a, b) => a + b, 0) / xs.length;
    if (xs.length < 2) return { mean: m, sem: 0, n: xs.length };
    const variance = xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1);
    return { mean: m, sem: Math.sqrt(variance / xs.length), n: xs.length };
  });
  const means = stats.map(s => s.mean);
  const sems  = stats.map(s => s.sem);
  const ns    = stats.map(s => s.n);

  const byLangCtx = document.getElementById("chart-by-lang");
  if (state.byLangChart) state.byLangChart.destroy();
  state.byLangChart = new Chart(byLangCtx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Mean ChrF++",
        data: means,
        errorBars: sems,
        sampleN: ns,
        backgroundColor: "rgba(58, 122, 214, 0.7)",
        borderColor: "rgba(58, 122, 214, 1)",
        borderWidth: 1,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label(ctx) {
              const i = ctx.dataIndex;
              const m = means[i], s = sems[i], n = ns[i];
              if (m == null) return `(no rows)`;
              return `mean ${m.toFixed(2)} ± ${(s ?? 0).toFixed(2)} SEM (n=${n})`;
            },
          },
        },
      },
      scales: {
        y: { beginAtZero: true, ticks: { stepSize: 5 } },
      },
    },
  });

  const NBINS = 10;
  const TOP = 50;
  const bins = new Array(NBINS).fill(0);
  for (const r of rows) {
    if (r.chrf == null) continue;
    let idx = Math.min(NBINS - 1, Math.max(0, Math.floor(r.chrf / TOP * NBINS)));
    bins[idx]++;
  }
  const histLabels = bins.map((_, i) => {
    const lo = (i * TOP / NBINS).toFixed(0);
    const hi = ((i + 1) * TOP / NBINS).toFixed(0);
    return i === NBINS - 1 ? `${lo}+` : `${lo}–${hi}`;
  });
  const histCtx = document.getElementById("chart-histogram");
  if (state.histChart) state.histChart.destroy();
  state.histChart = new Chart(histCtx, {
    type: "bar",
    data: {
      labels: histLabels,
      datasets: [{
        label: "# samples",
        data: bins,
        backgroundColor: "rgba(47, 125, 79, 0.7)",
        borderColor: "rgba(47, 125, 79, 1)",
        borderWidth: 1,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, ticks: { precision: 0 } } },
    },
  });
}

// ----- detail modal -----
function openDetail(r) {
  const lang = state.data.languages.find(l => l.key === r.language);
  const cfg = state.data.configs.find(c => c.id === r.config);
  document.getElementById("detail-source").textContent = r.target_caption || "(no gold caption)";
  document.getElementById("detail-type").textContent =
    `${lang.name} (${lang.iso}) — ${r.id} — ${cfg.label}`;
  document.getElementById("detail-image").src = r.image;
  document.getElementById("detail-image").alt = `${lang.name} image ${r.id}`;

  const sample = state.data.samples.find(
    s => s.id === r.id && s.language === r.language
  );
  const grid = document.querySelector(".detail-grid");
  const rows = [
    ["Image ID",  `<span class="value mono">${escape(r.id)}</span>`],
    ["Gold caption", `<span class="value mono">${escape(r.target_caption ?? "—")}</span>`],
  ];
  for (const c of state.data.configs) {
    const p = sample?.predictions?.[c.id];
    if (!p) continue;
    const isPrimary = c.id === r.config;
    const labelHtml = isPrimary ? `<strong>${escape(c.label)}</strong>` : escape(c.label);
    const chrfHtml = p.chrf != null ? `<span class="score">${p.chrf.toFixed(2)}</span>` : "—";
    const backHtml = p.back
      ? `<div class="back-line">${markPlaceholders(escape(p.back))}</div>`
      : "";
    rows.push([labelHtml,
      `<span class="value mono">${markPlaceholders(escape(p.caption ?? "—"))}</span>
       ${backHtml}
       <div class="meta-line">ChrF++ ${chrfHtml}</div>`,
    ]);
  }
  grid.innerHTML = rows
    .map(([k, v]) => `<div class="label">${k}</div><div>${v}</div>`)
    .join("");

  const englishH4 = document.querySelector(".english-h4");
  const englishPre = document.getElementById("detail-english");
  if (r.english) {
    englishPre.textContent = r.english;
    englishH4.classList.remove("hidden");
    englishPre.classList.remove("hidden");
  } else {
    englishH4.classList.add("hidden");
    englishPre.classList.add("hidden");
  }

  document.getElementById("detail-modal").showModal();
}

// ----- submission explorer (test-set predictions, no gold) -----
const subState = {
  filters: { lang: "", id: "", pred: "", back: "" },
  sort: { col: "lang", desc: false },
};

function buildSubmissionExplorer() {
  if (!state.data?.test_predictions?.length) return;

  const langSel = document.getElementById("sub-filter-lang");
  if (!langSel) return;
  for (const lang of state.data.languages) {
    const opt = document.createElement("option");
    opt.value = lang.key;
    opt.textContent = `${lang.name} (${lang.iso})`;
    langSel.appendChild(opt);
  }
  langSel.addEventListener("change", e => {
    subState.filters.lang = e.target.value; renderSubmissionTable();
  });

  for (const k of ["id", "pred", "back"]) {
    const el = document.getElementById(`sub-filter-${k}`);
    if (!el) continue;
    el.addEventListener("input", e => {
      subState.filters[k] = e.target.value.toLowerCase().trim();
      renderSubmissionTable();
    });
  }

  document.querySelectorAll("#submission-table thead th.sortable").forEach(th => {
    th.addEventListener("click", () => {
      const col = th.dataset.col;
      if (subState.sort.col === col) subState.sort.desc = !subState.sort.desc;
      else { subState.sort.col = col; subState.sort.desc = false; }
      renderSubmissionTable();
    });
  });

  renderSubmissionTable();
}

function renderSubmissionTable() {
  const f = subState.filters;
  const langOrder = Object.fromEntries(
    state.data.languages.map((l, i) => [l.key, i])
  );
  const langName = Object.fromEntries(
    state.data.languages.map(l => [l.key, l.iso])
  );
  let rows = state.data.test_predictions || [];
  if (f.lang) rows = rows.filter(r => r.language === f.lang);
  if (f.id)   rows = rows.filter(r => (r.id || "").toLowerCase().includes(f.id));
  if (f.pred) rows = rows.filter(r => (r.predicted_caption || "").toLowerCase().includes(f.pred));
  if (f.back) rows = rows.filter(r => (r.back_translation || "").toLowerCase().includes(f.back));

  const sortKey = subState.sort.col;
  const desc = subState.sort.desc ? -1 : 1;
  rows = [...rows];
  rows.sort((a, b) => {
    let av, bv;
    switch (sortKey) {
      case "lang": av = langOrder[a.language] ?? 99; bv = langOrder[b.language] ?? 99; break;
      case "id":   av = a.id; bv = b.id; break;
      case "pred": av = a.predicted_caption || ""; bv = b.predicted_caption || ""; break;
      case "back": av = a.back_translation || ""; bv = b.back_translation || ""; break;
      default:     av = 0; bv = 0;
    }
    if (av < bv) return -1 * desc;
    if (av > bv) return  1 * desc;
    return 0;
  });

  const total = (state.data.test_predictions || []).length;
  document.getElementById("submission-summary").textContent =
    `${rows.length.toLocaleString()} of ${total.toLocaleString()} test predictions` +
    (state.data.submission_label ? ` — ${state.data.submission_label}` : "");

  // Sort indicators
  document.querySelectorAll("#submission-table thead th.sortable").forEach(th => {
    th.classList.remove("sort-asc", "sort-desc");
    th.innerHTML = th.innerHTML.replace(/\s*[▾▲▼]\s*$/, "");
    if (th.dataset.col === subState.sort.col) {
      th.classList.add(subState.sort.desc ? "sort-desc" : "sort-asc");
    } else {
      th.innerHTML += " ▾";
    }
  });

  const tbody = document.querySelector("#submission-table tbody");
  tbody.innerHTML = "";
  if (rows.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" style="text-align:center;color:var(--fg-soft);padding:32px;">No predictions match.</td></tr>`;
    return;
  }
  const MAX_ROWS = 500;
  const truncated = rows.length > MAX_ROWS;
  const view = truncated ? rows.slice(0, MAX_ROWS) : rows;
  for (const r of view) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="thumb-col"><img class="thumb" src="${escape(r.image)}" alt="" loading="lazy" /></td>
      <td>${escape(langName[r.language] || r.language)}</td>
      <td><code>${escape(r.id)}</code></td>
      <td>${markPlaceholders(escape(r.predicted_caption || "—"))}</td>
      <td class="back-col">${r.back_translation ? markPlaceholders(escape(r.back_translation)) : '<span class="muted">—</span>'}</td>
    `;
    tr.addEventListener("click", () => openSubmissionDetail(r));
    tbody.appendChild(tr);
  }
  if (truncated) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="5" style="text-align:center;color:var(--fg-soft);padding:16px;">+${(rows.length - MAX_ROWS).toLocaleString()} more rows match — refine filters.</td>`;
    tbody.appendChild(tr);
  }
}

function openSubmissionDetail(r) {
  const lang = state.data.languages.find(l => l.key === r.language);
  document.getElementById("detail-source").textContent = r.predicted_caption || "(no prediction)";
  document.getElementById("detail-type").textContent =
    `${lang.name} (${lang.iso}) — ${r.id} — ${state.data.submission_label || "test submission"}`;
  document.getElementById("detail-image").src = r.image;
  document.getElementById("detail-image").alt = `${lang.name} test image ${r.id}`;
  const grid = document.querySelector(".detail-grid");
  const rows = [
    ["Image ID",  `<span class="value mono">${escape(r.id)}</span>`],
    ["Predicted (target)", `<span class="value mono">${markPlaceholders(escape(r.predicted_caption || "—"))}</span>`],
  ];
  if (r.back_translation) {
    rows.push(["Back-translation", `<span>${markPlaceholders(escape(r.back_translation))}</span>`]);
  }
  grid.innerHTML = rows
    .map(([k, v]) => `<div class="label">${k}</div><div>${v}</div>`)
    .join("");
  document.querySelector(".english-h4")?.classList.add("hidden");
  document.getElementById("detail-english")?.classList.add("hidden");
  document.getElementById("detail-modal").showModal();
}

document.addEventListener("click", e => {
  if (e.target.matches(".modal-close")) {
    document.getElementById("detail-modal").close();
  }
});
document.getElementById("detail-modal").addEventListener("click", e => {
  if (e.target === e.currentTarget) e.target.close();
});

// ----- helpers -----
function fmt(x) {
  return x == null || Number.isNaN(x) ? "—" : x.toFixed(2);
}

function escape(s) {
  if (s == null) return "";
  return String(s)
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;").replaceAll("'", "&#39;");
}

function markPlaceholders(s) {
  return s.replace(/\[([A-Za-z_][\w-]*)\]/g, '<span class="placeholder">[$1]</span>');
}
