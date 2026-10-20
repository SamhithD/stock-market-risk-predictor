import pandas as pd
import json
import numpy as np
from pathlib import Path

DATA = Path("/Users/samhithd/Downloads/dashboard_data")

# ── Load data ─────────────────────────────────────────────────────────────────
meta        = json.loads((DATA / "meta.json").read_text())
fc          = pd.read_csv(DATA / "latest_stock_forecasts.csv")
corr        = pd.read_csv(DATA / "correlation_250d.csv", index_col=0)
alloc_stats = pd.read_csv(DATA / "allocation_stats.csv")
alloc_wts   = pd.read_csv(DATA / "allocation_weights_latest.csv", index_col=0)
alloc_ret   = pd.read_csv(DATA / "allocation_daily_returns.csv", parse_dates=["Date"])
regime_ts   = pd.read_csv(DATA / "regime_3state_oos.csv", parse_dates=["Date"])
regime_yr   = pd.read_csv(DATA / "regime_share_by_year.csv")
track_vol   = pd.read_csv(DATA / "track_volatility.csv")
track_var   = pd.read_csv(DATA / "track_var.csv")
track_dir   = pd.read_csv(DATA / "track_direction.csv")

# ── Cumulative returns ────────────────────────────────────────────────────────
strats = ["Equal weight", "Min variance", "Min variance (ML)", "HRP", "HRP (ML)"]
cum_ret = alloc_ret.copy()
for s in strats:
    cum_ret[s] = (1 + alloc_ret[s]).cumprod()

# ── Data for JS ───────────────────────────────────────────────────────────────
TICKERS    = [t.replace(".NS", "") for t in fc["ticker"].tolist()]
FULL_NAMES = {
    "HDFCBANK":   "HDFC Bank",
    "HINDUNILVR": "Hindustan Unilever",
    "ICICIBANK":  "ICICI Bank",
    "INFY":       "Infosys",
    "ITC":        "ITC",
    "LT":         "Larsen & Toubro",
    "MARUTI":     "Maruti Suzuki",
    "RELIANCE":   "Reliance Industries",
    "SUNPHARMA":  "Sun Pharma",
    "TCS":        "TCS",
}
CLOSES      = fc["close"].tolist()
VOL_ANN     = fc["vol20_forecast"].tolist()
VAR5_ML     = fc["var5_ml"].tolist()
P_UP        = fc["p_up_5d"].tolist()
CORR_MAT    = corr.values.tolist()
HRP_WTS     = alloc_wts["HRP"].tolist()
MINV_WTS    = alloc_wts["Min variance"].tolist()
EW_WTS      = alloc_wts["Equal weight"].tolist()
STRESS_NOW  = meta["regime_p_stress_now"]

# ── Plotly reference charts ───────────────────────────────────────────────────
import plotly.graph_objects as go

BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", color="#374151", size=11),
    margin=dict(l=44, r=12, t=36, b=36),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
)
AX = dict(gridcolor="#e5e7eb", linecolor="#d1d5db", tickcolor="#d1d5db", zerolinecolor="#e5e7eb")

SCOLS = {
    "Equal weight":      "#94a3b8",
    "Min variance":      "#3b82f6",
    "Min variance (ML)": "#1d4ed8",
    "HRP":               "#22c55e",
    "HRP (ML)":          "#15803d",
}

def j(fig): return fig.to_json()

# Cumulative returns
fig_cum = go.Figure()
for s in strats:
    fig_cum.add_trace(go.Scatter(x=cum_ret["Date"], y=cum_ret[s], mode="lines",
        name=s, line=dict(color=SCOLS[s], width=2)))
fig_cum.update_layout(**BASE, height=280,
    title=dict(text="Strategy Cumulative Returns — Walk-Forward OOS (2016–2026)", font_size=12, x=0),
    yaxis=dict(**AX, title="₹1 start"), xaxis=dict(**AX), hovermode="x unified")
CUM_J = j(fig_cum)

# Stress time series
fig_st = go.Figure()
fig_st.add_trace(go.Scatter(x=regime_ts["Date"], y=regime_ts["p_stress"],
    mode="lines", fill="tozeroy", fillcolor="rgba(239,68,68,0.08)",
    line=dict(color="#ef4444", width=1.2), showlegend=False))
fig_st.add_hline(y=0.5, line_dash="dot", line_color="#9ca3af",
    annotation_text="0.5", annotation_position="bottom right", annotation_font_size=9)
fig_st.update_layout(**BASE, height=200,
    title=dict(text="Market Stress Probability — 3-State HMM (OOS)", font_size=12, x=0),
    yaxis=dict(**AX, range=[0,1], title="p(stress)"), xaxis=dict(**AX))
ST_J = j(fig_st)

# Regime by year
yr_colors = ["#ef4444" if v>0.35 else "#f59e0b" if v>0.15 else "#22c55e"
             for v in regime_yr["share_days_flagged"]]
fig_yr = go.Figure(go.Bar(x=regime_yr["Date"].astype(str), y=regime_yr["share_days_flagged"],
    marker_color=yr_colors, showlegend=False))
fig_yr.update_layout(**BASE, height=210,
    title=dict(text="Stress Days by Year", font_size=12, x=0),
    yaxis=dict(**AX, tickformat=".0%"), xaxis=dict(**AX))
YR_J = j(fig_yr)

# Correlation heatmap
labels = TICKERS
fig_corr = go.Figure(go.Heatmap(
    z=CORR_MAT, x=labels, y=labels,
    colorscale=[[0,"#f0f9ff"],[0.4,"#93c5fd"],[1,"#1e40af"]],
    zmin=0, zmax=1,
    text=np.round(CORR_MAT, 2), texttemplate="%{text}", textfont_size=9,
    showscale=True, colorbar=dict(thickness=8, len=0.8, tickfont_size=9),
))
fig_corr.update_layout(**{**BASE, "margin": dict(l=56, r=12, t=36, b=52)},
    height=320, title=dict(text="250-Day Pairwise Correlation", font_size=12, x=0),
    xaxis=dict(tickfont_size=9), yaxis=dict(tickfont_size=9, autorange="reversed"))
CORR_J = j(fig_corr)

# Alloc perf table
def perf_rows(period_key):
    rows = ""
    df = alloc_stats[alloc_stats["period"] == period_key]
    best_s = df["Sharpe"].max()
    for _, r in df.iterrows():
        bold = r["Sharpe"] == best_s
        rows += (f'<tr><td class="fw5">{r["strategy"]}</td>'
                 f'<td class="r">{r["Ann return"]*100:.1f}%</td>'
                 f'<td class="r">{r["Ann vol"]*100:.1f}%</td>'
                 f'<td class="r" style="color:{"#16a34a" if bold else "#374151"};font-weight:{"600" if bold else "400"}">{r["Sharpe"]:.3f}</td>'
                 f'<td class="r" style="color:#dc2626">{r["Max drawdown"]*100:.1f}%</td>'
                 f'<td class="r muted">{r["Turnover/yr"]:.1f}×</td></tr>')
    return rows

# Vol model table
def vol_rows():
    rows = ""
    best = track_vol[track_vol["period"]=="all"]["MAE"].min()
    for _, r in track_vol.iterrows():
        hi = r["period"]=="all" and r["MAE"]==best
        rows += (f'<tr><td class="fw5">{r["model"]}</td><td class="muted">{r["period"]}</td>'
                 f'<td class="r" style="color:{"#16a34a" if hi else ""}">{r["MAE"]:.4f}</td>'
                 f'<td class="r">{r["RMSE"]:.4f}</td><td class="r">{r["QLIKE"]:.4f}</td></tr>')
    return rows

# VaR table
def var_rows():
    rows = ""
    for _, r in track_var.iterrows():
        ok = abs(r["breach_all"]*100 - 5) < 0.9
        rows += (f'<tr><td class="fw5">{r["model"]}</td>'
                 f'<td class="r" style="color:{"#16a34a" if ok else "#dc2626"}">{r["breach_all"]:.3%}</td>'
                 f'<td class="r muted">{r["breach_ex2020"]:.3%}</td>'
                 f'<td class="r muted">{r["breach_2020"]:.3%}</td>'
                 f'<td class="r muted">{r["breach_2022"]:.3%}</td>'
                 f'<td class="r">{r["pinball_x1e4"]:.2f}</td></tr>')
    return rows

# Direction table
def dir_rows():
    rows = ""
    for _, r in track_dir.iterrows():
        gray = r["model"] == "Always up"
        rows += (f'<tr><td class="fw5{"  muted" if gray else ""}">{r["model"]}</td>'
                 f'<td class="muted">{r["period"]}</td>'
                 f'<td class="r">{r["Accuracy"]:.4f}</td><td class="r">{r["AUC"]:.4f}</td></tr>')
    return rows

# ── Embedded JS data ─────────────────────────────────────────────────────────
TICKERS_JS   = json.dumps(TICKERS)
FULL_NAMES_JS= json.dumps([FULL_NAMES[t] for t in TICKERS])
CLOSES_JS    = json.dumps([round(c, 2) for c in CLOSES])
VOL_JS       = json.dumps([round(v, 6) for v in VOL_ANN])
VAR_JS       = json.dumps([round(v, 6) for v in VAR5_ML])
PUP_JS       = json.dumps([round(p, 4) for p in P_UP])
CORR_JS      = json.dumps([[round(v, 6) for v in row] for row in CORR_MAT])
HRP_JS       = json.dumps(HRP_WTS)
MINV_JS      = json.dumps(MINV_WTS)
EW_JS        = json.dumps(EW_WTS)
STRESS_JS    = json.dumps(round(STRESS_NOW, 4))

# Stress state for Python-side rendering
STRESS_LABEL  = "High Stress" if STRESS_NOW > 0.5 else "Low Stress"
STRESS_CLASS  = "stress" if STRESS_NOW > 0.5 else "calm"

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nifty ML — Portfolio Risk</title>
<script src="{PLOTLY_CDN}"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#f1f5f9;--surf:#fff;--border:#e2e8f0;
  --text:#1e293b;--muted:#64748b;
  --blue:#2563eb;--green:#16a34a;--red:#dc2626;--amber:#d97706;
  --r:10px;
}}
body{{font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--text);font-size:13px;line-height:1.5;-webkit-font-smoothing:antialiased}}

/* ── top bar ── */
.topbar{{background:var(--surf);border-bottom:1px solid var(--border);padding:0 28px;height:52px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100;box-shadow:0 1px 2px rgba(0,0,0,.05)}}
.brand{{font-size:14px;font-weight:700;letter-spacing:-.3px}}.brand em{{color:var(--blue);font-style:normal}}
.as-of{{font-size:11px;color:var(--muted);background:#f8fafc;border:1px solid var(--border);border-radius:5px;padding:3px 9px}}
.regime-chip{{display:flex;align-items:center;gap:6px;border-radius:20px;padding:4px 12px;font-size:11px;font-weight:600}}
.regime-chip.calm{{background:#f0fdf4;color:#15803d;border:1px solid #bbf7d0}}
.regime-chip.stress{{background:#fef2f2;color:#dc2626;border:1px solid #fecaca}}
.rdot{{width:6px;height:6px;border-radius:50%;background:currentColor}}

/* ── nav ── */
.subnav{{background:var(--surf);border-bottom:1px solid var(--border);padding:0 28px;display:flex;overflow-x:auto}}
.nl{{padding:9px 15px;font-size:12px;font-weight:500;color:var(--muted);text-decoration:none;border-bottom:2px solid transparent;white-space:nowrap;cursor:pointer}}
.nl:hover{{color:var(--text)}}.nl.on{{color:var(--blue);border-bottom-color:var(--blue)}}

/* ── page ── */
.page{{max-width:1360px;margin:0 auto;padding:22px 20px 60px}}
.sec-label{{font-size:10px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);padding-bottom:10px;border-bottom:1px solid var(--border);margin:28px 0 14px}}

/* ── stock input grid ── */
.stock-grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}}
.stock-card{{background:var(--surf);border:1px solid var(--border);border-radius:var(--r);padding:14px 14px 12px;position:relative;transition:border-color .15s}}
.stock-card:focus-within{{border-color:var(--blue);box-shadow:0 0 0 3px rgba(37,99,235,.1)}}
.sc-ticker{{font-size:14px;font-weight:700;letter-spacing:-.2px}}
.sc-name{{font-size:10px;color:var(--muted);margin-top:1px;margin-bottom:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.sc-price{{font-size:11px;color:var(--muted);margin-bottom:8px}}
.sc-price strong{{color:var(--text);font-weight:600}}
.sc-input-row{{display:flex;align-items:center;gap:6px}}
.sc-input{{
  width:100%;padding:6px 8px;border:1px solid var(--border);border-radius:6px;
  font-size:13px;font-family:inherit;color:var(--text);background:#fafafa;
  -moz-appearance:textfield;
}}
.sc-input:focus{{outline:none;border-color:var(--blue);background:#fff}}
.sc-input::-webkit-inner-spin-button,.sc-input::-webkit-outer-spin-button{{-webkit-appearance:none}}
.sc-unit{{font-size:11px;color:var(--muted);white-space:nowrap}}
.sc-val{{font-size:11px;color:var(--muted);margin-top:5px;min-height:14px}}
.sc-val strong{{color:var(--text)}}

/* ── input mode toggle ── */
.mode-row{{display:flex;align-items:center;gap:10px;margin-bottom:16px}}
.mode-toggle{{display:flex;background:#f1f5f9;border:1px solid var(--border);border-radius:7px;padding:2px}}
.mt-btn{{padding:4px 14px;font-size:11px;font-weight:500;border:none;cursor:pointer;border-radius:5px;background:transparent;color:var(--muted);transition:all .12s}}
.mt-btn.on{{background:var(--surf);color:var(--blue);font-weight:600;box-shadow:0 1px 2px rgba(0,0,0,.08)}}
.preset-row{{display:flex;gap:6px;flex-wrap:wrap}}
.preset-btn{{font-size:11px;font-weight:500;padding:4px 10px;border:1px solid var(--border);border-radius:6px;cursor:pointer;background:var(--surf);color:var(--muted);transition:all .12s}}
.preset-btn:hover{{border-color:var(--blue);color:var(--blue)}}

/* ── total bar ── */
.total-bar{{background:var(--surf);border:1px solid var(--border);border-radius:var(--r);padding:14px 18px;display:flex;align-items:center;justify-content:space-between;margin-top:10px}}
.total-label{{font-size:12px;color:var(--muted)}}
.total-val{{font-size:20px;font-weight:700;letter-spacing:-.5px;color:var(--text)}}
.calc-btn{{
  background:var(--blue);color:#fff;font-size:13px;font-weight:600;
  padding:10px 28px;border:none;border-radius:8px;cursor:pointer;
  transition:background .15s;letter-spacing:.01em;
}}
.calc-btn:hover{{background:#1d4ed8}}
.calc-btn:active{{background:#1e40af}}
.clear-btn{{background:transparent;color:var(--muted);font-size:12px;border:1px solid var(--border);border-radius:8px;padding:10px 16px;cursor:pointer}}
.clear-btn:hover{{border-color:#94a3b8;color:var(--text)}}

/* ── results ── */
#results{{display:none}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:12px}}
.kpi{{background:var(--surf);border:1px solid var(--border);border-radius:var(--r);padding:16px 18px}}
.kpi-lbl{{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin-bottom:5px}}
.kpi-val{{font-size:22px;font-weight:700;letter-spacing:-.5px;line-height:1.1}}
.kpi-sub{{font-size:11px;color:var(--muted);margin-top:4px}}
.c-blue{{color:var(--blue)}}.c-green{{color:var(--green)}}.c-red{{color:var(--red)}}.c-amber{{color:var(--amber)}}

/* ── risk bar strip ── */
.risk-strip{{background:var(--surf);border:1px solid var(--border);border-radius:var(--r);padding:16px 18px;margin-bottom:12px}}
.rs-header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}}
.rs-title{{font-size:12px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}}
.rs-legend{{display:flex;gap:14px}}
.rs-leg-item{{display:flex;align-items:center;gap:5px;font-size:11px;color:var(--muted)}}
.rs-dot{{width:8px;height:8px;border-radius:2px}}
.risk-bar-row{{display:flex;align-items:center;gap:10px;margin-bottom:6px}}
.rb-ticker{{width:80px;font-weight:600;font-size:12px;flex-shrink:0}}
.rb-track{{flex:1;height:18px;background:#f1f5f9;border-radius:4px;overflow:hidden;position:relative}}
.rb-fill{{height:100%;border-radius:4px;transition:width .4s ease;display:flex;align-items:center;justify-content:flex-end;padding-right:6px}}
.rb-fill span{{font-size:10px;font-weight:600;color:#fff;white-space:nowrap}}
.rb-pct{{width:44px;text-align:right;font-size:11px;color:var(--muted)}}

/* ── weight comparison ── */
.grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.card{{background:var(--surf);border:1px solid var(--border);border-radius:var(--r);padding:16px;overflow:hidden}}
.card-title{{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin-bottom:12px}}

/* ── tables ── */
.tbl{{width:100%;border-collapse:collapse}}
.tbl th{{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);text-align:left;padding:5px 10px 7px;border-bottom:1px solid var(--border)}}
.tbl td{{padding:7px 10px;border-bottom:1px solid #f1f5f9;font-size:12px}}
.tbl tr:last-child td{{border-bottom:none}}
.tbl tr:hover td{{background:#f8fafc}}
.r{{text-align:right}}.fw5{{font-weight:500}}.muted{{color:var(--muted)}}

/* ── tabs ── */
.tabs{{display:flex;gap:2px;margin-bottom:12px}}
.tab{{padding:4px 12px;font-size:11px;font-weight:500;border:1px solid transparent;cursor:pointer;border-radius:6px;background:transparent;color:var(--muted)}}
.tab.on{{background:#eff6ff;color:var(--blue);border-color:#bfdbfe;font-weight:600}}
.pane{{display:none}}.pane.on{{display:block}}

/* ── callout ── */
.callout{{background:#f8fafc;border-left:3px solid var(--blue);border-radius:0 6px 6px 0;padding:10px 14px;font-size:12px;color:var(--muted);line-height:1.6;margin-bottom:14px}}
.callout strong{{color:var(--text)}}
.warn{{border-left-color:var(--amber);background:#fffbeb}}

/* ── disclaimer ── */
.disclaimer{{background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:12px 16px;font-size:11px;color:#92400e;line-height:1.6;margin-top:28px}}

/* ── no holdings state ── */
.empty-state{{text-align:center;padding:40px;color:var(--muted)}}
.empty-state h3{{font-size:14px;font-weight:600;color:var(--text);margin-bottom:6px}}

@media(max-width:1000px){{.stock-grid{{grid-template-columns:repeat(3,1fr)}}}}
@media(max-width:700px){{.stock-grid{{grid-template-columns:repeat(2,1fr)}}.kpis{{grid-template-columns:repeat(2,1fr)}}.grid-2{{grid-template-columns:1fr}}}}
</style>
</head>
<body>

<!-- ── Top bar ── -->
<header class="topbar">
  <div style="display:flex;align-items:center;gap:16px">
    <span class="brand">Nifty <em>ML</em> Portfolio Risk</span>
    <span class="as-of">As of {meta["as_of"]}</span>
  </div>
  <div class="regime-chip {STRESS_CLASS}">
    <span class="rdot"></span>
    {STRESS_LABEL} &nbsp;·&nbsp; p(stress) = {STRESS_NOW:.1%}
  </div>
</header>

<!-- ── Sub-nav ── -->
<nav class="subnav">
  <span class="nl on" onclick="scrollTo('calc')">Portfolio Risk</span>
  <span class="nl" onclick="scrollTo('strategies')">Strategies</span>
  <span class="nl" onclick="scrollTo('regime')">Regime</span>
  <span class="nl" onclick="scrollTo('models')">Model Track Record</span>
</nav>

<main class="page">

<!-- ══════════════════════════════════════════════════════════════════════
     PORTFOLIO RISK CALCULATOR
══════════════════════════════════════════════════════════════════════ -->
<div class="sec-label" id="calc">Portfolio Risk Calculator</div>

<div class="callout">
  Enter your holdings below. Risk is estimated using walk-forward ML volatility forecasts and the
  250-day correlation matrix. <strong>VaR uses a normal approximation</strong> —
  losses cluster in practice, so actual tail losses can exceed the estimate.
</div>

<!-- mode + presets -->
<div class="mode-row">
  <div class="mode-toggle">
    <button class="mt-btn on" id="modeSharesBtn" onclick="setMode('shares')">Shares</button>
    <button class="mt-btn" id="modeValueBtn" onclick="setMode('value')">₹ Value</button>
  </div>
  <div class="preset-row">
    <button class="preset-btn" onclick="applyPreset('ew',100000)">Equal weight ₹1L</button>
    <button class="preset-btn" onclick="applyPreset('hrp',100000)">HRP ₹1L</button>
    <button class="preset-btn" onclick="applyPreset('minv',100000)">Min-Var ₹1L</button>
    <button class="preset-btn" onclick="clearAll()">Clear</button>
  </div>
</div>

<!-- stock grid -->
<div class="stock-grid" id="stockGrid"></div>

<!-- total + buttons -->
<div class="total-bar">
  <div>
    <div class="total-label">Total portfolio value</div>
    <div class="total-val" id="totalVal">₹0</div>
  </div>
  <div style="display:flex;gap:8px;align-items:center">
    <button class="clear-btn" onclick="clearAll()">Clear</button>
    <button class="calc-btn" onclick="calculate()">Calculate Risk →</button>
  </div>
</div>

<!-- ══════════════════════════════════════════════════════════════════════
     RESULTS (hidden until calculated)
══════════════════════════════════════════════════════════════════════ -->
<div id="results">
  <div class="sec-label" style="margin-top:20px">Risk Estimate</div>

  <!-- KPI row -->
  <div class="kpis">
    <div class="kpi">
      <div class="kpi-lbl">1-Day VaR (95%)</div>
      <div class="kpi-val c-red" id="kpi-var95">—</div>
      <div class="kpi-sub" id="kpi-var95-pct">normal approx, diversified</div>
    </div>
    <div class="kpi">
      <div class="kpi-lbl">1-Day VaR (99%)</div>
      <div class="kpi-val c-red" id="kpi-var99">—</div>
      <div class="kpi-sub" id="kpi-var99-pct"></div>
    </div>
    <div class="kpi">
      <div class="kpi-lbl">Portfolio Vol (ann.)</div>
      <div class="kpi-val c-blue" id="kpi-vol">—</div>
      <div class="kpi-sub" id="kpi-div">—</div>
    </div>
    <div class="kpi">
      <div class="kpi-lbl">Market Regime</div>
      <div class="kpi-val {STRESS_CLASS.replace('calm','c-green').replace('stress','c-red')}" id="kpi-regime">{STRESS_LABEL}</div>
      <div class="kpi-sub">p(stress) = {STRESS_NOW:.1%}</div>
    </div>
  </div>

  <!-- Risk contribution bars -->
  <div class="risk-strip">
    <div class="rs-header">
      <span class="rs-title">Risk Contribution by Stock (% of portfolio VaR)</span>
      <div class="rs-legend">
        <div class="rs-leg-item"><div class="rs-dot" style="background:#2563eb"></div>Component VaR share</div>
        <div class="rs-leg-item"><div class="rs-dot" style="background:#e2e8f0"></div>Weight share</div>
      </div>
    </div>
    <div id="riskBars"></div>
  </div>

  <!-- Weight comparison + per-stock table -->
  <div class="grid-2">
    <div class="card">
      <div class="card-title">Your Weights vs. Reference Strategies</div>
      <div id="wtChart"></div>
    </div>
    <div class="card">
      <div class="card-title">Per-Stock Risk Detail</div>
      <table class="tbl">
        <thead><tr>
          <th>Stock</th>
          <th class="r">Weight</th>
          <th class="r">Position (₹)</th>
          <th class="r">1d VaR (₹)</th>
          <th class="r">Risk Contrib.</th>
        </tr></thead>
        <tbody id="stockRiskTable"></tbody>
      </table>
    </div>
  </div>

  <!-- Undiversified vs diversified -->
  <div class="callout" style="margin-top:12px" id="divBenefit"></div>
</div>

<!-- ══════════════════════════════════════════════════════════════════════
     STRATEGIES (reference)
══════════════════════════════════════════════════════════════════════ -->
<div class="sec-label" id="strategies">Strategy Performance — Reference</div>

<div class="grid-2">
  <div class="card">
    <div class="tabs">
      <button class="tab on" onclick="tabSwitch(this,'pall')">Full OOS</button>
      <button class="tab" onclick="tabSwitch(this,'p2020')">Since 2020</button>
      <button class="tab" onclick="tabSwitch(this,'p2022')">Since 2022</button>
    </div>
    <div id="pall" class="pane on">
      <table class="tbl"><thead><tr>
        <th>Strategy</th><th class="r">Ann Ret</th><th class="r">Ann Vol</th>
        <th class="r">Sharpe</th><th class="r">Max DD</th><th class="r">Turnover</th>
      </tr></thead><tbody>{perf_rows('all')}</tbody></table>
    </div>
    <div id="p2020" class="pane">
      <table class="tbl"><thead><tr>
        <th>Strategy</th><th class="r">Ann Ret</th><th class="r">Ann Vol</th>
        <th class="r">Sharpe</th><th class="r">Max DD</th><th class="r">Turnover</th>
      </tr></thead><tbody>{perf_rows('2020')}</tbody></table>
    </div>
    <div id="p2022" class="pane">
      <table class="tbl"><thead><tr>
        <th>Strategy</th><th class="r">Ann Ret</th><th class="r">Ann Vol</th>
        <th class="r">Sharpe</th><th class="r">Max DD</th><th class="r">Turnover</th>
      </tr></thead><tbody>{perf_rows('2022')}</tbody></table>
    </div>
  </div>
  <div class="card" style="padding-bottom:8px"><div id="cumChart"></div></div>
</div>

<!-- ══════════════════════════════════════════════════════════════════════
     REGIME
══════════════════════════════════════════════════════════════════════ -->
<div class="sec-label" id="regime">Regime</div>

<div class="callout warn">
  <strong>Regime model adds limited edge.</strong> The 3-state HMM identifies stress periods but
  walk-forward tests show no consistent improvement over holding constant exposure. March 2020 was
  the one notable exception. Use as context only.
</div>

<div class="grid-2">
  <div class="card" style="padding-bottom:8px"><div id="yrChart"></div></div>
  <div class="card" style="padding-bottom:8px"><div id="stressChart"></div></div>
</div>
<div class="card" style="margin-top:12px;padding-bottom:8px"><div id="corrChart"></div></div>

<!-- ══════════════════════════════════════════════════════════════════════
     MODEL TRACK RECORD
══════════════════════════════════════════════════════════════════════ -->
<div class="sec-label" id="models">Model Track Record</div>

<div class="grid-2">
  <div class="card">
    <div class="card-title">Volatility Forecast — MAE / RMSE / QLIKE</div>
    <table class="tbl"><thead><tr>
      <th>Model</th><th>Period</th><th class="r">MAE</th><th class="r">RMSE</th><th class="r">QLIKE</th>
    </tr></thead><tbody>{vol_rows()}</tbody></table>
  </div>
  <div class="card">
    <div class="card-title">VaR Breach Rates (target 5%)</div>
    <table class="tbl"><thead><tr>
      <th>Model</th><th class="r">All</th><th class="r">ex-2020</th>
      <th class="r">2020</th><th class="r">2022</th><th class="r">Pinball ×10⁴</th>
    </tr></thead><tbody>{var_rows()}</tbody></table>
  </div>
</div>

<div class="card" style="margin-top:12px">
  <div class="card-title">Direction Model — Accuracy &amp; AUC</div>
  <div class="callout" style="margin-bottom:0">
    <strong>No demonstrated edge.</strong> Accuracy (~52%) does not exceed the always-up baseline (52.9%).
    p(Up 5d) shown in the calculator above is for context only — not a trading signal.
  </div>
  <table class="tbl" style="margin-top:12px"><thead><tr>
    <th>Model</th><th>Period</th><th class="r">Accuracy</th><th class="r">AUC</th>
  </tr></thead><tbody>{dir_rows()}</tbody></table>
</div>

<div class="disclaimer">
  ⚠ <strong>Research tool only.</strong> {meta["note"]} —
  Data: Yahoo Finance, 10 Nifty 50 stocks, {meta["test_years"]} (walk-forward OOS, 20-day gap).
  Survivorship bias present. Normal VaR underestimates tail risk during crises.
  No BUY or SELL signals.
</div>

</main>

<!-- ══════════════════════════════════════════════════════════════════════
     JAVASCRIPT
══════════════════════════════════════════════════════════════════════ -->
<script>
// ── Embedded data ──────────────────────────────────────────────────────────
const TICKERS    = {TICKERS_JS};
const NAMES      = {FULL_NAMES_JS};
const CLOSES     = {CLOSES_JS};
const VOL_ANN    = {VOL_JS};
const VAR5_ML    = {VAR_JS};
const P_UP       = {PUP_JS};
const CORR       = {CORR_JS};
const HRP_WTS    = {HRP_JS};
const MINV_WTS   = {MINV_JS};
const EW_WTS     = {EW_JS};
const STRESS     = {STRESS_JS};

const N = TICKERS.length;
const STOCK_COLORS = [
  '#2563eb','#7c3aed','#0891b2','#059669','#65a30d',
  '#d97706','#dc2626','#9333ea','#0d9488','#ea580c'
];

// ── State ──────────────────────────────────────────────────────────────────
let inputMode = 'shares'; // 'shares' | 'value'

// ── Build stock input cards ────────────────────────────────────────────────
function buildGrid() {{
  const grid = document.getElementById('stockGrid');
  grid.innerHTML = '';
  TICKERS.forEach((tk, i) => {{
    const card = document.createElement('div');
    card.className = 'stock-card';
    card.innerHTML = `
      <div class="sc-ticker" style="color:${{STOCK_COLORS[i]}}">${{tk}}</div>
      <div class="sc-name">${{NAMES[i]}}</div>
      <div class="sc-price">₹<strong>${{fmt(CLOSES[i])}}</strong> per share</div>
      <div class="sc-input-row">
        <input class="sc-input" id="inp_${{i}}" type="number" min="0" step="1"
          placeholder="0" oninput="onInput(${{i}})" />
        <span class="sc-unit" id="unit_${{i}}">shares</span>
      </div>
      <div class="sc-val" id="val_${{i}}"></div>
    `;
    grid.appendChild(card);
  }});
}}

// ── Input helpers ──────────────────────────────────────────────────────────
function getPositionValues() {{
  return TICKERS.map((_, i) => {{
    const raw = parseFloat(document.getElementById('inp_' + i).value) || 0;
    if (inputMode === 'shares') return raw * CLOSES[i];
    return raw; // already ₹ value
  }});
}}

function onInput(i) {{
  const raw = parseFloat(document.getElementById('inp_' + i).value) || 0;
  const valEl = document.getElementById('val_' + i);
  if (raw === 0) {{ valEl.textContent = ''; return; }}
  if (inputMode === 'shares') {{
    valEl.innerHTML = `= <strong>₹${{fmt(raw * CLOSES[i])}}</strong>`;
  }} else {{
    const shares = raw / CLOSES[i];
    valEl.innerHTML = `≈ <strong>${{shares.toFixed(2)}} shares</strong>`;
  }}
  updateTotal();
}}

function updateTotal() {{
  const total = getPositionValues().reduce((a, b) => a + b, 0);
  document.getElementById('totalVal').textContent = '₹' + fmt(total);
}}

function setMode(mode) {{
  // convert existing inputs
  const vals = getPositionValues();
  inputMode = mode;
  document.getElementById('modeSharesBtn').classList.toggle('on', mode === 'shares');
  document.getElementById('modeValueBtn').classList.toggle('on', mode === 'value');
  TICKERS.forEach((_, i) => {{
    const inp = document.getElementById('inp_' + i);
    const unit = document.getElementById('unit_' + i);
    unit.textContent = mode === 'shares' ? 'shares' : '₹';
    if (vals[i] > 0) {{
      if (mode === 'shares') {{
        inp.value = (vals[i] / CLOSES[i]).toFixed(0);
      }} else {{
        inp.value = vals[i].toFixed(0);
      }}
    }} else {{
      inp.value = '';
    }}
    onInput(i);
  }});
}}

function applyPreset(type, total) {{
  let wts;
  if (type === 'ew')   wts = EW_WTS;
  if (type === 'hrp')  wts = HRP_WTS;
  if (type === 'minv') wts = MINV_WTS;

  TICKERS.forEach((_, i) => {{
    const posVal = total * wts[i];
    const inp = document.getElementById('inp_' + i);
    if (inputMode === 'shares') {{
      inp.value = Math.round(posVal / CLOSES[i]);
    }} else {{
      inp.value = Math.round(posVal);
    }}
    onInput(i);
  }});
  updateTotal();
}}

function clearAll() {{
  TICKERS.forEach((_, i) => {{
    document.getElementById('inp_' + i).value = '';
    document.getElementById('val_' + i).textContent = '';
  }});
  document.getElementById('totalVal').textContent = '₹0';
  document.getElementById('results').style.display = 'none';
}}

// ── Risk calculation ───────────────────────────────────────────────────────
function calculate() {{
  const posVals = getPositionValues();
  const total   = posVals.reduce((a, b) => a + b, 0);
  if (total <= 0) {{ alert('Enter at least one holding.'); return; }}

  const w = posVals.map(v => v / total);

  // Daily vols
  const dailyVol = VOL_ANN.map(v => v / Math.sqrt(252));

  // Covariance matrix: Σ[i][j] = corr[i][j] * σ_i * σ_j
  const cov = CORR.map((row, i) => row.map((c, j) => c * dailyVol[i] * dailyVol[j]));

  // Portfolio daily variance = w' Σ w
  let portVar = 0;
  for (let i = 0; i < N; i++) {{
    for (let j = 0; j < N; j++) {{
      portVar += w[i] * w[j] * cov[i][j];
    }}
  }}
  const portDailyVol = Math.sqrt(portVar);
  const portAnnVol   = portDailyVol * Math.sqrt(252);

  // VaR (normal)
  const var95 = total * portDailyVol * 1.6449;
  const var99 = total * portDailyVol * 2.3263;

  // Undiversified VaR: weighted sum of individual stock VaR (in ₹)
  const undivVar95 = posVals.reduce((s, pv, i) => s + Math.abs(VAR5_ML[i]) * pv, 0);

  // Component VaR: w[i] * (Σ w)[i] / portDailyVol * z95 * total
  const sigmaW = w.map((_, i) => {{
    let sv = 0;
    for (let j = 0; j < N; j++) sv += cov[i][j] * w[j];
    return sv;
  }});
  const compVar95 = sigmaW.map((sv, i) =>
    (w[i] * sv / portDailyVol) * 1.6449 * total
  );
  const compPct = compVar95.map(cv => cv / var95);

  // Diversification benefit
  const divBenefit = undivVar95 - var95;
  const divRatio   = portAnnVol / VOL_ANN.reduce((s, v, i) => s + w[i] * v, 0);

  // ── Render KPIs ──
  const pct = v => (v * 100).toFixed(1) + '%';
  document.getElementById('kpi-var95').textContent = '₹' + fmt(var95);
  document.getElementById('kpi-var95-pct').textContent = pct(var95 / total) + ' of portfolio · 1-in-20 day';
  document.getElementById('kpi-var99').textContent = '₹' + fmt(var99);
  document.getElementById('kpi-var99-pct').textContent = pct(var99 / total) + ' of portfolio · 1-in-100 day';
  document.getElementById('kpi-vol').textContent = pct(portAnnVol);
  document.getElementById('kpi-div').textContent = 'Diversification ratio: ' + (divRatio * 100).toFixed(0) + '% of weighted avg vol';

  // ── Risk contribution bars ──
  const barsEl = document.getElementById('riskBars');
  barsEl.innerHTML = '';

  // Sort by component VaR desc
  const order = compPct.map((_, i) => i).sort((a, b) => compPct[b] - compPct[a]);
  order.forEach(i => {{
    if (w[i] === 0) return;
    const cp = compPct[i];
    const wp = w[i];
    const row = document.createElement('div');
    row.className = 'risk-bar-row';
    const barW = Math.min(cp * 100, 100).toFixed(1);
    row.innerHTML = `
      <div class="rb-ticker" style="color:${{STOCK_COLORS[i]}}">${{TICKERS[i]}}</div>
      <div class="rb-track">
        <!-- weight reference line -->
        <div style="position:absolute;top:0;bottom:0;left:0;width:${{(wp*100).toFixed(1)}}%;background:#e2e8f0;border-radius:4px"></div>
        <!-- component VaR fill -->
        <div class="rb-fill" style="width:${{barW}}%;background:${{STOCK_COLORS[i]}};position:relative;z-index:1">
          <span>${{(cp*100).toFixed(1)}}%</span>
        </div>
      </div>
      <div class="rb-pct" style="color:${{STOCK_COLORS[i]}}">${{(cp*100).toFixed(1)}}%</div>
    `;
    barsEl.appendChild(row);
  }});

  // ── Per-stock table ──
  const tbody = document.getElementById('stockRiskTable');
  tbody.innerHTML = '';
  order.forEach(i => {{
    if (w[i] === 0) return;
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="fw5" style="color:${{STOCK_COLORS[i]}}">${{TICKERS[i]}}</td>
      <td class="r">${{(w[i]*100).toFixed(1)}}%</td>
      <td class="r">₹${{fmt(posVals[i])}}</td>
      <td class="r" style="color:#dc2626">₹${{fmt(Math.abs(VAR5_ML[i]) * posVals[i])}}</td>
      <td class="r">${{(compPct[i]*100).toFixed(1)}}%</td>
    `;
    tbody.appendChild(tr);
  }});

  // ── Weight comparison chart ──
  const wtData = [
    {{ name: 'Your portfolio', y: w.map(x => x*100), marker: {{ color: '#2563eb' }} }},
    {{ name: 'HRP',            y: HRP_WTS.map(x => x*100), marker: {{ color: '#22c55e' }} }},
    {{ name: 'Min Variance',   y: MINV_WTS.map(x => x*100), marker: {{ color: '#94a3b8' }} }},
  ];
  Plotly.react('wtChart',
    wtData.map(d => ({{
      type: 'bar', name: d.name,
      x: TICKERS, y: d.y,
      marker: d.marker,
    }})),
    {{
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: {{family: 'Inter, sans-serif', size: 10, color: '#374151'}},
      barmode: 'group', height: 220,
      margin: {{l: 36, r: 8, t: 8, b: 40}},
      legend: {{bgcolor: 'rgba(0,0,0,0)', borderwidth: 0, font: {{size: 10}}}},
      yaxis: {{gridcolor: '#e5e7eb', title: 'Weight (%)', ticksuffix: '%'}},
      xaxis: {{gridcolor: '#e5e7eb', tickfont: {{size: 10}}}},
    }},
    {{responsive: true, displayModeBar: false}}
  );

  // ── Diversification callout ──
  document.getElementById('divBenefit').innerHTML =
    `<strong>Diversification benefit:</strong> Your undiversified (sum-of-individual) VaR is
     <strong>₹${{fmt(undivVar95)}}</strong>. Correlation reduces this to
     <strong>₹${{fmt(var95)}}</strong> — a saving of <strong>₹${{fmt(divBenefit)}}</strong>
     (${{(divBenefit/undivVar95*100).toFixed(1)}}%).
     Diversification ratio: your portfolio vol is <strong>${{(divRatio*100).toFixed(0)}}%</strong>
     of the weighted-average individual vol.`;

  document.getElementById('results').style.display = 'block';
  document.getElementById('results').scrollIntoView({{behavior: 'smooth', block: 'start'}});
}}

// ── Reference charts ──────────────────────────────────────────────────────
const cfg = {{responsive: true, displayModeBar: false}};
Plotly.newPlot('cumChart',    {CUM_J},   cfg);
Plotly.newPlot('stressChart', {ST_J},    cfg);
Plotly.newPlot('yrChart',     {YR_J},    cfg);
Plotly.newPlot('corrChart',   {CORR_J},  cfg);

// ── Tabs ──────────────────────────────────────────────────────────────────
function tabSwitch(btn, id) {{
  const card = btn.closest('.card');
  card.querySelectorAll('.tab').forEach(b => b.classList.remove('on'));
  card.querySelectorAll('.pane').forEach(p => p.classList.remove('on'));
  btn.classList.add('on');
  document.getElementById(id).classList.add('on');
}}

// ── Nav scroll ────────────────────────────────────────────────────────────
function scrollTo(id) {{
  document.getElementById(id).scrollIntoView({{behavior: 'smooth', block: 'start'}});
  document.querySelectorAll('.nl').forEach(l => l.classList.remove('on'));
  event.target.classList.add('on');
}}

// ── Number format ─────────────────────────────────────────────────────────
function fmt(n) {{
  if (n >= 1e7) return (n/1e7).toFixed(2) + ' Cr';
  if (n >= 1e5) return (n/1e5).toFixed(2) + ' L';
  if (n >= 1e3) return (n/1e3).toFixed(1) + 'K';
  return n.toFixed(0);
}}

// ── Init ──────────────────────────────────────────────────────────────────
buildGrid();
</script>
</body>
</html>"""

out = DATA / "dashboard.html"
out.write_text(HTML, encoding="utf-8")
print(f"Written → {out}  ({out.stat().st_size/1024:.0f} KB)")
