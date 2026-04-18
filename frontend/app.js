// ULTRAMAX Frontend v3.0

// Auto-detect backend URL — always same origin since FastAPI serves both
let _stored = localStorage.getItem('um_backend') || '';
let backendUrl = (_stored && !_stored.includes('localhost')) ? _stored : window.location.origin;
let apiKey  = localStorage.getItem('um_key')   || '';
let dsKey   = localStorage.getItem('um_dskey') || '';
let fredKey = localStorage.getItem('um_fred')  || '';
let redditId = localStorage.getItem('um_reddit_id') || '';
let redditSecret = localStorage.getItem('um_reddit_secret') || '';
let alpacaKey = localStorage.getItem('um_alpaca_key') || '';
let alpacaSecret = localStorage.getItem('um_alpaca_secret') || '';
let asset   = 'BTC';
let horizon = 4;
let chartInst = null, mainSeries = null, predSeries = null;
let currentTab = 'crypto';
let lastResult = null;
let _predicting = false;
let _alertsCache = [];
let useR1 = localStorage.getItem('um_use_r1') !== 'false'; // default ON

const ASSETS = {
  crypto: ['BTC','ETH','SOL','BNB','XRP','DOGE'],
  stocks: ['AAPL','TSLA','NVDA','MSFT','GOOGL','SPY'],
  macro:  ['GC=F','CL=F','SI=F','XOM','LMT','RTX'],
};

const fmtP = p => {
  if (!p || isNaN(p)) return '—';
  if (p > 10000) return '$' + Number(p).toLocaleString('en', {maximumFractionDigits: 0});
  if (p > 100)   return '$' + Number(p).toLocaleString('en', {maximumFractionDigits: 2});
  return '$' + Number(p).toFixed(4);
};
const fmtPct = p => p == null || isNaN(p) ? '—' : `${p >= 0 ? '+' : ''}${Number(p).toFixed(2)}%`;

window.addEventListener('DOMContentLoaded', () => {
  renderAssets();
  syncR1Toggle();
  setDot('idle','READY'); checkBackend();
  setInterval(checkBackend, 30000);
  setInterval(tickPrice, 15000);
  setInterval(fetchAlerts, 60000);
  loadChart();
  retroactivelyScoreHistory();
  fetchAlerts();
  pushSettingsToBackend();
});

function toggleR1() {
  useR1 = !useR1;
  localStorage.setItem('um_use_r1', useR1 ? 'true' : 'false');
  syncR1Toggle();
  showToast(useR1 ? 'R1 ON — Deep reasoning (slower)' : 'R1 OFF — V3 fast mode');
}

function syncR1Toggle() {
  const el = document.getElementById('r1-toggle');
  const lbl = document.getElementById('r1-label');
  if (el) el.className = `r1-toggle ${useR1 ? 'on' : ''}`;
  if (lbl) lbl.textContent = useR1 ? 'R1' : 'V3';
}

async function checkBackend() {
  try {
    const r = await fetch(`${backendUrl}/health`, { signal: AbortSignal.timeout(10000) });
    const d = await r.json();
    document.getElementById('backend-dot').className = 'backend-dot ok';
    document.getElementById('backend-txt').textContent = `Backend v${d.version} · ${backendUrl}`;
    setDot('on', dsKey ? (useR1 ? 'READY — R1 mode' : 'READY — V3 fast') : 'READY');
    return true;
  } catch {
    document.getElementById('backend-dot').className = 'backend-dot err';
    document.getElementById('backend-txt').textContent = 'Backend not connected';
    setDot('idle','BACKEND OFFLINE');
    return false;
  }
}

function setDot(state, txt) {
  const d = document.getElementById('dot'), s = document.getElementById('stxt');
  if (d) d.className = `status-dot ${state}`;
  if (s && txt) s.textContent = txt;
}

async function tickPrice() {
  try {
    const r = await fetch(`${backendUrl}/price?asset=${asset}`, { signal: AbortSignal.timeout(25000) });
    if (!r.ok) { console.warn('Price fetch failed:', r.status); return; }
    const d = await r.json();
    const el = document.getElementById('plive'), ce = document.getElementById('pchg');
    if (el && d.price) el.textContent = fmtP(d.price);
    else if (el && d.error) el.textContent = 'err';
    if (ce && d.chg != null) { ce.textContent = fmtPct(d.chg); ce.className = `price-chg ${d.chg >= 0 ? 'up' : 'dn'}`; }
  } catch(e) { console.warn('Price tick error:', e.message); }
}

function renderAssets() {
  const row = document.getElementById('asset-row');
  if (!row) return;
  row.innerHTML = (ASSETS[currentTab] || ASSETS.crypto).map(a =>
    `<button class="asset-btn ${a === asset ? 'active' : ''}" onclick="selA('${a}')">${a}</button>`
  ).join('');
}

function switchTab(tab) {
  currentTab = tab;
  ['c','s','m'].forEach(t => document.getElementById(`tab-${t}`)?.classList.remove('active'));
  const tabMap = { crypto:'c', stocks:'s', macro:'m' };
  document.getElementById(`tab-${tabMap[tab]}`)?.classList.add('active');
  asset = ASSETS[tab][0];
  renderAssets();
  closeResult();
  // Clear stale price immediately then fetch fresh data
  const el = document.getElementById('plive'), ce = document.getElementById('pchg');
  if (el) el.textContent = '...';
  if (ce) { ce.textContent = ''; ce.className = 'price-chg'; }
  tickPrice();
  loadChart();
}

function selA(a) {
  asset = a;
  renderAssets();
  closeResult();
  // Clear stale price immediately then fetch fresh data
  const el = document.getElementById('plive'), ce = document.getElementById('pchg');
  if (el) el.textContent = '...';
  if (ce) { ce.textContent = ''; ce.className = 'price-chg'; }
  tickPrice();
  loadChart();
}

function setHz(h, btn) {
  horizon = h;
  document.querySelectorAll('.hz-btn').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
  closeResult();
}

async function loadChart() {
  const el = document.getElementById('chart');
  const errEl = document.getElementById('chart-err');
  if (!el) return;

  // Destroy previous chart
  if (chartInst) { try { chartInst.remove(); } catch(e) {} }
  chartInst = null; mainSeries = null; predSeries = null;
  el.innerHTML = '';
  if (errEl) errEl.textContent = 'Loading chart...';

  // Step 1: Check library loaded
  if (typeof LightweightCharts === 'undefined') {
    if (errEl) errEl.textContent = 'Chart library failed to load';
    return;
  }

  // Step 2: Fetch candle data
  let candles = [];
  try {
    const r = await fetch(`${backendUrl}/candles?asset=${asset}&interval=1h&limit=120`, { signal: AbortSignal.timeout(30000) });
    if (!r.ok) { if (errEl) errEl.textContent = `Candle fetch HTTP ${r.status}`; return; }
    const d = await r.json();
    if (d.error) { if (errEl) errEl.textContent = d.error; return; }
    candles = (d.candles || []).filter(c => c.open != null && c.high != null && c.low != null && c.close != null);
  } catch(e) {
    if (errEl) errEl.textContent = `Fetch error: ${e.message}`;
    return;
  }
  if (!candles.length) { if (errEl) errEl.textContent = `No candle data for ${asset}`; return; }

  // Step 3: Clean + sort data (all values must be plain numbers, time ascending)
  const data = candles.map(c => ({
    time: Math.floor(Number(c.time)),
    open: +c.open, high: +c.high, low: +c.low, close: +c.close
  })).filter(c => c.time > 0 && isFinite(c.open) && isFinite(c.close))
    .sort((a,b) => a.time - b.time);

  if (!data.length) { if (errEl) errEl.textContent = 'No valid candle data'; return; }

  // Step 4: Measure container (use parent .chart-wrap which has flex sizing)
  const rect = el.getBoundingClientRect();
  const w = Math.floor(rect.width) || window.innerWidth || 360;
  const h = Math.floor(rect.height) || 300;

  // Step 5: Create chart
  try {
    chartInst = LightweightCharts.createChart(el, {
      width: w,
      height: h,
      layout: {
        background: { type: LightweightCharts.ColorType.Solid, color: '#04080f' },
        textColor: '#5a7a9a',
      },
      grid: {
        vertLines: { color: 'rgba(0,229,255,.08)' },
        horzLines: { color: 'rgba(0,229,255,.08)' },
      },
      rightPriceScale: { borderColor: 'rgba(0,229,255,.25)' },
      timeScale: { borderColor: 'rgba(0,229,255,.25)', timeVisible: true },
      crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    });

    mainSeries = chartInst.addCandlestickSeries({
      upColor: '#00e676', downColor: '#ff1744',
      borderUpColor: '#00e676', borderDownColor: '#ff1744',
      wickUpColor: '#00e676', wickDownColor: '#ff1744',
    });

    mainSeries.setData(data);
    chartInst.timeScale().fitContent();

    // Update price display from chart data
    const last = data[data.length - 1];
    const prev = data.length > 1 ? data[data.length - 2] : null;
    if (last) {
      const pe = document.getElementById('plive');
      if (pe) pe.textContent = fmtP(last.close);
      if (prev) {
        const chg = (last.close - prev.close) / prev.close * 100;
        const ce = document.getElementById('pchg');
        if (ce) { ce.textContent = fmtPct(chg); ce.className = `price-chg ${chg >= 0 ? 'up' : 'dn'}`; }
      }
    }
    if (errEl) errEl.textContent = '';
  } catch(e) {
    console.error('Chart create error:', e);
    if (errEl) errEl.textContent = `Chart error: ${e.message}`;
  }
}

window.addEventListener('resize', () => {
  if (!chartInst) return;
  const el = document.getElementById('chart');
  if (!el) return;
  const rect = el.getBoundingClientRect();
  const w = Math.floor(rect.width) || window.innerWidth || 360;
  const h = Math.floor(rect.height) || 300;
  try { chartInst.resize(w, h); } catch(e) {}
});

async function predict() {
  if (_predicting) return;
  const ok = await checkBackend();
  if (!ok) { showToast('⚠ Backend not connected'); return; }

  _predicting = true;
  const pb = document.getElementById('pb');
  const loading = document.getElementById('loading');
  if (pb) pb.disabled = true;
  if (loading) loading.classList.add('on');

  clearSyslog();
  setAgentStep(1,''); setAgentStep(2,''); setAgentStep(3,'');
  syslog(`🚀 ${asset} ${horizon}H`);
  document.getElementById('lstep').textContent = 'Fetching market data...';

  const t1 = setTimeout(() => { setAgentStep(1,'running'); document.getElementById('lstep').textContent = '📐 Quant: 30+ indicators...'; }, 1000);
  const t2 = setTimeout(() => { setAgentStep(1,'done'); setAgentStep(2,'running'); document.getElementById('lstep').textContent = '📰 News: scanning 10+ feeds...'; }, 9000);
  const modelLabel = useR1 ? 'DeepSeek R1' : 'DeepSeek V3';
  const t3 = setTimeout(() => { setAgentStep(2,'done'); setAgentStep(3,'running'); document.getElementById('lstep').textContent = `🧠 ${modelLabel} deciding...`; }, useR1 ? 20000 : 8000);

  try {
    const r = await fetch(`${backendUrl}/predict`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ asset, horizon, api_key:apiKey, ds_key:dsKey||null, use_r1:useR1 }),
      signal: AbortSignal.timeout(useR1 ? 680000 : 120000)
    });
    [t1,t2,t3].forEach(clearTimeout);

    if (!r.ok) { const e = await r.json().catch(()=>({detail:`HTTP ${r.status}`})); throw new Error(e.detail); }
    const result = await r.json();
    setAgentStep(1,'done'); setAgentStep(2,'done'); setAgentStep(3,'done');

    if (result.logs) result.logs.forEach(l => syslog(l.msg, l.ts));
    lastResult = result;
    displayResult(result);
    if (result.candles) drawPrediction(result);
    saveToHistory(result);
    renderHistory();
  } catch(e) {
    [t1,t2,t3].forEach(clearTimeout);
    showToast(`❌ ${e.message}`);
    syslog(`ERROR: ${e.message}`);
  } finally {
    if (loading) loading.classList.remove('on');
    if (pb) pb.disabled = false;
    _predicting = false;
  }
}

function displayResult(r) {
  const dec = r.decision || 'NO_TRADE';
  const set = (id, txt) => { const el = document.getElementById(id); if (el) el.textContent = txt; };
  const setH = (id, html) => { const el = document.getElementById(id); if (el) el.innerHTML = html; };

  const vEl = document.getElementById('rp-verdict');
  if (vEl) { vEl.textContent = dec.replace('_',' '); vEl.className = `verdict ${dec}`; }
  set('rp-conf', `${r.confidence}% confidence`);

  const pu = Math.round(r.prob_up||50), pd = Math.round(r.prob_down||50);
  set('pu-txt', `↑${pu}%`); set('pd-txt', `↓${pd}%`);
  const bup = document.getElementById('bup'), bdn = document.getElementById('bdn');
  if (bup) bup.style.width = `${pu}%`;
  if (bdn) bdn.style.width = `${pd}%`;

  const q = r.quant||{}, n = r.news||{};
  const setAb = (i, verdict, conf, cls) => {
    const ab = document.getElementById(`ab${i}`), ac = document.getElementById(`ac${i}`);
    if (ab) { ab.textContent = verdict; ab.className = `ab-verdict ${cls}`; }
    if (ac) ac.textContent = conf;
  };
  setAb(1, q.direction||'—', `${q.confidence||0}%`, q.direction==='BUY'?'bull':q.direction==='SELL'?'bear':'neu');
  const sn = n.sentiment||'neutral';
  setAb(2, sn.toUpperCase(), `${n.sentiment_score>0?'+':''}${n.sentiment_score||0}`, sn==='bullish'?'bull':sn==='bearish'?'bear':'neu');
  setAb(3, dec.replace('_',' '), `${r.confidence}%`, dec==='BUY'?'bull':dec==='SELL'?'bear':'neu');

  const cur = r.ind?.cur;
  set('rp-entry', fmtP(cur));

  if (dec === 'NO_TRADE') {
    const dp = r.predicted_price || r.monte_carlo?.median;
    set('rp-target', dp && cur
      ? `${fmtP(dp)} (${fmtPct((dp-cur)/cur*100)}) — not traded`
      : r.original_decision ? `AI wanted ${r.original_decision} — gate blocked` : 'Abstained');
  } else {
    const tgt = r.price_target;
    set('rp-target', tgt && cur ? `${fmtP(tgt)} (${fmtPct((tgt-cur)/cur*100)})` : '—');
  }
  set('rp-bull', fmtP(r.price_target_bull || r.monte_carlo?.bull));
  set('rp-bear', fmtP(r.price_target_bear || r.monte_carlo?.bear));

  const ind = r.ind||{};
  const hmm = ind.hmm_probs||{};
  const hmmState = Object.entries(hmm).sort((a,b)=>b[1]-a[1])[0];
  set('rp-regime', ind.regime||'—');
  set('rp-hmm', hmmState ? `${hmmState[0]} ${(hmmState[1]*100).toFixed(0)}%` : '—');
  set('rp-hurst', ind.hurst_exp != null
    ? `${ind.hurst_exp.toFixed(3)} (${ind.hurst_exp>0.55?'trending':ind.hurst_exp<0.45?'mean-rev':'random'})` : '—');

  const modelMap = {'deepseek-r1':'🧠 R1','deepseek-v3':'⚡ V3','deepseek-chat':'⚡ V3','gpt-4o':'🎯 GPT-4o','error':'❌ Error'};
  let confLine = modelMap[r.agent_model] || r.agent_model || '—';
  if (r.raw_ai_conf) confLine += ` · AI:${r.raw_ai_conf}%`;
  if (r.bayesian_conf) confLine += ` B:${r.bayesian_conf?.toFixed(0)}%`;
  if (r.ml?.available) confLine += ` ML:${r.ml.score?.toFixed(0) || r.ml.confidence?.toFixed(0)}%`;
  set('rp-model', confLine);

  // Cluster info
  if (r.cluster && r.cluster.available) {
    set('rp-cluster', `#${r.cluster.cluster_id} · ${r.cluster.n_members} matches · ${(r.cluster.win_rate_4h||50).toFixed(0)}% win`);
  } else { set('rp-cluster', 'Not built yet'); }

  // Sentiment info
  if (r.sentiment_data && r.sentiment_data.sources_available > 0) {
    const sc = r.sentiment_data.combined_score || 0;
    set('rp-sentiment', `${sc > 0 ? '+' : ''}${sc.toFixed(2)} (${r.sentiment_data.sources_available} sources)`);
  } else { set('rp-sentiment', 'No data'); }

  // Correlation info
  if (r.correlation && r.correlation.available) {
    const parts = [];
    if (r.correlation.btc_corr != null) parts.push(`BTC:${r.correlation.btc_corr.toFixed(2)}`);
    if (r.correlation.spy_corr != null) parts.push(`SPY:${r.correlation.spy_corr.toFixed(2)}`);
    set('rp-corr', parts.join(' · ') || '—');
  } else { set('rp-corr', 'Building...'); }

  // ML ensemble info
  if (r.ml?.available) {
    const agree = r.ml.agreement ? '✓ agree' : '✗ disagree';
    set('rp-ml', `${r.ml.score?.toFixed(0)}% · XGB:${r.ml.xgb_score||'?'} RF:${r.ml.rf_score||'?'} · ${agree}`);
  } else { set('rp-ml', 'Not trained'); }

  const sim = r.similarity||{};
  const parts = [];
  if (r.insight) parts.push(`<strong>R1 Reasoning:</strong> ${r.insight}`);
  if (r.primary_reason) parts.push(`<strong>Key factor:</strong> ${r.primary_reason}`);
  if (n.reasoning) parts.push(`<strong>News:</strong> ${n.reasoning}`);
  if (n.macro_warning) parts.push(`<strong>⚠ Macro:</strong> ${n.macro_warning}`);
  if (sim.count > 0) parts.push(`<strong>Historical analogs:</strong> ${sim.count} similar periods — avg ${sim.avg_fwd_4h>0?'+':''}${(sim.avg_fwd_4h||0).toFixed(2)}% 4H return, ${(sim.win_rate||0).toFixed(0)}% win rate`);
  setH('rp-insight', parts.join('<br><br>') || '—');

  const gb = document.getElementById('gate-box');
  if (gb) { if (r.gate_reason) { gb.textContent = r.gate_reason; gb.style.display='block'; } else gb.style.display='none'; }

  const ld = document.getElementById('result-log'), le = document.getElementById('log-entries');
  if (ld && le && r.logs?.length) {
    ld.style.display='block';
    le.innerHTML = r.logs.map(l => `<div class="log-entry">[${l.ts}ms] ${l.msg}</div>`).join('');
  }

  document.getElementById('result-panel').classList.add('show');
}

function closeResult() { document.getElementById('result-panel')?.classList.remove('show'); }

function drawPrediction(r) {
  if (!chartInst || !mainSeries) return;
  try {
    if (predSeries) { chartInst.removeSeries(predSeries); predSeries = null; }
    predSeries = chartInst.addLineSeries({
      color:'#a855f7', lineWidth:2, lineStyle:LightweightCharts.LineStyle.Dashed,
      lastValueVisible:true, priceLineVisible:false, title:'→ AI'
    });
    const candles = r.candles;
    if (!candles?.length) return;
    const last = candles[candles.length-1];
    const target = r.price_target || r.predicted_price || r.monte_carlo?.median;
    if (!target) return;
    const hSecs = horizon*3600;
    const pts = [{ time:last.time, value:last.close }];
    if (r.predicted_path?.length >= 3) {
      const step = Math.floor(hSecs/(r.predicted_path.length+1));
      r.predicted_path.forEach((p,i) => { if (p&&!isNaN(p)) pts.push({ time:last.time+step*(i+1), value:p }); });
    }
    pts.push({ time:last.time+hSecs, value:target });
    predSeries.setData(pts);
    chartInst.timeScale().scrollToPosition(-8, false);
  } catch {}
}

function syslog(msg, tsMs=null) {
  const el = document.getElementById('syslog');
  if (!el) return;
  const line = document.createElement('div');
  const ts = tsMs!=null ? `${tsMs}ms` : new Date().toISOString().split('T')[1].slice(0,8);
  line.style.cssText='color:rgba(0,229,255,.7);padding:1px 0;font-size:7px;';
  line.textContent=`[${ts}] ${msg}`;
  el.insertBefore(line, el.firstChild);
  while(el.children.length>25) el.removeChild(el.lastChild);
}

function clearSyslog() { const el = document.getElementById('syslog'); if (el) el.innerHTML=''; }

function setAgentStep(n, state) {
  const dot = document.getElementById(`a${n}dot`), txt = document.getElementById(`a${n}txt`);
  const L = {1:'📐 Quant Agent',2:'📰 News Agent',3:'🧠 Decision Agent'};
  if (dot) dot.className = `astep-dot ${state}`;
  if (txt) txt.textContent = `${L[n]} — ${state==='running'?'analyzing...':state==='done'?'complete ✓':'waiting'}`;
}

function getHistory() { try { return JSON.parse(localStorage.getItem('um_history_v3')||'[]'); } catch { return []; } }
function saveHistory(h) { localStorage.setItem('um_history_v3', JSON.stringify(h.slice(0,500))); }

function saveToHistory(result) {
  const id = `pred_${Date.now()}_${Math.random().toString(36).slice(2,5)}`;
  const entry = {
    id, saved_at:Date.now(), asset, horizon,
    horizon_label: horizon<24 ? `${horizon}H` : horizon===24?'1D':horizon===72?'3D':'1W',
    decision:result.decision, confidence:result.confidence,
    entry_price:result.ind?.cur, target_price:result.price_target,
    predicted_price:result.predicted_price||result.monte_carlo?.median,
    target_bull:result.price_target_bull||result.monte_carlo?.bull,
    target_bear:result.price_target_bear||result.monte_carlo?.bear,
    prob_up:result.prob_up, prob_down:result.prob_down,
    original_decision:result.original_decision||null,
    gate_reason:result.gate_reason||null,
    insight:result.insight, primary_reason:result.primary_reason,
    agent_model:result.agent_model,
    quant_verdict:`${result.quant?.direction||'—'} ${result.quant?.confidence||0}%`,
    news_verdict:`${result.news?.sentiment||'neutral'} (${result.news?.sentiment_score||0})`,
    regime:result.ind?.regime, hurst_exp:result.ind?.hurst_exp,
    ind_snapshot:result.ind||null,
    feedback:null, outcome_price:null, target_hit:null,
  };
  const history = getHistory();
  history.unshift(entry);
  saveHistory(history);
  fetch(`${backendUrl}/history/save`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prediction:entry})}).catch(()=>{});
}

function retroactivelyScoreHistory() {
  const history = getHistory();
  let changed = false;
  const updated = history.map(e => {
    if (e.feedback!==null && e.feedback!==undefined) return e;
    if (!e.outcome_price||!e.entry_price) return e;
    const fb = scorePrediction(e);
    if (!fb) return e;
    changed = true;
    return {...e, feedback:fb};
  });
  if (changed) saveHistory(updated);
}

function scorePrediction(e) {
  const movedPct = (e.outcome_price - e.entry_price) / e.entry_price * 100;
  // BUY/SELL — simple direction check
  if (e.decision==='BUY') return movedPct>0?'correct':'wrong';
  if (e.decision==='SELL') return movedPct<0?'correct':'wrong';
  // NO_TRADE — evaluate AI's prediction accuracy, not the gate
  const target = e.predicted_price || e.target_price;
  const direction = e.original_decision;
  if (!direction && !target) return null;
  if (direction && !target) {
    if (direction==='BUY') return movedPct>0.5?'correct':'wrong';
    if (direction==='SELL') return movedPct<-0.5?'correct':'wrong';
    return null;
  }
  // Have target — check direction + proximity (40% of predicted move)
  const predMovePct = (target - e.entry_price) / e.entry_price * 100;
  if (Math.abs(predMovePct)<0.01) return null;
  const directionCorrect = (predMovePct>0 && movedPct>0) || (predMovePct<0 && movedPct<0);
  if (!directionCorrect) return 'wrong';
  return (movedPct/predMovePct)>=0.4 ? 'correct' : 'wrong';
}

function openHistory() { renderHistory(); document.getElementById('history-panel')?.classList.add('on'); }
function closeHistory() { document.getElementById('history-panel')?.classList.remove('on'); }

function renderHistory() {
  const history = getHistory();
  const list = document.getElementById('hp-list');
  if (!list) return;
  const rated = history.filter(h=>h.feedback==='correct'||h.feedback==='wrong');
  const correct = rated.filter(h=>h.feedback==='correct');
  const set = (id,v)=>{const el=document.getElementById(id);if(el)el.textContent=v;};
  set('hs-total',history.length); set('hs-correct',correct.length);
  set('hs-wrong',rated.length-correct.length);
  set('hs-acc',rated.length?`${Math.round(correct.length/rated.length*100)}%`:'—');

  if (!history.length) {
    list.innerHTML='<div style="text-align:center;padding:40px;font-size:9px;color:var(--muted)">No predictions yet</div>';
    return;
  }

  list.innerHTML = history.map(e => {
    const expiresAt = e.saved_at+(e.horizon*3600000), now = Date.now();
    const sat = new Date(e.saved_at).toLocaleString('en-US',{timeZone:'Asia/Riyadh',hour12:false,month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});
    let fbHtml = '';
    if (e.feedback==='correct') {
      const pct = e.outcome_price&&e.entry_price?((e.outcome_price-e.entry_price)/e.entry_price*100):null;
      fbHtml=`<div class="he-fb correct">✓ CORRECT${pct!=null?` · ${fmtPct(pct)}`:''}</div>`;
    } else if (e.feedback==='wrong') {
      const pct = e.outcome_price&&e.entry_price?((e.outcome_price-e.entry_price)/e.entry_price*100):null;
      fbHtml=`<div class="he-fb wrong">✗ WRONG${pct!=null?` · ${fmtPct(pct)}`:''}</div>`;
    } else if (e.feedback==='skipped') {
      const pct = e.outcome_price&&e.entry_price?((e.outcome_price-e.entry_price)/e.entry_price*100):null;
      fbHtml=`<div class="he-fb pending">⏸ NO TRADE${pct!=null?` · ${fmtPct(pct)}`:''}</div>`;
    } else if (now>=expiresAt) {
      fbHtml=`<div class="he-fb pending" style="display:flex;justify-content:space-between;align-items:center"><span>⏰ Expired</span><button class="check-btn" onclick="event.stopPropagation();checkOutcome('${e.id}')">GET PRICE</button></div>`;
    } else {
      fbHtml=`<div class="he-fb pending">⏳ ${Math.max(0,(expiresAt-now)/3600000).toFixed(1)}h remaining</div>`;
    }
    const tgtDisplay = e.decision==='NO_TRADE'&&e.predicted_price ? `${fmtP(e.predicted_price)} (not traded)` : fmtP(e.target_price);
    return `<div class="he ${e.feedback==='correct'?'correct':e.feedback==='wrong'?'wrong':''}" onclick="openHistDetail('${e.id}')">
      <div class="he-top"><span class="he-asset">${e.asset}</span><span class="he-dec ${e.decision}">${e.decision.replace('_',' ')}</span></div>
      <div class="he-row"><span>🕐 ${sat} · ${e.horizon_label||e.horizon+'H'}</span><span>${fmtP(e.entry_price)}</span></div>
      <div class="he-row"><span>Target: ${tgtDisplay}</span><span>${e.confidence}%</span></div>
      ${e.gate_reason?`<div style="font-size:7px;color:var(--red);margin-top:2px">${e.gate_reason}</div>`:''}
      ${fbHtml}
    </div>`;
  }).join('');
}

function openHistDetail(id) {
  const e = getHistory().find(h=>h.id===id);
  if (!e) return;
  const expiresAt=e.saved_at+(e.horizon*3600000), now=Date.now(), expired=now>=expiresAt;
  const sat=new Date(e.saved_at).toLocaleString('en-US',{timeZone:'Asia/Riyadh',hour12:false});
  const satExp=new Date(expiresAt).toLocaleString('en-US',{timeZone:'Asia/Riyadh',hour12:false});
  let pct=null;
  if (e.outcome_price&&e.entry_price) pct=(e.outcome_price-e.entry_price)/e.entry_price*100;

  let verdictHtml='';
  if (e.feedback==='correct') verdictHtml=`<div style="color:var(--green);font-size:22px;font-family:'Bebas Neue'">✓ CORRECT${pct!=null?' · '+fmtPct(pct):''}</div>`;
  else if (e.feedback==='wrong') verdictHtml=`<div style="color:var(--red);font-size:22px;font-family:'Bebas Neue'">✗ WRONG${pct!=null?' · '+fmtPct(pct):''}</div>`;
  else if (e.feedback==='skipped'&&e.outcome_price) {
    verdictHtml=`<div style="color:var(--muted);font-size:16px;font-family:'Bebas Neue'">⏸ NO TRADE · Moved ${fmtPct(pct)}</div>`;
  } else if (!expired) {
    verdictHtml=`<div style="color:var(--yellow);font-family:'Bebas Neue';font-size:18px">⏳ ${Math.max(0,(expiresAt-now)/3600000).toFixed(1)}H REMAINING</div>`;
  } else verdictHtml=`<div style="color:var(--yellow);font-family:'Bebas Neue';font-size:16px">⏰ EXPIRED — tap GET PRICE</div>`;

  let targetAccHtml='';
  if (e.outcome_price&&e.entry_price) {
    const predP=e.predicted_price||e.target_price;
    if (e.decision==='NO_TRADE') {
      if (predP) { const pp=(predP-e.entry_price)/e.entry_price*100; targetAccHtml=`AI predicted ${fmtP(predP)} (${fmtPct(pp)}) · Actual: ${fmtPct(pct)} · Gap: ${Math.abs(pct-pp).toFixed(2)}%`; }
      else targetAccHtml=`Market moved ${fmtPct(pct)}`;
    } else if (e.target_price) {
      const tp=(e.target_price-e.entry_price)/e.entry_price*100;
      const hit=tp>0?pct>=tp*0.8:pct<=tp*0.8;
      targetAccHtml=hit?`🎯 Target reached — actual: ${fmtPct(pct)} vs predicted: ${fmtPct(tp)}`
                       :`📍 Missed by ${Math.abs(pct-tp).toFixed(2)}% — actual: ${fmtPct(pct)} vs predicted: ${fmtPct(tp)}`;
    }
  } else if (e.predicted_price||e.target_price) {
    targetAccHtml=`Predicted: ${fmtP(e.predicted_price||e.target_price)} — ${expired?'tap GET PRICE for outcome':'not expired yet'}`;
  }

  const rows=[
    ['🕐 Predicted at',sat+' AST'],['⌛ Expires at',satExp+' AST'],
    ['💰 Entry price',fmtP(e.entry_price)],
    ['🎯 Predicted target',e.decision==='NO_TRADE'?(e.predicted_price?`${fmtP(e.predicted_price)} (not traded)`:e.original_decision?`AI wanted ${e.original_decision}`:'Abstained'):fmtP(e.target_price)],
    ['📊 Actual at expiry',e.outcome_price?fmtP(e.outcome_price):expired?'⏰ tap GET PRICE':'⏳ not expired'],
    ['📉 Price change',pct!=null?fmtPct(pct):'—'],
    ['🤖 Model',e.agent_model||'—'],['💪 Confidence',`${e.confidence}%`],
    ['📈 Bull target',fmtP(e.target_bull)],['📉 Bear target',fmtP(e.target_bear)],
    ['🌊 Regime',e.regime||'—'],['🏄 Hurst',e.hurst_exp!=null?e.hurst_exp.toFixed(3):'—'],
  ];

  // Build mini price chart with real candles
  const chartId = 'hist-chart-' + Date.now();

  document.body.insertAdjacentHTML('beforeend',`
    <div onclick="this.remove()" style="position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,.88);z-index:500;overflow-y:auto;padding:16px">
      <div onclick="event.stopPropagation()" style="background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:16px;max-width:420px;margin:0 auto">
        <div style="display:flex;justify-content:space-between;margin-bottom:12px">
          <div style="font-family:'Bebas Neue';font-size:18px;color:var(--cyan)">${e.asset} · ${e.decision.replace('_',' ')} · ${e.horizon_label||e.horizon+'H'}</div>
          <button onclick="event.stopPropagation();this.closest('[style*=z-index]').remove()" style="background:none;border:none;color:var(--muted);font-size:20px;cursor:pointer">✕</button>
        </div>
        <div style="margin-bottom:10px">${verdictHtml}</div>
        <div id="${chartId}" style="width:100%;height:200px;border:1px solid rgba(0,229,255,.12);border-radius:6px;margin-bottom:10px;position:relative;overflow:hidden"></div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--border);margin-bottom:10px">
          ${rows.map(([l,v])=>`<div style="background:var(--surface);padding:7px 10px"><div style="font-size:7px;color:var(--muted)">${l}</div><div style="font-size:10px">${v}</div></div>`).join('')}
        </div>
        ${targetAccHtml?`<div style="font-size:8px;color:var(--cyan);padding:6px 10px;background:rgba(0,229,255,.05);border-radius:4px;margin-bottom:8px">${targetAccHtml}</div>`:''}
        ${e.gate_reason?`<div style="font-size:8px;color:var(--red);padding:4px 10px;margin-bottom:8px">${e.gate_reason}</div>`:''}
        ${e.primary_reason?`<div style="font-size:8px;color:var(--text);padding:4px 10px;margin-bottom:6px"><strong>Key factor:</strong> ${e.primary_reason}</div>`:''}
        ${e.insight?`<div style="font-size:8px;color:var(--muted);line-height:1.6;padding:6px 10px;margin-bottom:8px"><strong style="color:var(--cyan)">R1 Reasoning:</strong><br>${e.insight}</div>`:''}
        ${expired&&!e.outcome_price?`<button onclick="event.stopPropagation();checkOutcome('${e.id}');this.closest('[style*=z-index]').remove();" class="check-btn" style="width:100%;padding:9px;font-size:10px;margin-top:4px">🔄 GET PRICE AT EXPIRY</button>`:''}
      </div>
    </div>`);

  // Render real candlestick chart
  drawHistoryChart(chartId, e);
}

async function drawHistoryChart(containerId, e) {
  const container = document.getElementById(containerId);
  if (!container || typeof LightweightCharts === 'undefined') return;

  const entry = e.entry_price;
  const target = e.target_price || e.predicted_price;
  const actual = e.outcome_price;
  const entryTime = Math.floor(e.saved_at / 1000);
  const expiryTime = entryTime + e.horizon * 3600;

  // Fetch real candle data
  let candles = [];
  try {
    const r = await fetch(`${backendUrl}/candles?asset=${e.asset}&interval=1h&limit=200`, { signal: AbortSignal.timeout(15000) });
    if (r.ok) {
      const d = await r.json();
      candles = (d.candles || [])
        .filter(c => c.open != null && c.close != null && c.high != null && c.low != null)
        .map(c => ({ time: Math.floor(Number(c.time)), open: +c.open, high: +c.high, low: +c.low, close: +c.close }))
        .filter(c => c.time > 0 && isFinite(c.open))
        .sort((a, b) => a.time - b.time);
    }
  } catch (err) { console.warn('History chart candle fetch:', err); }

  if (!candles.length) {
    container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;font-size:8px;color:var(--muted)">No candle data available</div>';
    return;
  }

  const rect = container.getBoundingClientRect();
  const w = Math.floor(rect.width) || 380;
  const h = Math.floor(rect.height) || 200;

  const chart = LightweightCharts.createChart(container, {
    width: w, height: h,
    layout: { background: { type: LightweightCharts.ColorType.Solid, color: '#04080f' }, textColor: '#5a7a9a' },
    grid: { vertLines: { color: 'rgba(0,229,255,.06)' }, horzLines: { color: 'rgba(0,229,255,.06)' } },
    rightPriceScale: { borderColor: 'rgba(0,229,255,.2)', scaleMargins: { top: 0.1, bottom: 0.1 } },
    timeScale: { borderColor: 'rgba(0,229,255,.2)', timeVisible: true, rightOffset: 3 },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    handleScroll: false, handleScale: false,
  });

  // Candlestick series
  const series = chart.addCandlestickSeries({
    upColor: '#00e676', downColor: '#ff1744',
    borderUpColor: '#00e676', borderDownColor: '#ff1744',
    wickUpColor: '#00e676', wickDownColor: '#ff1744',
  });
  series.setData(candles);

  // Entry price line (cyan, dashed)
  if (entry) {
    series.createPriceLine({
      price: entry, color: '#00e5ff', lineWidth: 2,
      lineStyle: LightweightCharts.LineStyle.Dashed,
      axisLabelVisible: true, title: 'ENTRY',
    });
  }

  // Target price line (purple, dashed)
  if (target) {
    series.createPriceLine({
      price: target, color: '#a855f7', lineWidth: 2,
      lineStyle: LightweightCharts.LineStyle.Dashed,
      axisLabelVisible: true, title: 'TARGET',
    });
  }

  // Actual outcome price line (green/red, solid)
  if (actual) {
    const outcomeColor = actual >= entry ? '#00e676' : '#ff1744';
    series.createPriceLine({
      price: actual, color: outcomeColor, lineWidth: 2,
      lineStyle: LightweightCharts.LineStyle.Solid,
      axisLabelVisible: true,
      title: (actual >= entry ? '✓' : '✗') + ' OUTCOME',
    });
  }

  // Bull/bear target lines (faint)
  if (e.target_bull) {
    series.createPriceLine({
      price: e.target_bull, color: 'rgba(0,230,118,.3)', lineWidth: 1,
      lineStyle: LightweightCharts.LineStyle.Dotted,
      axisLabelVisible: false, title: '',
    });
  }
  if (e.target_bear) {
    series.createPriceLine({
      price: e.target_bear, color: 'rgba(255,23,68,.3)', lineWidth: 1,
      lineStyle: LightweightCharts.LineStyle.Dotted,
      axisLabelVisible: false, title: '',
    });
  }

  // Markers: entry point and expiry point on the candles
  const markers = [];
  // Find candle closest to entry time
  let entryIdx = candles.findIndex(c => c.time >= entryTime);
  if (entryIdx < 0) entryIdx = candles.length - 1;
  markers.push({
    time: candles[entryIdx].time, position: 'aboveBar', color: '#00e5ff',
    shape: 'arrowDown', text: 'ENTRY'
  });

  // Find candle closest to expiry time
  if (actual) {
    let expiryIdx = candles.findIndex(c => c.time >= expiryTime);
    if (expiryIdx < 0) expiryIdx = candles.length - 1;
    const outcomeColor = actual >= entry ? '#00e676' : '#ff1744';
    markers.push({
      time: candles[expiryIdx].time, position: 'belowBar', color: outcomeColor,
      shape: 'arrowUp', text: (actual >= entry ? '✓' : '✗') + ' OUTCOME'
    });
  }

  series.setMarkers(markers.sort((a, b) => a.time - b.time));

  // Scroll to show the entry area
  const entryCandle = candles[entryIdx];
  if (entryCandle) {
    chart.timeScale().setVisibleRange({
      from: entryCandle.time - e.horizon * 3600 * 2,
      to: entryCandle.time + e.horizon * 3600 * 3,
    });
  } else {
    chart.timeScale().fitContent();
  }

  // Expiry time vertical line using a separate line series
  const expiryCandle = candles.find(c => c.time >= expiryTime);
  if (expiryCandle) {
    const timeLabel = new Date(expiryTime * 1000).toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' });
    // Create a thin vertical marker via a separate series
    const vline = chart.addLineSeries({ color: 'rgba(255,214,0,.4)', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
    const lo = Math.min(...candles.slice(-50).map(c => c.low));
    const hi = Math.max(...candles.slice(-50).map(c => c.high));
    vline.setData([{ time: expiryCandle.time, value: lo }, { time: expiryCandle.time, value: hi }]);
  }
}

async function checkOutcome(id) {
  const history=getHistory(), entry=history.find(e=>e.id===id);
  if (!entry) return;
  showToast('🔄 Fetching outcome...');
  try {
    const r=await fetch(`${backendUrl}/history/outcome`,{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({pred_id:id,asset:entry.asset,entry_price:entry.entry_price,target_price:entry.target_price,predicted_price:entry.predicted_price||null,original_decision:entry.original_decision,decision:entry.decision,horizon:entry.horizon})
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const result=await r.json();
    let feedback=result.feedback;
    const updated=history.map(e=>e.id===id?{...e,feedback,outcome_price:result.price,target_hit:result.target_hit}:e);
    saveHistory(updated);
    renderHistory();
    showToast(`${feedback==='correct'?'✓':feedback==='wrong'?'✗':'⏸'} ${entry.asset}: ${fmtPct(result.moved_pct)} → ${feedback}`);
  } catch(e) { showToast(`⚠ ${e.message}`); }
}

function openModal() {
  document.getElementById('kmodal')?.classList.add('on');
  const set=(id,v)=>{const el=document.getElementById(id);if(el)el.value=v;};
  set('k-backend',backendUrl); set('k-openai',apiKey); set('k-deepseek',dsKey);
  set('k-fred',fredKey); set('k-reddit-id',redditId); set('k-reddit-secret',redditSecret);
  set('k-alpaca-key',alpacaKey); set('k-alpaca-secret',alpacaSecret);
}

function saveKeys() {
  const bk=document.getElementById('k-backend')?.value.trim();
  const oai=document.getElementById('k-openai')?.value.trim();
  const ds=document.getElementById('k-deepseek')?.value.trim();
  backendUrl=(bk && !bk.includes('localhost')) ? bk : window.location.origin; apiKey=oai||''; dsKey=ds||'';
  fredKey=document.getElementById('k-fred')?.value.trim()||'';
  redditId=document.getElementById('k-reddit-id')?.value.trim()||'';
  redditSecret=document.getElementById('k-reddit-secret')?.value.trim()||'';
  alpacaKey=document.getElementById('k-alpaca-key')?.value.trim()||'';
  alpacaSecret=document.getElementById('k-alpaca-secret')?.value.trim()||'';
  localStorage.setItem('um_backend',backendUrl);
  localStorage.setItem('um_key',apiKey);
  localStorage.setItem('um_dskey',dsKey);
  localStorage.setItem('um_fred',fredKey);
  localStorage.setItem('um_reddit_id',redditId);
  localStorage.setItem('um_reddit_secret',redditSecret);
  localStorage.setItem('um_alpaca_key',alpacaKey);
  localStorage.setItem('um_alpaca_secret',alpacaSecret);
  document.getElementById('kmodal')?.classList.remove('on');
  checkBackend();
  pushSettingsToBackend();
  showToast('✓ Keys saved');
}

async function pushSettingsToBackend() {
  const settings = {};
  if (fredKey) settings.FRED_API_KEY = fredKey;
  if (redditId) settings.REDDIT_CLIENT_ID = redditId;
  if (redditSecret) settings.REDDIT_CLIENT_SECRET = redditSecret;
  if (alpacaKey) settings.ALPACA_KEY = alpacaKey;
  if (alpacaSecret) settings.ALPACA_SECRET = alpacaSecret;
  if (Object.keys(settings).length === 0) return;
  try {
    await fetch(`${backendUrl}/settings`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(settings), signal: AbortSignal.timeout(5000)
    });
  } catch {}
}

async function fetchAlerts() {
  try {
    const r = await fetch(`${backendUrl}/alerts`, { signal: AbortSignal.timeout(5000) });
    if (!r.ok) return;
    const data = await r.json();
    _alertsCache = data.alerts || [];
    const btn = document.getElementById('alerts-btn');
    if (btn) btn.textContent = _alertsCache.length > 0 ? `ALERTS (${_alertsCache.length})` : 'ALERTS';
  } catch {}
}

function openAlerts() {
  document.getElementById('alerts-panel')?.classList.add('on');
  const list = document.getElementById('alerts-list');
  const info = document.getElementById('alerts-info');
  if (info) info.textContent = `${_alertsCache.length} active alerts`;
  if (!list) return;
  if (_alertsCache.length === 0) {
    list.innerHTML = '<div style="text-align:center;padding:40px;font-size:9px;color:var(--muted)">No high-confluence alerts right now</div>';
    return;
  }
  list.innerHTML = _alertsCache.map(a => {
    const dirCls = a.direction === 'BUY' ? 'BUY' : a.direction === 'SELL' ? 'SELL' : 'NO_TRADE';
    return `<div class="he" onclick="selA('${a.asset}');closeAlerts()">
      <div class="he-top">
        <span class="he-asset">${a.asset} · ${a.asset_name}</span>
        <span class="he-dec ${dirCls}">${a.direction} · ${a.score}/10</span>
      </div>
      <div class="he-row"><span>${fmtP(a.price)}</span><span>${a.regime}</span></div>
      <div style="font-size:7px;color:var(--cyan);margin-top:4px">${a.signals.join(' · ')}</div>
    </div>`;
  }).join('');
}

function closeAlerts() { document.getElementById('alerts-panel')?.classList.remove('on'); }

async function checkAllOutcomes() {
  showToast('Checking all outcomes...');
  try {
    const r = await fetch(`${backendUrl}/history/check-all`, { signal: AbortSignal.timeout(30000) });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const result = await r.json();
    if (result.resolved > 0) {
      // Update local history with resolved outcomes
      const history = getHistory();
      for (const item of result.items) {
        const idx = history.findIndex(h => h.id === item.id);
        if (idx >= 0) {
          history[idx].feedback = item.feedback;
          history[idx].outcome_price = item.price;
        }
      }
      saveHistory(history);
      renderHistory();
      showToast(`Resolved ${result.resolved} predictions`);
    } else {
      showToast('No expired predictions to resolve');
    }
  } catch (e) {
    showToast(`Error: ${e.message}`);
  }
}

async function retrainML() {
  showToast('Retraining ML model...');
  try {
    const r = await fetch(`${backendUrl}/ml/retrain`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: AbortSignal.timeout(60000)
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const result = await r.json();
    if (result.ok) {
      showToast(`ML retrained: ${result.samples} samples, accuracy: ${(result.accuracy*100).toFixed(0)}%`);
    } else {
      showToast(`ML retrain failed: ${result.reason || 'unknown error'}`);
    }
  } catch (e) {
    showToast(`Error: ${e.message}`);
  }
}

let _toastTimer;
function showToast(msg) {
  const t=document.getElementById('toast');
  if (!t) return;
  t.textContent=msg; t.classList.add('on');
  clearTimeout(_toastTimer);
  _toastTimer=setTimeout(()=>t.classList.remove('on'),3500);
}
