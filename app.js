// ULTRAMAX Frontend v3.0

let backendUrl = localStorage.getItem('um_backend') || 'http://localhost:8000';
let apiKey  = localStorage.getItem('um_key')   || '';
let dsKey   = localStorage.getItem('um_dskey') || '';
let asset   = 'BTC';
let horizon = 4;
let chartInst = null, mainSeries = null, predSeries = null;
let currentTab = 'crypto';
let lastResult = null;
let _predicting = false;

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
  if (!apiKey) openModal();
  else { setDot('idle','READY'); checkBackend(); }
  setInterval(checkBackend, 30000);
  setInterval(tickPrice, 15000);
  loadChart();
  retroactivelyScoreHistory();
});

async function checkBackend() {
  try {
    const r = await fetch(`${backendUrl}/health`, { signal: AbortSignal.timeout(5000) });
    const d = await r.json();
    document.getElementById('backend-dot').className = 'backend-dot ok';
    document.getElementById('backend-txt').textContent = `Backend v${d.version} · ${backendUrl}`;
    setDot('on', dsKey ? 'READY — R1 + V3.2' : 'READY');
    return true;
  } catch {
    document.getElementById('backend-dot').className = 'backend-dot err';
    document.getElementById('backend-txt').textContent = 'Backend offline — run ./start.sh';
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
    const r = await fetch(`${backendUrl}/price?asset=${asset}`, { signal: AbortSignal.timeout(5000) });
    if (!r.ok) return;
    const d = await r.json();
    const el = document.getElementById('plive'), ce = document.getElementById('pchg');
    if (el && d.price) el.textContent = fmtP(d.price);
    if (ce && d.chg != null) { ce.textContent = fmtPct(d.chg); ce.className = `price-chg ${d.chg >= 0 ? 'up' : 'dn'}`; }
  } catch {}
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
  loadChart();
}

function selA(a) { asset = a; renderAssets(); closeResult(); loadChart(); }

function setHz(h, btn) {
  horizon = h;
  document.querySelectorAll('.hz-btn').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
}

async function loadChart() {
  const container = document.getElementById('chart');
  if (!container) return;
  if (chartInst) { try { chartInst.remove(); } catch {} chartInst = mainSeries = predSeries = null; }
  try {
    const r = await fetch(`${backendUrl}/candles?asset=${asset}&interval=1h&limit=120`, { signal: AbortSignal.timeout(8000) });
    if (!r.ok) return;
    const d = await r.json();
    if (!d.candles?.length) return;

    chartInst = LightweightCharts.createChart(container, {
      width: container.clientWidth, height: container.clientHeight || 300,
      layout: { background: { color: 'transparent' }, textColor: '#2d4d6e' },
      grid: { vertLines: { color: 'rgba(20,38,61,.4)' }, horzLines: { color: 'rgba(20,38,61,.4)' } },
      rightPriceScale: { borderColor: 'rgba(20,38,61,.5)' },
      timeScale: { borderColor: 'rgba(20,38,61,.5)', timeVisible: true },
      handleScroll: true, handleScale: true,
    });

    mainSeries = chartInst.addCandlestickSeries({
      upColor:'#00e676', downColor:'#ff1744',
      borderUpColor:'#00e676', borderDownColor:'#ff1744',
      wickUpColor:'#00e676', wickDownColor:'#ff1744',
    });

    const valid = d.candles.filter(c => c.close && c.open);
    mainSeries.setData(valid.map(c => ({ time:c.time, open:c.open, high:c.high, low:c.low, close:c.close })));

    const last = valid[valid.length-1], prev = valid[valid.length-2];
    if (last) {
      document.getElementById('plive').textContent = fmtP(last.close);
      if (prev) {
        const chg = (last.close-prev.close)/prev.close*100;
        const ce = document.getElementById('pchg');
        if (ce) { ce.textContent = fmtPct(chg); ce.className = `price-chg ${chg>=0?'up':'dn'}`; }
      }
    }
    chartInst.timeScale().fitContent();
  } catch {}
}

window.addEventListener('resize', () => {
  if (chartInst) {
    const c = document.getElementById('chart');
    if (c) chartInst.applyOptions({ width:c.clientWidth, height:c.clientHeight });
  }
});

async function predict() {
  if (_predicting) return;
  if (!apiKey) { openModal(); return; }
  const ok = await checkBackend();
  if (!ok) { showToast('⚠ Backend offline — run ./start.sh'); return; }

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
  const t3 = setTimeout(() => { setAgentStep(2,'done'); setAgentStep(3,'running'); document.getElementById('lstep').textContent = `🧠 ${dsKey?'DeepSeek R1':'GPT-4o'} deciding...`; }, 20000);

  try {
    const r = await fetch(`${backendUrl}/predict`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ asset, horizon, api_key:apiKey, ds_key:dsKey||null }),
      signal: AbortSignal.timeout(680000)
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

  const modelMap = {'deepseek-r1':'🧠 R1','deepseek-chat':'🤖 V3','gpt-4o':'🎯 GPT-4o','error':'❌ Error'};
  let confLine = modelMap[r.agent_model] || r.agent_model || '—';
  if (r.raw_ai_conf) confLine += ` · AI:${r.raw_ai_conf}%`;
  if (r.bayesian_conf) confLine += ` B:${r.bayesian_conf?.toFixed(0)}%`;
  if (r.ml?.available) confLine += ` ML:${r.ml.confidence?.toFixed(0)}%`;
  set('rp-model', confLine);

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
    if (!e.outcome_price||!e.entry_price||!e.original_decision) return e;
    const moved = e.outcome_price - e.entry_price;
    const fb = e.original_decision==='BUY' ? (moved<0?'correct':'wrong')
              : e.original_decision==='SELL' ? (moved>0?'correct':'wrong') : null;
    if (!fb) return e;
    changed = true;
    return {...e, feedback:fb};
  });
  if (changed) saveHistory(updated);
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
      const orig=e.original_decision;
      const wouldHaveLost=orig==='BUY'?(pct||0)<-0.3:orig==='SELL'?(pct||0)>0.3:false;
      const tag=wouldHaveLost?' · ✓ SMART AVOID':Math.abs(pct||0)>0.5?' · ⚠ MISSED MOVE':'';
      fbHtml=`<div class="he-fb pending">⏸ NO TRADE${pct!=null?` · ${fmtPct(pct)}${tag}`:''}</div>`;
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
  if (e.feedback==='correct') verdictHtml=`<div style="color:var(--green);font-size:22px;font-family:'Bebas Neue'">✓ CORRECT DIRECTION</div>`;
  else if (e.feedback==='wrong') verdictHtml=`<div style="color:var(--red);font-size:22px;font-family:'Bebas Neue'">✗ WRONG DIRECTION</div>`;
  else if (e.feedback==='skipped'&&e.outcome_price) {
    const orig=e.original_decision;
    const wl=orig==='BUY'?pct<-0.3:orig==='SELL'?pct>0.3:false;
    const ww=orig==='BUY'?pct>0.3:orig==='SELL'?pct<-0.3:false;
    verdictHtml=wl?`<div style="color:var(--green);font-size:18px;font-family:'Bebas Neue'">✓ SMART AVOID · ${fmtPct(pct)}</div>`
                :ww?`<div style="color:var(--yellow);font-size:18px;font-family:'Bebas Neue'">⚠ MISSED MOVE · ${fmtPct(pct)}</div>`
                :`<div style="color:var(--muted);font-size:16px;font-family:'Bebas Neue'">⏸ NO TRADE · Moved ${fmtPct(pct)}</div>`;
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

  document.body.insertAdjacentHTML('beforeend',`
    <div onclick="this.remove()" style="position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,.88);z-index:500;overflow-y:auto;padding:16px">
      <div onclick="event.stopPropagation()" style="background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:16px;max-width:420px;margin:0 auto">
        <div style="display:flex;justify-content:space-between;margin-bottom:12px">
          <div style="font-family:'Bebas Neue';font-size:18px;color:var(--cyan)">${e.asset} · ${e.decision.replace('_',' ')} · ${e.horizon_label||e.horizon+'H'}</div>
          <button onclick="this.closest('[onclick]').click()" style="background:none;border:none;color:var(--muted);font-size:20px;cursor:pointer">✕</button>
        </div>
        <div style="margin-bottom:10px">${verdictHtml}</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--border);margin-bottom:10px">
          ${rows.map(([l,v])=>`<div style="background:var(--surface);padding:7px 10px"><div style="font-size:7px;color:var(--muted)">${l}</div><div style="font-size:10px">${v}</div></div>`).join('')}
        </div>
        ${targetAccHtml?`<div style="font-size:8px;color:var(--cyan);padding:6px 10px;background:rgba(0,229,255,.05);border-radius:4px;margin-bottom:8px">${targetAccHtml}</div>`:''}
        ${e.gate_reason?`<div style="font-size:8px;color:var(--red);padding:4px 10px;margin-bottom:8px">${e.gate_reason}</div>`:''}
        ${e.primary_reason?`<div style="font-size:8px;color:var(--text);padding:4px 10px;margin-bottom:6px"><strong>Key factor:</strong> ${e.primary_reason}</div>`:''}
        ${e.insight?`<div style="font-size:8px;color:var(--muted);line-height:1.6;padding:6px 10px;margin-bottom:8px"><strong style="color:var(--cyan)">R1 Reasoning:</strong><br>${e.insight}</div>`:''}
        ${expired&&!e.outcome_price?`<button onclick="event.stopPropagation();checkOutcome('${e.id}');this.closest('[onclick]').click();" class="check-btn" style="width:100%;padding:9px;font-size:10px;margin-top:4px">🔄 GET PRICE AT EXPIRY</button>`:''}
      </div>
    </div>`);
}

async function checkOutcome(id) {
  const history=getHistory(), entry=history.find(e=>e.id===id);
  if (!entry) return;
  showToast('🔄 Fetching outcome...');
  try {
    const r=await fetch(`${backendUrl}/history/outcome`,{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({pred_id:id,asset:entry.asset,entry_price:entry.entry_price,target_price:entry.target_price,original_decision:entry.original_decision,horizon:entry.horizon})
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const result=await r.json();
    let feedback=result.feedback;
    if (feedback==='skipped'&&entry.original_decision) {
      const moved=result.price-entry.entry_price;
      feedback=entry.original_decision==='BUY'?(moved<0?'correct':'wrong'):entry.original_decision==='SELL'?(moved>0?'correct':'wrong'):'skipped';
    }
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
}

function saveKeys() {
  const bk=document.getElementById('k-backend')?.value.trim();
  const oai=document.getElementById('k-openai')?.value.trim();
  const ds=document.getElementById('k-deepseek')?.value.trim();
  if (!oai||oai.length<10) { showToast('⚠ OpenAI key required'); return; }
  backendUrl=bk||'http://localhost:8000'; apiKey=oai; dsKey=ds||'';
  localStorage.setItem('um_backend',backendUrl);
  localStorage.setItem('um_key',apiKey);
  localStorage.setItem('um_dskey',dsKey);
  document.getElementById('kmodal')?.classList.remove('on');
  checkBackend();
  showToast('✓ Keys saved');
}

let _toastTimer;
function showToast(msg) {
  const t=document.getElementById('toast');
  if (!t) return;
  t.textContent=msg; t.classList.add('on');
  clearTimeout(_toastTimer);
  _toastTimer=setTimeout(()=>t.classList.remove('on'),3500);
}
