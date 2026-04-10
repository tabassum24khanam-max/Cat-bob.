// ULTRAMAX Cloudflare Worker v3.0
// Proxies: Yahoo Finance, macro data, news aggregation
// Deploy: wrangler publish

const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type,Authorization',
};

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { ...CORS, 'Content-Type': 'application/json' }
  });
}

function err(msg, status = 400) {
  return json({ error: msg }, status);
}

async function yahooFetch(symbol, range, interval) {
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?range=${range}&interval=${interval}`;
  const r = await fetch(url, {
    headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36' }
  });
  if (!r.ok) throw new Error(`Yahoo ${r.status}`);
  return r.json();
}

async function getCandles(sym, iv = '1h') {
  const rangeMap = {
    '15m': '5d', '30m': '10d', '1h': '60d',
    '4h': '3mo', '1d': '2y', '1wk': '5y'
  };
  const range = rangeMap[iv] || '60d';
  const data = await yahooFetch(sym, range, iv);
  const res = data?.chart?.result?.[0];
  if (!res) throw new Error('No data');
  const { timestamp: ts = [], indicators: { quote: [q = {}] = [] } = {} } = res;
  const candles = [];
  for (let i = 0; i < ts.length; i++) {
    if (q.close?.[i] != null && q.open?.[i] != null) {
      candles.push({
        time: ts[i],
        open: q.open[i],
        high: q.high[i],
        low: q.low[i],
        close: q.close[i],
        volume: q.volume?.[i] ?? 0,
      });
    }
  }
  return candles;
}

async function getPrice(sym) {
  const data = await yahooFetch(sym, '1d', '1d');
  const res = data?.chart?.result?.[0];
  if (!res) throw new Error('No data');
  const price = res.meta?.regularMarketPrice;
  const prevClose = res.meta?.chartPreviousClose || res.meta?.previousClose;
  const chg = prevClose && price ? ((price - prevClose) / prevClose * 100) : null;
  return { price, chg };
}

async function getMacro() {
  const symbols = {
    vix: '^VIX', dxy: 'DX-Y.NYB', spy: 'SPY',
    tnx: '^TNX', gold: 'GC=F', oil: 'CL=F',
  };

  const results = await Promise.allSettled(
    Object.entries(symbols).map(([key, sym]) =>
      getPrice(sym).then(d => [key, d])
    )
  );

  const macro = {};
  for (const r of results) {
    if (r.status === 'fulfilled') {
      const [key, d] = r.value;
      macro[key] = d.price;
      macro[`${key}Chg`] = d.chg;
    }
  }

  // Fed rate approximation from SOFR (approximated)
  macro.fedRate = 5.33;

  return macro;
}

async function getNews(sym, name) {
  const encoded = encodeURIComponent(`${name} ${sym} price market`);
  const url = `https://news.google.com/rss/search?q=${encoded}&hl=en-US&gl=US&ceid=US:en`;
  const r = await fetch(url);
  const text = await r.text();

  // Parse RSS titles
  const headlines = [];
  const matches = text.matchAll(/<title><!\[CDATA\[(.*?)\]\]><\/title>/g);
  for (const m of matches) {
    const h = m[1].trim();
    if (h.length > 15 && !h.includes('Google News')) {
      headlines.push(h);
    }
    if (headlines.length >= 15) break;
  }
  return headlines;
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname;

    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: CORS });
    }

    try {
      // ── /candles ──────────────────────────────────────────────────────
      if (path === '/candles') {
        const sym = url.searchParams.get('sym');
        const iv  = url.searchParams.get('iv') || '1h';
        if (!sym) return err('sym required');
        const candles = await getCandles(sym, iv);
        return json({ candles, sym, iv, count: candles.length });
      }

      // ── /price ────────────────────────────────────────────────────────
      if (path === '/price') {
        const sym = url.searchParams.get('sym');
        if (!sym) return err('sym required');
        const d = await getPrice(sym);
        return json({ ...d, sym });
      }

      // ── /macro ────────────────────────────────────────────────────────
      if (path === '/macro') {
        const macro = await getMacro();
        return json(macro);
      }

      // ── /news ─────────────────────────────────────────────────────────
      if (path === '/news') {
        const sym  = url.searchParams.get('sym') || 'BTC';
        const name = url.searchParams.get('name') || sym;
        const headlines = await getNews(sym, name);
        return json({ headlines, sym });
      }

      // ── /fg (Fear & Greed) ────────────────────────────────────────────
      if (path === '/fg') {
        const r = await fetch('https://api.alternative.me/fng/?limit=2');
        const d = await r.json();
        return json(d);
      }

      // ── /health ───────────────────────────────────────────────────────
      if (path === '/health') {
        return json({ ok: true, version: '3.0', ts: Date.now() });
      }

      return err('Unknown route', 404);

    } catch(e) {
      return json({ error: e.message }, 500);
    }
  }
};
