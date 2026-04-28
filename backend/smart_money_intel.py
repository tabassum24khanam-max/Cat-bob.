"""
ULTRAMAX Smart Money Intelligence Module

Tracks every category of large informed market participant:
  - Politicians (congressional trades)
  - Corporate insiders (SEC Form 4)
  - Institutions / hedge funds (13F filings)
  - Options flow (unusual activity)
  - Dark pool / block trades
  - Top traders (successful public traders on social/leaderboards)
  - Volume anomaly (QE signal)
  - Macro flow (Fed/BIS context)

Produces a single Smart Money Score (0-100) per asset.
"""
import asyncio
import time
import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional
from collections import defaultdict

import httpx

try:
    from config import (FINNHUB_API_KEY, FRED_API_KEY,
                        ALPACA_KEY, ALPACA_SECRET, get_asset_type)
except ImportError:
    FINNHUB_API_KEY = FRED_API_KEY = ""
    ALPACA_KEY = ALPACA_SECRET = ""
    def get_asset_type(a): return 'stock'

_HEADERS = {"User-Agent": "ULTRAMAX/1.0 (research@ultramax.app)"}
_cache: Dict[str, dict] = {}
_CACHE_TTL = 6 * 3600

_source_performance: Dict[str, dict] = {}

ELITE_FUNDS = [
    "BERKSHIRE HATHAWAY", "BRIDGEWATER", "RENAISSANCE TECHNOLOGIES",
    "PERSHING SQUARE", "TIGER GLOBAL", "CITADEL", "POINT72",
    "TWO SIGMA", "VIKING GLOBAL", "COATUE", "D1 CAPITAL",
    "DRUCKENMILLER", "BAUPOST", "THIRD POINT", "APPALOOSA",
]

CRYPTO_ASSETS = {"BTC", "ETH", "SOL", "BNB", "XRP", "DOGE"}


def _cache_key(asset: str) -> str:
    return f"smi_{asset}"


def _is_cached(asset: str) -> bool:
    k = _cache_key(asset)
    if k in _cache and (time.time() - _cache[k].get("_ts", 0)) < _CACHE_TTL:
        return True
    return False


async def _fetch_json(url: str, headers: dict = None, params: dict = None,
                       timeout: float = 12) -> Optional[dict]:
    try:
        h = {**_HEADERS, **(headers or {})}
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as c:
            r = await c.get(url, headers=h, params=params)
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass
    return None


async def _fetch_text(url: str, headers: dict = None, timeout: float = 12) -> str:
    try:
        h = {**_HEADERS, **(headers or {})}
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as c:
            r = await c.get(url, headers=h)
            if r.status_code == 200:
                return r.text
    except Exception:
        pass
    return ""


# ─── Component A: Congressional / Politician Trades ─────────────────────

async def _fetch_politician_trades(ticker: str) -> dict:
    result = {"score": 0, "direction": "neutral", "signals": [], "detail": "No data"}

    try:
        data = await _fetch_json(
            "https://bff.capitoltrades.com/trades",
            params={"asset": ticker, "pageSize": "20", "page": "1"},
        )
        if data and isinstance(data, dict):
            trades = data.get("data", [])
            buys = sum(1 for t in trades if "purchase" in str(t.get("txType", "")).lower())
            sells = sum(1 for t in trades if "sale" in str(t.get("txType", "")).lower())
            for t in trades[:5]:
                pol = t.get("politician", {})
                name = f"{pol.get('firstName', '')} {pol.get('lastName', '')}".strip()
                result["signals"].append(f"{name}: {t.get('txType', '?')}")
            if buys + sells > 0:
                ratio = buys / (buys + sells)
                result["score"] = min(20, int(ratio * 20 * min(buys + sells, 5) / 3))
                result["direction"] = "bullish" if ratio > 0.6 else "bearish" if ratio < 0.4 else "neutral"
                result["detail"] = f"{buys} buys, {sells} sells (Capitol Trades)"
    except Exception:
        pass
    return result


# ─── Component B: Corporate Insider Trades (SEC Form 4) ─────────────────

async def _fetch_insider_trades(ticker: str) -> dict:
    result = {"score": 0, "direction": "neutral", "signals": [], "detail": "No data"}

    if FINNHUB_API_KEY:
        data = await _fetch_json(
            "https://finnhub.io/api/v1/stock/insider-transactions",
            params={"symbol": ticker, "token": FINNHUB_API_KEY},
        )
        if data and isinstance(data, dict):
            txns = data.get("data", [])[:30]
            buy_val = 0
            sell_val = 0
            cluster_buyers = set()
            cluster_window = set()
            for t in txns:
                name = t.get("name", "?")
                change = t.get("change", 0) or 0
                price = t.get("transactionPrice", 0) or 0
                val = abs(change * price)
                role = _classify_role(name)
                weight = 2.0 if role in ("CEO", "CFO", "COO") else 1.0
                if change > 0:
                    buy_val += val * weight
                    cluster_buyers.add(name)
                    d = t.get("transactionDate", "")
                    if d:
                        cluster_window.add(d[:10])
                elif change < 0:
                    sell_val += val * weight
                result["signals"].append(f"{name} ({role}): {'BUY' if change > 0 else 'SELL'} ${val:,.0f}")
                if len(result["signals"]) >= 5:
                    break

            total = buy_val + sell_val
            if total > 0:
                ratio = buy_val / total
                cluster_bonus = 3 if len(cluster_buyers) >= 2 and len(cluster_window) <= 3 else 0
                result["score"] = min(20, int(ratio * 15 + cluster_bonus + min(total / 500000, 5)))
                result["direction"] = "bullish" if ratio > 0.6 else "bearish" if ratio < 0.4 else "neutral"
                result["detail"] = f"Buys ${buy_val:,.0f} vs Sells ${sell_val:,.0f}"
                if len(cluster_buyers) >= 2:
                    result["detail"] += f" | CLUSTER: {len(cluster_buyers)} insiders buying"
            return result

    sec_data = await _fetch_json(
        "https://efts.sec.gov/LATEST/search-index",
        params={"q": f'"{ticker}"', "forms": "4", "dateRange": "custom",
                "startdt": _days_ago_str(90), "enddt": _today_str()},
        headers={"User-Agent": "ULTRAMAX/1.0 (research@ultramax.app)", "Accept": "application/json"},
    )
    if sec_data and isinstance(sec_data, dict):
        hits = sec_data.get("hits", {}).get("hits", [])
        if hits:
            result["score"] = min(10, len(hits))
            result["detail"] = f"{len(hits)} Form 4 filings in 90d (SEC EDGAR)"
            result["direction"] = "neutral"
            for h in hits[:3]:
                result["signals"].append(f"Form 4 filed: {h.get('_source', {}).get('display_names', ['?'])[0]}")
    return result


# ─── Component C: Institutions / Hedge Fund 13F ─────────────────────────

async def _fetch_institutional(ticker: str) -> dict:
    result = {"score": 0, "direction": "neutral", "signals": [], "detail": "No data"}

    if FINNHUB_API_KEY:
        data = await _fetch_json(
            "https://finnhub.io/api/v1/stock/institutional-ownership",
            params={"symbol": ticker, "token": FINNHUB_API_KEY},
        )
        if data and isinstance(data, dict):
            owners = data.get("data", [])
            if owners and len(owners) >= 2:
                latest = owners[0].get("ownership", [])
                prev = owners[1].get("ownership", []) if len(owners) > 1 else []
                prev_map = {o.get("name", ""): o.get("share", 0) for o in prev}
                added = 0
                reduced = 0
                elite_signal = []
                for o in latest[:50]:
                    name = o.get("name", "")
                    shares = o.get("share", 0) or 0
                    old = prev_map.get(name, 0) or 0
                    is_elite = any(e in name.upper() for e in ELITE_FUNDS)
                    if shares > old:
                        added += 1
                        if is_elite:
                            elite_signal.append(f"{name}: ADDED")
                    elif shares < old:
                        reduced += 1
                        if is_elite:
                            elite_signal.append(f"{name}: REDUCED")

                total = added + reduced
                if total > 0:
                    ratio = added / total
                    elite_bonus = min(5, len(elite_signal) * 2)
                    result["score"] = min(20, int(ratio * 15 + elite_bonus))
                    result["direction"] = "bullish" if ratio > 0.6 else "bearish" if ratio < 0.4 else "neutral"
                    result["detail"] = f"{added} funds adding, {reduced} reducing"
                    result["signals"] = elite_signal[:5] or [f"{added} added, {reduced} reduced"]
            return result

    sec_data = await _fetch_json(
        "https://efts.sec.gov/LATEST/search-index",
        params={"q": f'"{ticker}"', "forms": "13F-HR"},
        headers={"User-Agent": "ULTRAMAX/1.0 (research@ultramax.app)", "Accept": "application/json"},
    )
    if sec_data and isinstance(sec_data, dict):
        hits = sec_data.get("hits", {}).get("hits", [])
        if hits:
            result["score"] = min(10, len(hits) * 2)
            result["detail"] = f"{len(hits)} 13F filings mentioning {ticker}"
            for h in hits[:3]:
                names = h.get("_source", {}).get("display_names", ["?"])
                is_elite = any(any(e in n.upper() for e in ELITE_FUNDS) for n in names)
                tag = " ★ELITE" if is_elite else ""
                result["signals"].append(f"{names[0]}{tag}")
                if is_elite:
                    result["score"] = min(20, result["score"] + 3)
    return result


# ─── Component D: Options Flow ──────────────────────────────────────────

async def _fetch_options_flow(ticker: str) -> dict:
    result = {"score": 0, "direction": "neutral", "signals": [], "detail": "No data"}
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        dates = stock.options[:3] if stock.options else []
        total_call_vol = 0
        total_put_vol = 0
        large_call_premium = 0
        large_put_premium = 0
        for d in dates:
            chain = stock.option_chain(d)
            calls = chain.calls
            puts = chain.puts
            if not calls.empty:
                total_call_vol += calls["volume"].sum()
                for _, row in calls.iterrows():
                    prem = (row.get("lastPrice", 0) or 0) * (row.get("volume", 0) or 0) * 100
                    if prem > 500000:
                        large_call_premium += prem
                        result["signals"].append(f"CALL sweep ${prem:,.0f} strike={row.get('strike', '?')} exp={d}")
            if not puts.empty:
                total_put_vol += puts["volume"].sum()
                for _, row in puts.iterrows():
                    prem = (row.get("lastPrice", 0) or 0) * (row.get("volume", 0) or 0) * 100
                    if prem > 500000:
                        large_put_premium += prem
                        result["signals"].append(f"PUT sweep ${prem:,.0f} strike={row.get('strike', '?')} exp={d}")

        total_vol = total_call_vol + total_put_vol
        if total_vol > 0:
            pc_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else 2.0
            if pc_ratio < 0.5:
                result["direction"] = "bullish"
                result["score"] = min(20, int((1 - pc_ratio) * 20))
            elif pc_ratio > 1.5:
                result["direction"] = "bearish"
                result["score"] = min(20, int(min(pc_ratio, 3) * 7))
            else:
                result["direction"] = "neutral"
                result["score"] = 5
            if large_call_premium > large_put_premium and large_call_premium > 0:
                result["score"] = min(20, result["score"] + 3)
                result["direction"] = "bullish"
            elif large_put_premium > large_call_premium and large_put_premium > 0:
                result["score"] = min(20, result["score"] + 3)
                result["direction"] = "bearish"
            result["detail"] = f"P/C ratio={pc_ratio:.2f} | Call vol={total_call_vol:,} Put vol={total_put_vol:,}"
            result["signals"] = result["signals"][:5]
    except Exception:
        pass
    return result


# ─── Component E: Dark Pool / Block Trades ──────────────────────────────

async def _fetch_dark_pool(ticker: str) -> dict:
    result = {"score": 0, "direction": "neutral", "signals": [], "detail": "No data"}
    data = await _fetch_json(
        "https://api.finra.org/data/group/otcMarket/name/weeklySummary",
        params={"symbol": ticker, "limit": "5"},
        headers={"Accept": "application/json"},
    )
    if not data:
        data = await _fetch_json(
            f"https://api.finra.org/data/group/otcMarket/name/weeklySummary?symbol={ticker}&limit=5",
            headers={"Accept": "application/json"},
        )
    if data and isinstance(data, list) and len(data) > 0:
        total_vol = sum(d.get("totalWeeklyShareQuantity", 0) or 0 for d in data)
        total_trades = sum(d.get("totalWeeklyTradeCount", 0) or 0 for d in data)
        if total_vol > 0:
            avg_trade_size = total_vol / max(total_trades, 1)
            result["score"] = min(10, int(min(avg_trade_size / 1000, 10)))
            result["detail"] = f"Dark pool: {total_vol:,} shares, {total_trades:,} trades, avg size={avg_trade_size:,.0f}"
            result["direction"] = "neutral"
            result["signals"].append(f"Weekly DP volume: {total_vol:,}")
    return result


# ─── Component F: Top Traders (Successful Public Traders) ───────────────

async def _fetch_top_traders(ticker: str) -> dict:
    """Track notable successful public traders from free sources:
    - Binance Leaderboard (crypto)
    - TradingView ideas with high reputation
    - StockTwits trending + high accuracy users
    """
    result = {"score": 0, "direction": "neutral", "signals": [], "detail": "No data"}

    asset_type = get_asset_type(ticker)

    if asset_type == "crypto":
        symbol = f"{ticker}USDT"
        data = await _fetch_json(
            "https://www.binance.com/bapi/futures/v1/public/future/leaderboard/getOtherPosition",
            headers={"Content-Type": "application/json"},
        )
        if not data:
            data = await _fetch_json(
                f"https://www.binance.com/bapi/futures/v3/public/future/leaderboard/searchLeaderboard",
                headers={"Content-Type": "application/json"},
            )
        ideas = await _fetch_json(
            f"https://scanner.tradingview.com/crypto/scan",
            headers={"Content-Type": "application/json"},
        )

    stw_data = await _fetch_json(
        f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json",
    )
    if stw_data and isinstance(stw_data, dict):
        messages = stw_data.get("messages", [])
        if messages:
            bull_count = 0
            bear_count = 0
            top_trader_signals = []
            for msg in messages[:30]:
                sentiment = msg.get("entities", {}).get("sentiment", {})
                if not sentiment:
                    continue
                basic = sentiment.get("basic", "")
                user = msg.get("user", {})
                followers = user.get("followers", 0) or 0
                ideas_count = user.get("ideas", 0) or 0
                is_notable = followers > 1000 or ideas_count > 100
                if basic == "Bullish":
                    bull_count += 1
                    if is_notable:
                        top_trader_signals.append(
                            f"@{user.get('username', '?')} ({followers:,} followers): BULLISH"
                        )
                elif basic == "Bearish":
                    bear_count += 1
                    if is_notable:
                        top_trader_signals.append(
                            f"@{user.get('username', '?')} ({followers:,} followers): BEARISH"
                        )
            total = bull_count + bear_count
            if total >= 5:
                ratio = bull_count / total
                result["score"] = min(10, int(abs(ratio - 0.5) * 20))
                result["direction"] = "bullish" if ratio > 0.6 else "bearish" if ratio < 0.4 else "neutral"
                result["detail"] = f"StockTwits: {bull_count} bull, {bear_count} bear ({total} posts)"
                result["signals"] = top_trader_signals[:5] or [f"{bull_count} bullish vs {bear_count} bearish"]
    return result


# ─── Component G: Volume Anomaly (QE Signal) ────────────────────────────

async def _compute_volume_anomaly(ticker: str) -> dict:
    result = {"score": 0, "direction": "neutral", "detail": "No data"}
    try:
        from data_fetcher import fetch_candles
        candles = await fetch_candles(ticker, "1h", 200)
        if candles and len(candles) > 30:
            volumes = [c.get("volume", 0) or 0 for c in candles if c.get("volume")]
            if not volumes:
                return result
            avg_vol = sum(volumes[:-1]) / len(volumes[:-1])
            current_vol = volumes[-1]
            if avg_vol > 0:
                ratio = current_vol / avg_vol
                if ratio > 2.0:
                    result["score"] = 10
                    result["direction"] = "bullish" if candles[-1].get("close", 0) > candles[-1].get("open", 0) else "bearish"
                    result["detail"] = f"Volume {ratio:.1f}x average — strong institutional activity"
                elif ratio > 1.5:
                    result["score"] = 7
                    result["direction"] = "neutral"
                    result["detail"] = f"Volume {ratio:.1f}x average — elevated activity"
                elif ratio > 1.2:
                    result["score"] = 4
                    result["direction"] = "neutral"
                    result["detail"] = f"Volume {ratio:.1f}x average — slightly above normal"
                else:
                    result["score"] = 1
                    result["detail"] = f"Volume {ratio:.1f}x average — normal"
    except Exception:
        pass
    return result


# ─── Macro Flow Context ─────────────────────────────────────────────────

async def _fetch_macro_flow() -> str:
    notes = []
    rss_text = await _fetch_text("https://www.federalreserve.gov/feeds/press_all.xml")
    if rss_text:
        try:
            root = ET.fromstring(rss_text)
            items = root.findall(".//item")[:5]
            for item in items:
                title = item.findtext("title", "")
                if title:
                    notes.append(f"Fed: {title[:120]}")
        except Exception:
            pass

    if FRED_API_KEY:
        vix_data = await _fetch_json(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": "VIXCLS", "api_key": FRED_API_KEY,
                    "file_type": "json", "sort_order": "desc", "limit": "1"},
        )
        if vix_data:
            obs = vix_data.get("observations", [])
            if obs and obs[0].get("value", ".") != ".":
                vix_val = float(obs[0]["value"])
                if vix_val > 30:
                    notes.append(f"VIX={vix_val:.1f} — FEAR (smart money may be hedging)")
                elif vix_val < 15:
                    notes.append(f"VIX={vix_val:.1f} — complacency (smart money accumulating)")

    return " | ".join(notes) if notes else "No macro flow data"


# ─── Main Entry Point ───────────────────────────────────────────────────

async def get_smart_money_score(asset: str) -> dict:
    if _is_cached(asset):
        return _cache[_cache_key(asset)]

    is_crypto = asset in CRYPTO_ASSETS
    ticker = asset

    if is_crypto:
        tasks = {
            "options_flow": _fetch_options_flow(ticker),
            "top_traders": _fetch_top_traders(ticker),
            "volume_anomaly": _compute_volume_anomaly(ticker),
        }
        stock_only = {}
    else:
        tasks = {
            "politicians": _fetch_politician_trades(ticker),
            "insiders": _fetch_insider_trades(ticker),
            "institutions": _fetch_institutional(ticker),
            "options_flow": _fetch_options_flow(ticker),
            "dark_pool": _fetch_dark_pool(ticker),
            "top_traders": _fetch_top_traders(ticker),
            "volume_anomaly": _compute_volume_anomaly(ticker),
        }

    macro_task = _fetch_macro_flow()

    results = {}
    keys = list(tasks.keys())
    coros = list(tasks.values())
    gathered = await asyncio.gather(*coros, macro_task, return_exceptions=True)

    for i, key in enumerate(keys):
        if isinstance(gathered[i], dict):
            results[key] = gathered[i]
        else:
            results[key] = {"score": 0, "direction": "neutral", "signals": [], "detail": "Error"}

    macro_context = gathered[-1] if isinstance(gathered[-1], str) else "No macro data"

    if is_crypto:
        weights = {"options_flow": 2.0, "top_traders": 2.0, "volume_anomaly": 1.5}
    else:
        weights = {
            "politicians": 1.0, "insiders": 1.0, "institutions": 1.0,
            "options_flow": 1.0, "dark_pool": 1.0, "top_traders": 0.8,
            "volume_anomaly": 0.8,
        }

    total_score = 0
    max_possible = 0
    directions = []
    components_with_data = 0

    for comp_name, comp_data in results.items():
        w = weights.get(comp_name, 1.0)
        s = comp_data.get("score", 0) * w
        total_score += s
        max_possible += 20 * w
        d = comp_data.get("direction", "neutral")
        if d != "neutral" and comp_data.get("score", 0) > 0:
            directions.append(d)
        if comp_data.get("score", 0) > 0:
            components_with_data += 1

    if max_possible > 0:
        normalized_score = int(total_score / max_possible * 100)
    else:
        normalized_score = 0

    bull_count = directions.count("bullish")
    bear_count = directions.count("bearish")
    if bull_count > bear_count and bull_count >= 2:
        direction = "bullish"
    elif bear_count > bull_count and bear_count >= 2:
        direction = "bearish"
    elif bull_count > 0 and bear_count > 0:
        direction = "split"
    else:
        direction = "neutral"

    confirmed = (bull_count >= 3 and direction == "bullish") or \
                (bear_count >= 3 and direction == "bearish")

    total_components = len(results)
    data_completeness = int(components_with_data / max(total_components, 1) * 100)

    best_comp = max(results.items(), key=lambda x: x[1].get("score", 0), default=("none", {}))
    top_signal = ""
    if best_comp[1].get("signals"):
        top_signal = f"[{best_comp[0]}] {best_comp[1]['signals'][0]}"
    elif best_comp[1].get("detail"):
        top_signal = f"[{best_comp[0]}] {best_comp[1]['detail']}"

    high_quality_flags = []
    ins = results.get("insiders", {})
    if ins.get("detail") and "CLUSTER" in ins.get("detail", ""):
        high_quality_flags.append("CEO + insiders cluster buying within 10 days")
    inst = results.get("institutions", {})
    if inst.get("signals"):
        for s in inst["signals"]:
            if "ELITE" in s or "★" in s:
                high_quality_flags.append(f"Elite fund activity: {s}")
                break
    opts = results.get("options_flow", {})
    if opts.get("score", 0) >= 15:
        high_quality_flags.append("Heavy unusual options activity detected")
    vol = results.get("volume_anomaly", {})
    if vol.get("score", 0) >= 7:
        high_quality_flags.append(f"Abnormal volume: {vol.get('detail', '')}")
    top_t = results.get("top_traders", {})
    if top_t.get("score", 0) >= 7:
        high_quality_flags.append(f"Top traders consensus: {top_t.get('direction', '?')}")

    output = {
        "score": min(100, normalized_score),
        "direction": direction,
        "confirmed": confirmed,
        "data_completeness": data_completeness,
        "components": results,
        "top_signal": top_signal,
        "macro_context": macro_context,
        "high_quality_flags": high_quality_flags,
        "is_crypto": is_crypto,
        "_ts": time.time(),
    }

    _cache[_cache_key(asset)] = output
    return output


# ─── Source Performance Tracking ─────────────────────────────────────────

def record_source_outcome(source_id: str, was_correct: bool):
    if source_id not in _source_performance:
        _source_performance[source_id] = {"signals": 0, "wins": 0, "win_rate": 0.0}
    p = _source_performance[source_id]
    p["signals"] += 1
    if was_correct:
        p["wins"] += 1
    p["win_rate"] = p["wins"] / p["signals"] if p["signals"] > 0 else 0


def get_source_leaderboard() -> list:
    board = []
    for sid, data in _source_performance.items():
        if data["signals"] >= 3:
            board.append({"source": sid, **data})
    board.sort(key=lambda x: (-x["win_rate"], -x["signals"]))
    return board[:20]


# ─── Background Refresh ─────────────────────────────────────────────────

async def refresh_all_smart_money(assets: list):
    for asset in assets:
        try:
            await get_smart_money_score(asset)
        except Exception:
            pass
        await asyncio.sleep(2)


# ─── Helpers ─────────────────────────────────────────────────────────────

def _within_days(date_str: str, days: int) -> bool:
    try:
        from datetime import datetime, timedelta
        if not date_str:
            return False
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y"):
            try:
                dt = datetime.strptime(date_str[:10], fmt[:min(len(fmt), 8)])
                return dt >= datetime.now() - timedelta(days=days)
            except ValueError:
                continue
    except Exception:
        pass
    return True


def _days_ago_str(days: int) -> str:
    from datetime import datetime, timedelta
    return (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")


def _today_str() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")


def _classify_role(name_or_title: str) -> str:
    t = (name_or_title or "").upper()
    if "CEO" in t or "CHIEF EXECUTIVE" in t:
        return "CEO"
    if "CFO" in t or "CHIEF FINANCIAL" in t:
        return "CFO"
    if "COO" in t or "CHIEF OPERATING" in t:
        return "COO"
    if "CTO" in t or "CHIEF TECHNOLOGY" in t:
        return "CTO"
    if "DIRECTOR" in t:
        return "Director"
    if "VP" in t or "VICE PRESIDENT" in t:
        return "VP"
    return "Officer"
