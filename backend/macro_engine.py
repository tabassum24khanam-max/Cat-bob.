"""
ULTRAMAX Macro Engine — Economic Calendar, Event Reaction Matrix, FRED Integration
"""
import asyncio
import time
import httpx
from datetime import datetime, timezone, timedelta
from config import FRED_API_KEY, is_configured
from database import save_macro_event, get_upcoming_events
from data_fetcher import fetch_fred_data


# ─── Historical Event Reaction Matrix ───────────────────────────────────────
# Average % move after major events (based on well-known market patterns)
EVENT_REACTIONS = {
    'FOMC': {
        'description': 'Federal Reserve Interest Rate Decision',
        'BTC':  {'hawkish': -3.2, 'dovish': 4.1, 'neutral': 0.5},
        'ETH':  {'hawkish': -4.0, 'dovish': 5.0, 'neutral': 0.3},
        'SPY':  {'hawkish': -1.5, 'dovish': 1.8, 'neutral': 0.2},
        'GC=F': {'hawkish': -1.0, 'dovish': 2.0, 'neutral': 0.1},
        'CL=F': {'hawkish': -0.8, 'dovish': 1.2, 'neutral': 0.0},
    },
    'CPI': {
        'description': 'Consumer Price Index Report',
        'BTC':  {'hot': -2.5, 'cold': 3.0, 'inline': 0.2},
        'ETH':  {'hot': -3.0, 'cold': 3.5, 'inline': 0.1},
        'SPY':  {'hot': -1.2, 'cold': 1.5, 'inline': 0.1},
        'GC=F': {'hot': 1.5, 'cold': -0.8, 'inline': 0.0},
    },
    'NFP': {
        'description': 'Non-Farm Payrolls',
        'BTC':  {'strong': -1.5, 'weak': 2.0, 'inline': 0.0},
        'SPY':  {'strong': 0.8, 'weak': -1.0, 'inline': 0.1},
        'GC=F': {'strong': -0.5, 'weak': 1.0, 'inline': 0.0},
    },
    'PCE': {
        'description': 'Personal Consumption Expenditures',
        'BTC':  {'hot': -2.0, 'cold': 2.5, 'inline': 0.1},
        'SPY':  {'hot': -0.8, 'cold': 1.0, 'inline': 0.1},
    },
    'OPEC': {
        'description': 'OPEC Production Decision',
        'CL=F': {'cut': 5.0, 'increase': -3.0, 'unchanged': 0.0},
        'BTC':  {'cut': 0.5, 'increase': -0.3, 'unchanged': 0.0},
        'SPY':  {'cut': -0.5, 'increase': 0.3, 'unchanged': 0.0},
    },
    'GDP': {
        'description': 'GDP Growth Rate',
        'SPY':  {'strong': 1.0, 'weak': -1.2, 'inline': 0.1},
        'BTC':  {'strong': 0.5, 'weak': -0.8, 'inline': 0.0},
    },
}

# Known scheduled events (manually updated or scraped)
# Stored as list of {event_type, event_ts, description, impact_level}
# These are seeded on startup and refreshed periodically
KNOWN_RECURRING_EVENTS = [
    # FOMC meetings are roughly every 6 weeks
    # CPI is monthly, usually 2nd week
    # NFP is first Friday of each month
]


async def scrape_forex_factory() -> list:
    """Scrape ForexFactory economic calendar for upcoming events."""
    events = []
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://www.forexfactory.com/calendar",
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'text/html',
                }
            )
            if resp.status_code != 200:
                return events

            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, 'lxml')

            rows = soup.select('tr.calendar__row')
            current_date = None

            for row in rows:
                # Date cell
                date_cell = row.select_one('td.calendar__date')
                if date_cell and date_cell.get_text(strip=True):
                    try:
                        date_text = date_cell.get_text(strip=True)
                        # ForexFactory uses format like "Mon Jan 15"
                        current_date = datetime.strptime(
                            f"{date_text} {datetime.now().year}", "%a%b %d %Y"
                        ).replace(tzinfo=timezone.utc)
                    except Exception:
                        pass

                if not current_date:
                    continue

                # Impact
                impact_cell = row.select_one('td.calendar__impact')
                impact = 'low'
                if impact_cell:
                    icon = impact_cell.select_one('span')
                    if icon:
                        classes = icon.get('class', [])
                        if any('high' in c for c in classes):
                            impact = 'high'
                        elif any('medium' in c for c in classes):
                            impact = 'medium'

                # Currency
                currency_cell = row.select_one('td.calendar__currency')
                currency = currency_cell.get_text(strip=True) if currency_cell else ''

                # Event name
                event_cell = row.select_one('td.calendar__event')
                event_name = event_cell.get_text(strip=True) if event_cell else ''

                if not event_name or impact == 'low':
                    continue

                # Time
                time_cell = row.select_one('td.calendar__time')
                time_text = time_cell.get_text(strip=True) if time_cell else ''

                event_ts = int(current_date.timestamp())
                if time_text and ':' in time_text:
                    try:
                        h, m = time_text.replace('am', '').replace('pm', '').split(':')
                        h = int(h)
                        m = int(m)
                        if 'pm' in time_text.lower() and h != 12:
                            h += 12
                        event_ts = int(current_date.replace(hour=h, minute=m).timestamp())
                    except Exception:
                        pass

                # Map to our event types
                event_type = classify_event(event_name)

                events.append({
                    'event_type': event_type,
                    'event_ts': event_ts,
                    'description': event_name,
                    'impact_level': impact,
                    'currency': currency,
                })

    except Exception:
        pass

    return events


def classify_event(name: str) -> str:
    """Map event name to standard type."""
    name_lower = name.lower()
    if 'fomc' in name_lower or 'federal funds rate' in name_lower or 'fed' in name_lower:
        return 'FOMC'
    if 'cpi' in name_lower or 'consumer price' in name_lower:
        return 'CPI'
    if 'nonfarm' in name_lower or 'non-farm' in name_lower or 'payrolls' in name_lower:
        return 'NFP'
    if 'pce' in name_lower or 'personal consumption' in name_lower:
        return 'PCE'
    if 'gdp' in name_lower:
        return 'GDP'
    if 'opec' in name_lower:
        return 'OPEC'
    if 'employment' in name_lower or 'jobless' in name_lower:
        return 'JOBS'
    if 'retail sales' in name_lower:
        return 'RETAIL'
    if 'pmi' in name_lower or 'manufacturing' in name_lower:
        return 'PMI'
    return 'OTHER'


def get_event_reaction(event_type: str, asset: str) -> dict:
    """Get historical reaction data for an event + asset combo."""
    event_data = EVENT_REACTIONS.get(event_type, {})
    asset_data = event_data.get(asset)
    if not asset_data:
        # Try related assets
        if asset in ['ETH', 'SOL', 'BNB', 'XRP', 'DOGE']:
            asset_data = event_data.get('BTC', {})
        elif asset in ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL']:
            asset_data = event_data.get('SPY', {})
        elif asset in ['SI=F', 'XOM', 'LMT']:
            asset_data = event_data.get('GC=F', {})
    return asset_data or {}


async def refresh_calendar():
    """Scrape and store upcoming events."""
    events = await scrape_forex_factory()
    for event in events:
        # Add historical reactions
        reactions = EVENT_REACTIONS.get(event['event_type'], {})
        event['historical_btc_reaction'] = reactions.get('BTC', {}).get('neutral', 0)
        event['historical_spy_reaction'] = reactions.get('SPY', {}).get('neutral', 0)
        event['historical_gold_reaction'] = reactions.get('GC=F', {}).get('neutral', 0)
        try:
            await save_macro_event(event)
        except Exception:
            pass
    return len(events)


async def get_macro_context(asset: str, horizon_hours: int = 24) -> dict:
    """Get full macro context for prediction: upcoming events + FRED data."""
    upcoming = await get_upcoming_events(max(horizon_hours, 72))

    # Filter to relevant events
    imminent = [e for e in upcoming if e.get('hours_until', 999) <= horizon_hours]
    high_impact = [e for e in imminent if e.get('impact_level') == 'high']

    # Get event reactions for this asset
    event_warnings = []
    for event in high_impact:
        reaction = get_event_reaction(event['event_type'], asset)
        if reaction:
            event_warnings.append({
                'event': event['event_type'],
                'description': event.get('description', ''),
                'hours_until': round(event.get('hours_until', 0), 1),
                'impact': event.get('impact_level'),
                'historical_reactions': reaction,
            })

    # FRED data if available
    fred_data = {}
    if is_configured('FRED_API_KEY'):
        try:
            vix = await fetch_fred_data('VIXCLS')
            if vix is not None:
                fred_data['vix_fred'] = vix
            dxy = await fetch_fred_data('DTWEXBGS')
            if dxy is not None:
                fred_data['dxy_fred'] = dxy
            ten_y = await fetch_fred_data('DGS10')
            if ten_y is not None:
                fred_data['ten_year_fred'] = ten_y
        except Exception:
            pass

    return {
        'upcoming_events': upcoming[:10],
        'imminent_events': imminent,
        'high_impact_events': high_impact,
        'event_warnings': event_warnings,
        'fred_data': fred_data,
        'has_high_impact_imminent': len(high_impact) > 0,
    }
