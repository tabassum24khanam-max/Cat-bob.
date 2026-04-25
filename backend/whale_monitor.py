"""ULTRAMAX Whale Monitor — Free blockchain transaction monitoring"""
import os
import httpx


# Approximate BTC price for USD estimation (updated at runtime if possible)
_APPROX_BTC_PRICE = 60000.0
_WHALE_THRESHOLD_USD = 1_000_000


async def _get_btc_price() -> float:
    """Try to fetch current BTC price for accurate USD calculations."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(
                "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
            )
            if resp.status_code == 200:
                return float(resp.json()['price'])
    except Exception:
        pass
    return _APPROX_BTC_PRICE


async def _get_btc_whales() -> list:
    """Fetch recent large BTC transactions from blockchain.info."""
    try:
        btc_price = await _get_btc_price()
        threshold_btc = _WHALE_THRESHOLD_USD / btc_price

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://blockchain.info/unconfirmed-transactions?format=json"
            )
            if resp.status_code != 200:
                return []
            data = resp.json()

        txs = data.get('txs', [])
        whales = []

        for tx in txs:
            total_out = sum(o.get('value', 0) for o in tx.get('out', []))
            total_btc = total_out / 1e8  # satoshis to BTC
            amount_usd = total_btc * btc_price

            if amount_usd >= _WHALE_THRESHOLD_USD:
                # Determine direction: if any output has no 'addr' it may be
                # an exchange deposit (known exchange addresses are complex to
                # detect without a database, so we use input/output count heuristic)
                n_inputs = len(tx.get('inputs', []))
                n_outputs = len(tx.get('out', []))

                # Many inputs -> few outputs = consolidation (often exchange deposit)
                # Few inputs -> many outputs = distribution (often exchange withdrawal)
                if n_inputs > n_outputs:
                    direction = 'exchange_inflow'
                elif n_outputs > n_inputs + 1:
                    direction = 'exchange_outflow'
                else:
                    direction = 'transfer'

                whales.append({
                    'type': 'BTC',
                    'amount_usd': round(amount_usd, 2),
                    'direction': direction,
                })

        return whales[:20]  # Cap at 20 most recent

    except Exception:
        return []


async def _get_eth_whales() -> list:
    """Fetch recent large ETH transactions from Etherscan if API key available."""
    etherscan_key = os.getenv('ETHERSCAN_API_KEY', '')
    if not etherscan_key:
        return []

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # Get latest ETH price
            price_resp = await client.get(
                "https://api.etherscan.io/api",
                params={
                    'module': 'stats',
                    'action': 'ethprice',
                    'apikey': etherscan_key,
                }
            )
            eth_price = 3000.0  # fallback
            if price_resp.status_code == 200:
                price_data = price_resp.json()
                if price_data.get('status') == '1':
                    eth_price = float(price_data['result']['ethusd'])

            # Get latest blocks for large transactions
            resp = await client.get(
                "https://api.etherscan.io/api",
                params={
                    'module': 'proxy',
                    'action': 'eth_blockNumber',
                    'apikey': etherscan_key,
                }
            )
            if resp.status_code != 200:
                return []

            block_hex = resp.json().get('result', '0x0')
            block_num = int(block_hex, 16)

            # Get transactions from latest block
            block_resp = await client.get(
                "https://api.etherscan.io/api",
                params={
                    'module': 'proxy',
                    'action': 'eth_getBlockByNumber',
                    'tag': hex(block_num),
                    'boolean': 'true',
                    'apikey': etherscan_key,
                }
            )
            if block_resp.status_code != 200:
                return []

            block_data = block_resp.json().get('result', {})
            transactions = block_data.get('transactions', [])

            whales = []
            for tx in transactions:
                value_wei = int(tx.get('value', '0x0'), 16)
                value_eth = value_wei / 1e18
                amount_usd = value_eth * eth_price

                if amount_usd >= _WHALE_THRESHOLD_USD:
                    whales.append({
                        'type': 'ETH',
                        'amount_usd': round(amount_usd, 2),
                        'direction': 'transfer',
                    })

            return whales[:20]

    except Exception:
        return []


async def get_whale_activity(asset: str) -> dict:
    """Check for recent large transactions.
    Returns: {available, recent_whales: [{type, amount_usd, direction}],
              net_flow: float, bias: 'bullish'|'bearish'|'neutral'}
    """
    try:
        if asset == 'BTC':
            whales = await _get_btc_whales()
        elif asset == 'ETH':
            whales = await _get_eth_whales()
        else:
            return {'available': False}

        if not whales:
            return {'available': False}

        # Calculate net flow:
        # Positive net_flow = more flowing to exchanges (bearish)
        # Negative net_flow = more flowing out of exchanges (bullish)
        net_flow = 0.0
        for w in whales:
            if w['direction'] == 'exchange_inflow':
                net_flow += w['amount_usd']
            elif w['direction'] == 'exchange_outflow':
                net_flow -= w['amount_usd']
            # 'transfer' does not affect net flow

        # Determine bias
        if net_flow < -_WHALE_THRESHOLD_USD:
            bias = 'bullish'   # Money leaving exchanges
        elif net_flow > _WHALE_THRESHOLD_USD:
            bias = 'bearish'   # Money entering exchanges
        else:
            bias = 'neutral'

        return {
            'available': True,
            'recent_whales': whales,
            'net_flow': round(net_flow, 2),
            'bias': bias,
        }

    except Exception:
        return {'available': False}
