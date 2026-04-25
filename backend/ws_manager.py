"""ULTRAMAX WebSocket — Real-time price updates"""
import asyncio
import json
from typing import Set
from fastapi import WebSocket


class WSManager:
    def __init__(self):
        self.connections: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.add(ws)

    def disconnect(self, ws: WebSocket):
        self.connections.discard(ws)

    async def broadcast(self, data: dict):
        dead = set()
        for ws in self.connections:
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)
        self.connections -= dead

    async def price_feed(self, fetch_price_fn, assets: list):
        """Background task: broadcast prices every 10s."""
        while True:
            for asset in assets:
                try:
                    result = await fetch_price_fn(asset)
                    await self.broadcast({'type': 'price', 'asset': asset, 'price': result.get('price')})
                except Exception:
                    pass
            await asyncio.sleep(10)

ws_manager = WSManager()
