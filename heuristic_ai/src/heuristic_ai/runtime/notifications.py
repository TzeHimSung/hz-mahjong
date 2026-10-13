"""Bounded SSE wakeups; notification watermarks never consume game events."""
from __future__ import annotations

import asyncio
from collections import deque
import json
import threading
import time

import httpx
import h2.exceptions


class GameNotifications:
    def __init__(self, client, game_id, record):
        self.client, self.game_id, self.record = client, game_id, record
        self.condition = threading.Condition()
        self.seq = -1
        self.covered_seq = -1
        self.pending_frames = deque(maxlen=64)
        self.generation = 0
        self.connected = False
        self.received_at = None
        self.coalesced = 0
        self.closed = False
        self.future = client._transport.runtime.submit(self._run())

    def snapshot(self):
        with self.condition:
            seq, received, initial = self.pending_frames[0] if self.pending_frames else (None, None, True)
            return {'notification_seq': self.seq, 'notification_generation': self.generation,
                'notification_connected': self.connected, 'notification_received_monotonic': self.received_at,
                'notification_coalesced': self.coalesced,
                'notification_pending_seq': seq,
                'notification_pending_received_monotonic': received,
                'notification_pending_initial': initial}

    def event_received_at(self, seq):
        with self.condition:
            if self.connected:
                return next((received for number, received, initial in self.pending_frames
                             if number == seq and not initial), None)
        return None

    def wait(self, watermark, stop, seconds):
        end = time.monotonic() + seconds
        with self.condition:
            while not stop.is_set() and not self.closed:
                if not self.connected or self.seq > watermark or time.monotonic() >= end:
                    return False
                self.condition.wait(min(.05, end-time.monotonic()))
        return stop.is_set()

    def covered(self, watermark):
        with self.condition:
            self.covered_seq = max(self.covered_seq, watermark)
            while self.pending_frames and self.pending_frames[0][0] <= self.covered_seq:
                self.pending_frames.popleft()

    def _frame(self, data):
        value = json.loads(data)
        if (not isinstance(value, dict) or type(value.get('seq')) is not int or value['seq'] < 0
                or ('closed' in value and type(value['closed']) is not bool)):
            raise ValueError('invalid notification frame')
        with self.condition:
            initial = not self.connected
            received = time.monotonic()
            if initial:
                self.pending_frames.clear()  # Never carry a continuous-stream bound across reconnects.
            if value['seq'] > self.covered_seq and (value['seq'] > self.seq or initial and value['seq'] == self.seq):
                # Retain original receipt times when a snapshot covers only a
                # prefix. Duplicates and older frames cannot refresh a bound.
                self.pending_frames.append((value['seq'], received, initial))
            self.coalesced += int(self.seq >= 0 and (self.seq > self.covered_seq or value['seq'] <= self.seq))
            self.seq = max(self.seq, value['seq'])
            self.received_at = received
            self.connected = not value.get('closed', False)
            self.condition.notify_all()
        self.record({'kind': 'notification', 'game_id': self.game_id,
            'initial': initial, 'closed': value.get('closed', False), **self.snapshot()})
        return value.get('closed', False)

    async def _stream(self):
        request = self.client._make_request('GET', f'/api/games/{self.game_id}/notify', None)
        headers = dict(request.header_items())
        headers['Accept'] = 'text/event-stream'
        pool = self.client._transport._client('notify')
        async with pool.stream('GET', request.full_url, headers=headers,
                               timeout=httpx.Timeout(40, connect=10)) as response:
            self.client._transport._protect_http2_connections(pool, 'notify')
            if response.status_code != 200:
                # Do not expose bodies or auth headers in transport failures.
                delay = response.headers.get('retry-after', '')
                if response.status_code == 429 and delay.isdigit():
                    await asyncio.sleep(min(int(delay), 3600))
                raise OSError(f'notification HTTP {response.status_code}')
            pending, data = bytearray(), bytearray()
            async for chunk in response.aiter_bytes():
                pending.extend(chunk)
                if len(pending) + len(data) > 65536:
                    raise ValueError('notification frame exceeds 64 KiB')
                while b'\n' in pending:
                    line, _, rest = pending.partition(b'\n')
                    pending = bytearray(rest)
                    line = line.rstrip(b'\r')
                    if not line:
                        if data and self._frame(bytes(data)):
                            return
                        data.clear()
                    elif line.startswith(b'data:'):
                        if data:
                            data.extend(b'\n')
                        data.extend(line[5:].lstrip(b' '))
                    if len(data) > 65536:
                        raise ValueError('notification frame exceeds 64 KiB')

    async def _run(self):
        delay = 1
        while not self.closed:
            started = time.monotonic()
            with self.condition:
                self.generation += 1
            try:
                await self._stream()
            except asyncio.CancelledError:
                raise
            except (OSError, ValueError, httpx.HTTPError, h2.exceptions.ProtocolError) as error:
                self.record({'kind': 'notification_error', 'game_id': self.game_id,
                             'generation': self.generation, 'error_type': type(error).__name__})
            finally:
                with self.condition:
                    self.connected = False
                    self.condition.notify_all()
            if not self.closed:
                if time.monotonic()-started >= 30:
                    delay = 1
                await asyncio.sleep(delay)
                delay = min(4, delay * 2)

    def close(self):
        with self.condition:
            self.closed = True
            self.connected = False
            self.condition.notify_all()
        self.future.cancel()
