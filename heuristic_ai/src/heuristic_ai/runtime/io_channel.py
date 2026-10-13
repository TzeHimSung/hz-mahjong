"""Bounded FIFO pipe framing shared by the I/O parent and child."""
import io
import logging
import multiprocessing
import pickle
import queue
import threading
import time

_MAX_MESSAGE = 2 * 1024 * 1024
_MAX_BATCH_MESSAGES = 64


class IoFailure(RuntimeError):
    pass


def _read_messages(connection, stop=None):
    while stop is None or not stop.is_set():
        if stop is not None and not connection.poll(0.1):
            continue
        data = connection.recv_bytes(_MAX_MESSAGE)
        stream = io.BytesIO(data)
        for _ in range(_MAX_BATCH_MESSAGES):
            if stop is not None and stop.is_set():
                return
            # Each queued pickle has its own memo; use a fresh unpickler.
            yield pickle.load(stream)
            if stream.tell() == len(data):
                break
        else:
            raise IoFailure('I/O batch exceeds bounded message count')


class _PipeWriter:
    def __init__(self, connection, fail):
        self.connection, self.fail = connection, fail
        self.queue = queue.Queue(2048)
        self.error = None
        self.closed = False
        self._failure_lock = threading.Lock()
        self.write_started_at = self.write_finished_at = 0.0
        self.write_bytes = 0
        self.write_message_count = 0
        self.write_in_progress = False
        self.thread = threading.Thread(target=self._run, name='mahjong-ipc-writer', daemon=True)
        self.thread.start()

    def _failed(self, error, message_kind=None):
        with self._failure_lock:
            if self.error is not None:
                return
            self.error = error
        try:
            logging.getLogger(__name__).error(
                'ipc_writer_failure unix=%.6f monotonic=%.6f process=%s pid=%s '
                'error=%s queued=%s writer_alive=%s write_started=%.6f '
                'write_finished=%.6f write_bytes=%s write_message_count=%s write_in_progress=%s failed_kind=%s',
                time.time(), time.monotonic(), multiprocessing.current_process().name,
                multiprocessing.current_process().pid, type(error).__name__,
                self.queue.qsize(), self.thread.is_alive(), self.write_started_at,
                self.write_finished_at, self.write_bytes, self.write_message_count, self.write_in_progress, message_kind)
        except Exception:
            pass  # Diagnostic output must not prevent the original channel failure.
        self.fail(error)

    def send(self, message):
        if self.error is not None or self.closed:
            raise IoFailure('I/O message channel unavailable')
        try:
            payload = pickle.dumps(message, protocol=5)
            if len(payload) > _MAX_MESSAGE:
                raise IoFailure('I/O message exceeds bounded channel limit')
            self.queue.put_nowait(payload)
        except BaseException as error:
            kind = message[0] if isinstance(message, tuple) and message and isinstance(message[0], str) else type(message).__name__
            self._failed(error, kind)
            raise

    def _run(self):
        pending = None
        try:
            while True:
                payload = self.queue.get() if pending is None else pending
                pending = None
                if payload is None:
                    return
                batch, size, stopping = [payload], len(payload), False
                # Only drain messages already queued: no batching timer/delay.
                while len(batch) < _MAX_BATCH_MESSAGES:
                    try:
                        following = self.queue.get_nowait()
                    except queue.Empty:
                        break
                    if following is None:
                        stopping = True
                        break
                    if size + len(following) > _MAX_MESSAGE:
                        pending = following
                        break
                    batch.append(following)
                    size += len(following)
                payload = batch[0] if len(batch) == 1 else b''.join(batch)
                self.write_bytes = len(payload)
                self.write_message_count = len(batch)
                self.write_started_at = time.monotonic()
                self.write_in_progress = True
                try:
                    self.connection.send_bytes(payload)
                finally:
                    self.write_in_progress = False
                self.write_finished_at = time.monotonic()
                if stopping:
                    return
        except BaseException as error:
            self._failed(error)

    def close(self):
        self.closed = True
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        self.thread.join(1)
