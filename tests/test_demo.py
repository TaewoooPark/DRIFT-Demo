"""Torch-free smoke tests for the demo's plumbing — the state fold, the
exactly-once attach semantics, the API validation, and the UDP emitter.
These are exactly the pieces the review rounds found bugs in.

    .venv/bin/python -m unittest discover -s tests
"""

from __future__ import annotations

import http.client
import json
import queue
import socket
import threading
import time
import unittest

from demo import events as demo_events
from demo.server import Bus, Handler, ThreadingHTTPServer

NODE = "aa" * 8


def receipt_evt(node: str = NODE, start: int = 0, end: int = 14, checked: int = 1):
    return {"t": "head.receipts", "ts": 0.0, "session": "s", "seq": 1,
            "mode": "decode", "checked": checked, "suspects": [],
            "hops": [{"node": node, "start": start, "end": end,
                      "in": "x", "out": "y", "sig": "z"}]}


class TestBusFold(unittest.TestCase):
    def test_ledger_fold_matches_layer_token_math(self):
        bus = Bus()
        for _ in range(5):
            bus.publish(receipt_evt(start=14, end=28))
        a = bus.snapshot()["ledger"][NODE]
        self.assertEqual(a["tokens"], 5)
        self.assertEqual(a["layer_tokens"], 5 * 14)

    def test_session_lifecycle(self):
        bus = Bus()
        bus.publish({"t": "head.session_start", "session": "s1", "prompt": "p"})
        self.assertTrue(bus.snapshot()["busy"])
        bus.publish({"t": "head.token", "text": "hi", "i": 1, "tps": 1.0})
        bus.publish({"t": "head.session_end"})
        s = bus.snapshot()
        self.assertFalse(s["busy"])
        self.assertEqual(s["last"]["text"], "hi")
        self.assertEqual(s["sessions"], 1)

    def test_malformed_events_never_break_the_fold(self):
        bus = Bus()
        bus.publish({"t": "head.receipts", "hops": [{"bad": 1}]})
        bus.publish({"t": "unknown.event", "x": object.__class__.__name__})
        self.assertEqual(bus.snapshot()["ledger"], {})  # dropped, not crashed


class TestAttachExactlyOnce(unittest.TestCase):
    def test_no_double_count_across_attach(self):
        """A client attaching mid-traffic must account for every event exactly
        once: folded-into-snapshot + delivered-to-queue == published."""
        bus = Bus()
        n_events = 1500  # < queue maxsize, so nothing is dropped

        def pump():
            for _ in range(n_events):
                bus.publish(receipt_evt())

        t = threading.Thread(target=pump)
        t.start()
        attaches = []
        for _ in range(20):
            attaches.append(bus.attach())
            time.sleep(0.001)
        t.join()
        attaches.append(bus.attach())  # and one after the dust settles

        for q, snap in attaches:
            folded = snap["ledger"].get(NODE, {}).get("tokens", 0)
            drained = 0
            try:
                while True:
                    if q.get_nowait()["t"] == "head.receipts":
                        drained += 1
            except queue.Empty:
                pass
            self.assertEqual(folded + drained, n_events,
                             "an event was double-delivered or lost at attach time")


class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Handler.bus = Bus()
        Handler.head = None  # not assembled — generate must answer 503
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.httpd.daemon_threads = True
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def _post(self, body: bytes, content_length: int | None = None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            if content_length is not None:
                c.putrequest("POST", "/api/generate")
                c.putheader("Content-Type", "application/json")
                c.putheader("Content-Length", str(content_length))
                c.endheaders()
                c.send(body)
            else:
                c.request("POST", "/api/generate", body,
                          {"Content-Type": "application/json"})
            r = c.getresponse()
            return r.status, json.loads(r.read() or b"{}")
        finally:
            c.close()

    def test_state_ok(self):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            c.request("GET", "/api/state")
            r = c.getresponse()
            self.assertEqual(r.status, 200)
            self.assertIn("ready", json.loads(r.read()))
        finally:
            c.close()

    def test_bad_json_400(self):
        self.assertEqual(self._post(b"{nope")[0], 400)

    def test_empty_prompt_400(self):
        self.assertEqual(self._post(json.dumps({"prompt": "  "}).encode())[0], 400)

    def test_bad_max_new_tokens_400(self):
        st, _ = self._post(json.dumps({"prompt": "x", "max_new_tokens": "abc"}).encode())
        self.assertEqual(st, 400)

    def test_body_too_large_413(self):
        st, _ = self._post(b"{}", content_length=2_000_000)
        self.assertEqual(st, 413)

    def test_not_assembled_503(self):
        st, _ = self._post(json.dumps({"prompt": "hi"}).encode())
        self.assertEqual(st, 503)


class TestEmitter(unittest.TestCase):
    def test_delivery_and_oversize_truncation(self):
        rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rx.bind(("127.0.0.1", 0))
        rx.settimeout(2)
        em = demo_events.Emitter(f"127.0.0.1:{rx.getsockname()[1]}")

        em.emit("small", a=1)
        e = json.loads(rx.recvfrom(65535)[0])
        self.assertEqual((e["t"], e["a"]), ("small", 1))

        em.emit("big", blob="x" * 20000)  # over the datagram cap
        e = json.loads(rx.recvfrom(65535)[0])
        self.assertEqual(e["t"], "big")
        self.assertTrue(e.get("truncated"))
        rx.close()

    def test_disabled_without_targets(self):
        em = demo_events.Emitter("")
        em.emit("noop", x=1)  # must be a no-op, not an error


if __name__ == "__main__":
    unittest.main()
