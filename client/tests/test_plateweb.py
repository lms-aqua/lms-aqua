"""Tests for the live plate dashboard server.

These talk to a real socket rather than poking at the handler directly, because
the parts most likely to break are exactly the ones a unit test would stub out:
SSE framing, the blocking read on ``/events``, and what a client sees when it
connects between depth frames.
"""

from __future__ import annotations

import base64
import http.client
import json
import socket
import threading
import time

import numpy as np
import pytest

from lostcam.plate import (
    HeightMap,
    Plane,
    PlateCalibration,
    PlateObject,
    summarise,
)
from lostcam.plateweb import (
    PlateWebServer,
    _Subscriber,
    build_payload,
    encode_height_map,
)

# MARK: - Fixtures


def a_calibration(cell_mm: float = 4.0, plate_mm: float = 200.0,
                  tilt_normal: tuple[float, float, float] = (0.0, 0.0, -1.0)
                  ) -> PlateCalibration:
    return PlateCalibration(
        plane=Plane(normal=tilt_normal, offset=400.0),
        intrinsics=(200.0, 200.0, 64.0, 48.0),
        depth_size=(128, 96),
        origin=(0.0, 0.0, 400.0),
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(0.0, 1.0, 0.0),
        plate_width_mm=plate_mm,
        plate_height_mm=plate_mm,
        cell_mm=cell_mm,
    )


def a_height_map(rows: int = 6, columns: int = 8,
                 cell_mm: float = 4.0) -> HeightMap:
    heights = np.zeros((rows, columns), dtype=np.float32)
    valid = np.ones((rows, columns), dtype=bool)
    heights[2:4, 3:6] = 20.0
    valid[0, 0] = False  # one genuinely unmeasured cell
    return HeightMap(heights=heights, valid=valid, cell_mm=cell_mm,
                     points_used=rows * columns)


def an_object(**overrides: object) -> PlateObject:
    fields = {
        "object_id": 1,
        "centre_u_mm": 10.0,
        "centre_v_mm": -5.0,
        "bbox_u_mm": 30.0,
        "bbox_v_mm": 20.0,
        "bbox_min_u_mm": -5.0,
        "bbox_min_v_mm": -15.0,
        "footprint_mm2": 480.0,
        "height_max_mm": 20.0,
        "height_mean_mm": 18.0,
        "volume_mm3": 8640.0,
        "cells": 30,
        "solidity": 0.8,
        "track_id": 3,
        "age_frames": 12,
    }
    fields.update(overrides)
    return PlateObject(**fields)  # type: ignore[arg-type]


@pytest.fixture
def server():
    instance = PlateWebServer(port=0)
    instance.start()
    try:
        yield instance
    finally:
        instance.stop()


def get(server: PlateWebServer, path: str, timeout: float = 5.0
        ) -> tuple[int, dict, bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", server.bound_port,
                                            timeout=timeout)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


# MARK: - Height-map encoding


class TestEncodeHeightMap:
    def test_round_trips_through_base64(self):
        height_map = a_height_map()
        encoded = encode_height_map(height_map)
        raw = base64.b64decode(encoded["data"])
        grid = np.frombuffer(raw, dtype="<u2").reshape(
            encoded["height"], encoded["width"])
        assert np.array_equal(grid, height_map.to_u16_mm())

    def test_reports_the_grid_shape_and_cell_size(self):
        encoded = encode_height_map(a_height_map(rows=5, columns=9, cell_mm=3.5))
        assert (encoded["height"], encoded["width"]) == (5, 9)
        assert encoded["cell_mm"] == 3.5
        assert encoded["format"] == "u16mm+1"

    def test_unmeasured_cells_stay_zero(self):
        """The page draws 0 as "no data", so it must not mean "0 mm"."""
        encoded = encode_height_map(a_height_map())
        grid = np.frombuffer(base64.b64decode(encoded["data"]),
                             dtype="<u2").reshape(encoded["height"],
                                                  encoded["width"])
        assert grid[0, 0] == 0

    def test_heights_carry_the_plus_one_offset(self):
        encoded = encode_height_map(a_height_map())
        grid = np.frombuffer(base64.b64decode(encoded["data"]),
                             dtype="<u2").reshape(encoded["height"],
                                                  encoded["width"])
        assert grid[2, 3] == 21  # 20 mm + 1
        assert grid[0, 1] == 1  # a measured 0 mm cell

    def test_costs_a_fixed_size_whatever_the_heights_are(self):
        """The actual reason for base64: a frame's size cannot grow as the print
        does, so a page left open for hours has a flat budget."""
        flat = a_height_map(rows=63, columns=63)
        tall = a_height_map(rows=63, columns=63)
        tall.heights[:] = 9999.0
        assert len(encode_height_map(flat)["data"]) == \
            len(encode_height_map(tall)["data"])

    def test_is_no_bigger_than_the_equivalent_json(self):
        height_map = a_height_map(rows=63, columns=63)
        height_map.heights[10:40, 15:45] = 30.0
        as_json = json.dumps(height_map.to_u16_mm().tolist())
        assert len(encode_height_map(height_map)["data"]) <= len(as_json)


# MARK: - Payload assembly


class TestBuildPayload:
    def test_carries_the_plate_geometry(self):
        calibration = a_calibration(cell_mm=3.5, plate_mm=220.0)
        payload = build_payload(summarise(a_height_map(), []), None, calibration)
        assert payload["plate"] == {
            "width_mm": 220.0,
            "height_mm": 220.0,
            "cell_mm": 3.5,
            "tilt_degrees": 0.0,
        }

    def test_includes_the_state_fields(self):
        state = summarise(a_height_map(), [an_object()])
        payload = build_payload(state, None, a_calibration(), timestamp_ms=1234)
        assert payload["t"] == 1234
        assert payload["object_count"] == 1
        assert payload["tallest_mm"] == 20.0
        assert payload["objects"][0]["track"] == 3

    def test_omits_the_map_when_there_is_none(self):
        payload = build_payload(summarise(a_height_map(), []), None,
                                a_calibration())
        assert "map" not in payload

    def test_includes_the_map_when_given_one(self):
        payload = build_payload(summarise(a_height_map(), []), a_height_map(),
                                a_calibration())
        assert payload["map"]["width"] == 8

    def test_extra_fields_are_merged(self):
        payload = build_payload(summarise(a_height_map(), []), None,
                                a_calibration(), extra={"source": "mock"})
        assert payload["source"] == "mock"

    def test_reports_a_tilted_rig(self):
        tilted = a_calibration(tilt_normal=(0.0, -0.5, -0.8660254))
        payload = build_payload(summarise(a_height_map(), []), None, tilted)
        assert payload["plate"]["tilt_degrees"] == pytest.approx(30.0, abs=0.05)

    def test_is_json_serialisable(self):
        """numpy scalars leaking into the payload would break every client."""
        state = summarise(a_height_map(), [an_object()])
        payload = build_payload(state, a_height_map(), a_calibration())
        json.dumps(payload)


# MARK: - Latest-wins subscriber


class TestSubscriber:
    def test_takes_the_frame_it_was_offered(self):
        subscriber = _Subscriber()
        subscriber.offer(b"one")
        assert subscriber.take(0.5) == b"one"

    def test_a_second_offer_replaces_the_first(self):
        subscriber = _Subscriber()
        subscriber.offer(b"stale")
        subscriber.offer(b"fresh")
        assert subscriber.take(0.5) == b"fresh"
        assert subscriber.dropped == 1

    def test_returns_none_when_nothing_arrives(self):
        """This is what triggers the keep-alive comment."""
        assert _Subscriber().take(0.01) is None

    def test_close_wakes_a_waiting_reader(self):
        subscriber = _Subscriber()
        threading.Timer(0.05, subscriber.close).start()
        subscriber.take(5.0)
        assert not subscriber.alive


# MARK: - Routes


class TestRoutes:
    def test_healthz_is_ok(self, server):
        status, _, body = get(server, "/healthz")
        assert (status, body) == (200, b"ok")

    def test_root_serves_the_dashboard(self, server):
        status, headers, body = get(server, "/")
        assert status == 200
        assert headers["Content-Type"].startswith("text/html")
        assert b"EventSource" in body

    def test_state_json_starts_empty_but_valid(self, server):
        status, headers, body = get(server, "/state.json")
        assert status == 200
        assert headers["Content-Type"] == "application/json"
        assert json.loads(body) == {}

    def test_state_json_returns_the_last_published_state(self, server):
        server.publish_state(summarise(a_height_map(), [an_object()]),
                             a_height_map(), a_calibration(), timestamp_ms=7)
        _, _, body = get(server, "/state.json")
        document = json.loads(body)
        assert document["t"] == 7
        assert document["object_count"] == 1

    def test_unknown_paths_are_404(self, server):
        status, _, _ = get(server, "/nope")
        assert status == 404

    def test_trailing_slashes_are_ignored(self, server):
        assert get(server, "/healthz/")[0] == 200

    def test_query_strings_are_ignored(self, server):
        assert get(server, "/state.json?t=1")[0] == 200

    def test_the_page_is_not_cached(self, server):
        """A stale dashboard next to a running printer is worse than none."""
        _, headers, _ = get(server, "/state.json")
        assert headers["Cache-Control"] == "no-store"


# MARK: - The event stream


class _EventClient:
    """A minimal SSE reader, so the framing is checked and not assumed."""

    def __init__(self, port: int, timeout: float = 5.0) -> None:
        self.connection = http.client.HTTPConnection("127.0.0.1", port,
                                                     timeout=timeout)
        self.connection.request("GET", "/events")
        self.response = self.connection.getresponse()
        self.buffer = b""

    @property
    def content_type(self) -> str:
        return self.response.getheader("Content-Type", "")

    def next_block(self, deadline: float = 5.0) -> str:
        """Read up to the next blank-line-terminated SSE block."""
        end = time.monotonic() + deadline
        while b"\n\n" not in self.buffer:
            if time.monotonic() > end:
                raise AssertionError("no SSE block arrived")
            chunk = self.response.read1(4096)
            if not chunk:
                raise AssertionError("stream closed")
            self.buffer += chunk
        block, _, self.buffer = self.buffer.partition(b"\n\n")
        return block.decode("utf-8")

    def next_event(self, deadline: float = 5.0) -> dict:
        while True:
            block = self.next_block(deadline)
            if block.startswith("data: "):
                return json.loads(block[len("data: "):])

    def close(self) -> None:
        # The response must be closed too, not just the connection: the response's
        # file object holds a reference to the same socket, so closing only the
        # connection leaves the descriptor open and the server never sees the tab
        # go away.
        try:
            self.response.close()
            self.connection.close()
        except OSError:
            pass


def wait_for_clients(server: PlateWebServer, count: int,
                     deadline: float = 5.0) -> None:
    end = time.monotonic() + deadline
    while server.client_count < count:
        if time.monotonic() > end:
            raise AssertionError(
                f"expected {count} clients, have {server.client_count}")
        time.sleep(0.01)


class TestEventStream:
    def test_declares_the_event_stream_type(self, server):
        client = _EventClient(server.bound_port)
        try:
            assert client.content_type == "text/event-stream"
        finally:
            client.close()

    def test_a_new_tab_gets_the_current_state_at_once(self, server):
        """Otherwise the page is blank until the next depth frame — minutes, on
        a slow capture."""
        server.publish_state(summarise(a_height_map(), [an_object()]),
                             a_height_map(), a_calibration(), timestamp_ms=99)
        client = _EventClient(server.bound_port)
        try:
            assert client.next_event()["t"] == 99
        finally:
            client.close()

    def test_publishes_reach_a_connected_client(self, server):
        client = _EventClient(server.bound_port)
        try:
            client.next_event()  # the priming frame
            wait_for_clients(server, 1)
            server.publish({"t": 1, "object_count": 2})
            assert client.next_event()["object_count"] == 2
        finally:
            client.close()

    def test_frames_are_separated_by_a_blank_line(self, server):
        client = _EventClient(server.bound_port)
        try:
            client.next_block()
            wait_for_clients(server, 1)
            server.publish({"t": 1})
            block = client.next_block()
            assert block.startswith("data: ")
            assert "\n" not in block  # one line, then the blank separator
        finally:
            client.close()

    def test_every_client_gets_every_update(self, server):
        first = _EventClient(server.bound_port)
        second = _EventClient(server.bound_port)
        try:
            first.next_event()
            second.next_event()
            wait_for_clients(server, 2)
            server.publish({"t": 5})
            assert first.next_event()["t"] == 5
            assert second.next_event()["t"] == 5
        finally:
            first.close()
            second.close()

    def test_a_closed_tab_is_forgotten(self, server):
        client = _EventClient(server.bound_port)
        client.next_event()
        wait_for_clients(server, 1)
        client.close()
        # The handler only notices on its next write, so nudge it.
        end = time.monotonic() + 5.0
        while server.client_count and time.monotonic() < end:
            server.publish({"t": 0})
            time.sleep(0.05)
        assert server.client_count == 0

    def test_publishing_with_no_clients_is_harmless(self, server):
        for index in range(5):
            server.publish({"t": index})
        assert server.published == 5
        assert json.loads(server.latest)["t"] == 4

    def test_stop_releases_the_clients(self, server):
        client = _EventClient(server.bound_port)
        try:
            client.next_event()
            wait_for_clients(server, 1)
            server.stop()
            assert server.client_count == 0
        finally:
            client.close()


# MARK: - Lifecycle


class TestLifecycle:
    def test_reports_the_port_it_actually_bound(self):
        with PlateWebServer(port=0) as instance:
            assert instance.bound_port != 0
            assert str(instance.bound_port) in instance.url

    def test_url_names_a_reachable_host_when_bound_to_every_interface(self):
        """``0.0.0.0`` is not something you can paste into a browser."""
        with PlateWebServer(port=0, host="0.0.0.0") as instance:
            assert instance.url.startswith("http://127.0.0.1:")

    def test_stopping_frees_the_port(self):
        instance = PlateWebServer(port=0)
        instance.start()
        port = instance.bound_port
        instance.stop()
        # Binding again is the only honest check that the socket really closed.
        probe = socket.socket()
        try:
            probe.bind(("127.0.0.1", port))
        finally:
            probe.close()

    def test_stopping_twice_is_not_an_error(self):
        instance = PlateWebServer(port=0)
        instance.start()
        instance.stop()
        instance.stop()

    def test_a_busy_port_raises_rather_than_binding_silently(self):
        """The CLI catches this and carries on without a dashboard, which only
        works if it is actually raised."""
        holder = socket.socket()
        holder.bind(("127.0.0.1", 0))
        holder.listen(1)
        try:
            with pytest.raises(OSError):
                PlateWebServer(port=holder.getsockname()[1]).start()
        finally:
            holder.close()
