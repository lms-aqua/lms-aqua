"""Unit tests for adb, audio, discovery and source-URL handling."""

from __future__ import annotations

import numpy as np
import pytest

from lostcam.adb import AdbError, Device, parse_devices, pick_serial
from lostcam.audio import (
    AudioFormat,
    parse_audio_content_type,
    s16le_to_float32,
)
from lostcam.discovery import parse_reply
from lostcam.puller import Source, resolve_size


class TestParseDevices:
    def test_parses_ready_device(self):
        output = "List of devices attached\nR3CN90ABCDE\tdevice\n\n"
        assert parse_devices(output) == [Device("R3CN90ABCDE", "device")]

    def test_parses_multiple_states(self):
        output = (
            "List of devices attached\n"
            "aaaa\tdevice\n"
            "bbbb\tunauthorized\n"
            "cccc\toffline\n"
        )
        devices = parse_devices(output)
        assert [d.state for d in devices] == ["device", "unauthorized", "offline"]
        assert [d.usable for d in devices] == [True, False, False]

    def test_skips_daemon_chatter(self):
        output = (
            "* daemon not running; starting now at tcp:5037\n"
            "* daemon started successfully\n"
            "List of devices attached\n"
            "xyz\tdevice\n"
        )
        assert parse_devices(output) == [Device("xyz", "device")]

    def test_empty_output(self):
        assert parse_devices("List of devices attached\n\n") == []


class TestPickSerial:
    def test_single_usable_device_is_chosen(self):
        assert pick_serial([Device("only", "device")]) == "only"

    def test_ambiguous_requires_serial(self):
        devices = [Device("a", "device"), Device("b", "device")]
        with pytest.raises(AdbError, match="pass --serial"):
            pick_serial(devices)

    def test_requested_serial_wins(self):
        devices = [Device("a", "device"), Device("b", "device")]
        assert pick_serial(devices, "b") == "b"

    def test_requested_but_absent(self):
        with pytest.raises(AdbError, match="not attached"):
            pick_serial([Device("a", "device")], "zzz")

    def test_requested_but_unauthorized_explains_the_prompt(self):
        with pytest.raises(AdbError, match="USB debugging prompt"):
            pick_serial([Device("a", "unauthorized")], "a")

    def test_no_devices_at_all(self):
        with pytest.raises(AdbError, match="Connect the phone by USB"):
            pick_serial([])

    def test_only_unusable_devices_lists_them(self):
        with pytest.raises(AdbError, match="Attached but not ready"):
            pick_serial([Device("a", "offline")])


class TestAudioFormat:
    def test_defaults(self):
        fmt = parse_audio_content_type(None)
        assert (fmt.rate, fmt.channels) == (44100, 1)

    def test_parses_rate_and_channels(self):
        fmt = parse_audio_content_type("audio/L16; rate=48000; channels=2")
        assert (fmt.rate, fmt.channels) == (48000, 2)

    def test_ignores_unparseable_values(self):
        fmt = parse_audio_content_type("audio/L16; rate=abc; channels=2")
        assert (fmt.rate, fmt.channels) == (44100, 2)

    def test_frame_bytes_and_duration(self):
        fmt = AudioFormat(rate=8000, channels=2)
        assert fmt.frame_bytes == 4
        assert fmt.duration(8000 * 4) == pytest.approx(1.0)


class TestPCMConversion:
    def test_converts_full_scale_values(self):
        data = np.array([0, 32767, -32768], dtype="<i2").tobytes()
        out = s16le_to_float32(data)
        assert out.shape == (3, 1)
        assert out[0, 0] == pytest.approx(0.0)
        assert out[1, 0] == pytest.approx(1.0, abs=1e-4)
        assert out[2, 0] == pytest.approx(-1.0)

    def test_stereo_deinterleaves(self):
        data = np.array([100, 200, 300, 400], dtype="<i2").tobytes()
        out = s16le_to_float32(data, channels=2)
        assert out.shape == (2, 2)
        assert out[0, 0] < out[0, 1]

    def test_partial_sample_is_discarded_not_misaligned(self):
        data = np.array([1000, 2000], dtype="<i2").tobytes() + b"\x01"
        assert s16le_to_float32(data).shape == (2, 1)

    def test_empty_input(self):
        assert s16le_to_float32(b"").shape == (0, 1)


class TestDiscoveryReply:
    def test_parses_valid_reply(self):
        payload = b'{"product":"LostCam","port":4747,"device":"iPhone","platform":"ios"}'
        found = parse_reply(payload, "192.168.1.5")
        assert found is not None
        assert (found.host, found.port, found.device) == ("192.168.1.5", 4747, "iPhone")
        assert "192.168.1.5:4747" in found.label

    def test_defaults_port_when_absent(self):
        found = parse_reply(b'{"product":"LostCam"}', "10.0.0.2")
        assert found is not None and found.port == 4747

    @pytest.mark.parametrize(
        "payload",
        [
            b"not json at all",
            b"[1,2,3]",
            b'{"product":"SomethingElse"}',
            b'{"product":"LostCam","port":0}',
            b'{"product":"LostCam","port":99999}',
            b'{"product":"LostCam","port":"4747"}',
            b"\xff\xfe\x00binary",
        ],
    )
    def test_rejects_junk_without_raising(self, payload):
        """Anything on a broadcast port may answer; validate, never trust."""
        assert parse_reply(payload, "1.2.3.4") is None


class TestSource:
    def test_plain_url(self):
        source = Source("192.168.1.9")
        assert source.url == "http://192.168.1.9:4747/video"

    def test_query_parameters_are_encoded(self):
        source = Source("h", width=1920, height=1080, fps=24, quality=70, camera="front")
        path = source.request_path
        assert path.startswith("/video?")
        for expected in ("w=1920", "h=1080", "fps=24", "q=70", "cam=front"):
            assert expected in path

    def test_token_goes_in_query_and_header(self):
        source = Source("h", token="s3cret")
        assert "token=s3cret" in source.request_path
        assert source.headers["X-LostCam-Token"] == "s3cret"

    def test_no_token_header_when_absent(self):
        assert "X-LostCam-Token" not in Source("h").headers


class TestResolveSize:
    def test_explicit_flags_win(self):
        source = Source("h", width=800, height=600)
        info = {"video": {"width": 1920, "height": 1080}}
        assert resolve_size(source, info) == (800, 600)

    def test_falls_back_to_info(self):
        info = {"video": {"width": 1280, "height": 720}}
        assert resolve_size(Source("h"), info) == (1280, 720)

    @pytest.mark.parametrize(
        "info",
        [
            None,
            {},
            {"video": "nonsense"},
            {"video": {"width": 0, "height": 720}},
            {"video": {"width": "1280", "height": "720"}},
        ],
    )
    def test_returns_none_when_unknown(self, info):
        assert resolve_size(Source("h"), info) is None

    def test_partial_flags_are_not_enough(self):
        assert resolve_size(Source("h", width=800), None) is None
