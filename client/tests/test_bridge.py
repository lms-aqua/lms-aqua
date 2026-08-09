"""OSC encoding and bridge fan-out tests."""

from __future__ import annotations

import json
import socket
import struct

import pytest

from lostcam import osc
from lostcam.bridge import (
    Bridge,
    CallbackSink,
    CSVSink,
    JSONLSink,
    OSCSink,
    UDPJSONSink,
)
from lostcam.datastream import Sample


def sample(channel: str = "ar.face", **fields: object) -> Sample:
    raw = {"t": 1000, "seq": 1, "ch": channel, **fields}
    return Sample(1000, 1, channel, raw)


class TestOSCEncoding:
    def test_address_is_null_terminated_and_padded(self):
        # "/ab" is 3 bytes + null = 4, already aligned.
        assert osc.message("/ab") == b"/ab\x00,\x00\x00\x00"

    def test_four_char_address_pads_to_eight(self):
        """The classic off-by-four: null terminator then pad."""
        encoded = osc.message("/abc")
        assert encoded[:8] == b"/abc\x00\x00\x00\x00"

    def test_float_argument(self):
        encoded = osc.message("/f", 0.5)
        assert encoded.endswith(struct.pack(">f", 0.5))
        assert b",f\x00\x00" in encoded

    def test_int_argument(self):
        encoded = osc.message("/i", 42)
        assert encoded.endswith(struct.pack(">i", 42))

    def test_bool_uses_type_tag_only(self):
        """OSC booleans carry no payload — T/F is the whole value."""
        true_msg = osc.message("/b", True)
        false_msg = osc.message("/b", False)
        assert b",T" in true_msg
        assert b",F" in false_msg
        assert len(true_msg) == len(false_msg)

    def test_string_argument(self):
        encoded = osc.message("/s", "normal")
        assert b"normal\x00" in encoded

    def test_none_becomes_nil_tag(self):
        assert b",N" in osc.message("/n", None)

    def test_mixed_arguments_keep_tag_order(self):
        encoded = osc.message("/m", 1, 2.0, "three")
        assert b",ifs" in encoded

    def test_huge_int_falls_back_to_float(self):
        """Out of int32 range must not silently truncate."""
        encoded = osc.message("/big", 2**40)
        assert b",f" in encoded

    def test_every_message_length_is_multiple_of_four(self):
        for args in ([], [1], [1.5], ["x"], ["abcd"], [1, "ab", 2.5, True]):
            assert len(osc.message("/t", *args)) % 4 == 0

    def test_unsupported_type_rejected(self):
        with pytest.raises(osc.OSCError, match="cannot encode"):
            osc.message("/x", {"a": 1})


class TestOSCAddress:
    def test_dots_become_path_separators(self):
        assert osc.sanitize_address("blend.jawOpen") == "/blend/jawOpen"

    def test_leading_slash_added(self):
        assert osc.sanitize_address("motion") == "/motion"

    @pytest.mark.parametrize("char", ["#", "*", ",", "?", "[", "]", "{", "}", " "])
    def test_reserved_characters_replaced(self, char):
        assert char not in osc.sanitize_address(f"a{char}b")

    def test_double_slashes_collapsed(self):
        assert osc.sanitize_address("/a//b") == "/a/b"

    def test_empty_rejected(self):
        with pytest.raises(osc.OSCError):
            osc.sanitize_address("")


class TestOSCBundle:
    def test_bundle_header_and_immediate_timetag(self):
        encoded = osc.bundle([osc.message("/a", 1)])
        assert encoded.startswith(b"#bundle\x00")
        assert encoded[8:16] == struct.pack(">Q", 1)

    def test_elements_are_length_prefixed(self):
        element = osc.message("/a", 1)
        encoded = osc.bundle([element])
        assert encoded[16:20] == struct.pack(">i", len(element))

    def test_bad_timetag_rejected(self):
        with pytest.raises(osc.OSCError, match="8 bytes"):
            osc.bundle([], timetag=b"short")


class TestOSCSink:
    def test_blendshape_lands_at_expected_address(self):
        sink = OSCSink()
        payload = sink.build(sample("ar.face", blend={"jawOpen": 0.4}))
        assert b"/lostcam/ar/face/blend/jawOpen" in payload
        sink.close()

    def test_pose_elements_are_addressed_individually(self):
        sink = OSCSink()
        payload = sink.build(sample("ar.world", pose=[0.0] * 15 + [1.0]))
        assert b"/lostcam/ar/world/pose/12" in payload
        sink.close()

    def test_timestamp_always_included(self):
        sink = OSCSink()
        assert b"/lostcam/ar/face/t" in sink.build(sample())
        sink.close()

    def test_control_fields_are_not_duplicated_as_values(self):
        sink = OSCSink()
        payload = sink.build(sample("motion", accel=[1.0]))
        assert payload.count(b"/lostcam/motion/seq") == 0
        sink.close()

    def test_send_to_nothing_does_not_raise(self):
        """A UDP receiver that is not running must not break the bridge."""
        sink = OSCSink("127.0.0.1", 9)
        sink.handle(sample())
        sink.close()


class TestUDPJSONSink:
    def test_datagram_arrives_intact(self):
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(2.0)
        port = receiver.getsockname()[1]

        sink = UDPJSONSink("127.0.0.1", port)
        sink.handle(sample("motion", accel=[0.0, 0.0, 0.98]))
        payload, _ = receiver.recvfrom(65535)
        receiver.close()
        sink.close()

        record = json.loads(payload)
        assert record["ch"] == "motion"
        assert record["accel"] == [0.0, 0.0, 0.98]

    def test_oversized_sample_is_counted_not_fragmented(self):
        sink = UDPJSONSink("127.0.0.1", 9)
        sink.handle(sample("big", blob="x" * 70000))
        assert sink.oversized == 1
        assert sink.stats.delivered == 0
        sink.close()


class TestFileSinks:
    def test_jsonl_writes_one_line_per_sample(self, tmp_path):
        path = tmp_path / "out.jsonl"
        sink = JSONLSink(path)
        sink.handle(sample("motion", accel=[1, 2, 3]))
        sink.handle(sample("motion", accel=[4, 5, 6]))
        sink.close()

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[1])["accel"] == [4, 5, 6]

    def test_csv_writes_one_file_per_channel(self, tmp_path):
        sink = CSVSink(tmp_path)
        sink.handle(sample("motion", accel=[1, 2, 3]))
        sink.handle(sample("ar.face", blend={"jawOpen": 0.5}))
        sink.close()

        assert (tmp_path / "motion.csv").exists()
        assert (tmp_path / "ar_face.csv").exists()

    def test_csv_header_comes_from_flattened_keys(self, tmp_path):
        sink = CSVSink(tmp_path)
        sink.handle(sample("attitude", q=[0.0, 0.1, 0.2, 0.3]))
        sink.close()

        header = (tmp_path / "attitude.csv").read_text().splitlines()[0]
        assert "q.0" in header and "q.3" in header

    def test_warmup_unions_sparse_blendshape_columns(self, tmp_path):
        """Sparse channels must not have their columns fixed by sample one."""
        sink = CSVSink(tmp_path, warmup=10)
        sink.handle(sample("ar.face", blend={"jawOpen": 0.4}))
        sink.handle(sample("ar.face", blend={"eyeBlinkLeft": 0.2}))
        sink.handle(sample("ar.face", blend={"mouthSmileRight": 0.7}))
        sink.close()

        header = (tmp_path / "ar_face.csv").read_text().splitlines()[0]
        for column in ("blend.jawOpen", "blend.eyeBlinkLeft", "blend.mouthSmileRight"):
            assert column in header, f"{column} missing from header"
        assert not sink.late_columns

    def test_all_rows_survive_the_warmup_buffer(self, tmp_path):
        sink = CSVSink(tmp_path, warmup=5)
        for _ in range(7):
            sink.handle(sample("light", lumens=900.0))
        sink.close()
        rows = (tmp_path / "light.csv").read_text().strip().splitlines()
        assert len(rows) == 8  # header + 7

    def test_channel_below_warmup_is_still_flushed_on_close(self, tmp_path):
        """A short recording must not produce an empty file."""
        sink = CSVSink(tmp_path, warmup=1000)
        sink.handle(sample("light", lumens=900.0))
        sink.close()
        rows = (tmp_path / "light.csv").read_text().strip().splitlines()
        assert len(rows) == 2

    def test_late_column_value_is_preserved_not_dropped(self, tmp_path):
        """A silently missing column would misrepresent the recording."""
        sink = CSVSink(tmp_path, warmup=1)
        sink.handle(sample("motion", accel=[1, 2, 3]))
        sink.handle(sample("motion", accel=[1, 2, 3], mag=[9, 8, 7]))
        sink.close()

        assert "mag.0" in sink.late_columns["motion"]
        text = (tmp_path / "motion.csv").read_text()
        assert CSVSink.EXTRA_COLUMN in text.splitlines()[0]
        # The value itself must still be in the file.
        assert "mag.0" in text and "9" in text

    def test_extra_json_is_valid_json(self, tmp_path):
        sink = CSVSink(tmp_path, warmup=1)
        sink.handle(sample("motion", accel=[1]))
        sink.handle(sample("motion", accel=[1], mag=[5]))
        sink.close()

        import csv as csv_module

        with (tmp_path / "motion.csv").open() as handle:
            rows = list(csv_module.DictReader(handle))
        extra = json.loads(rows[-1][CSVSink.EXTRA_COLUMN])
        assert extra["mag.0"] == 5


class TestBridge:
    def test_fans_out_to_every_sink(self):
        seen_a, seen_b = [], []
        bridge = Bridge()
        bridge.add(CallbackSink(seen_a.append, "a"))
        bridge.add(CallbackSink(seen_b.append, "b"))
        bridge.handle(sample())
        assert len(seen_a) == 1 and len(seen_b) == 1

    def test_one_failing_sink_does_not_starve_others(self):
        """A consumer's bug must not take the bridge down."""
        seen = []

        def explode(_: Sample) -> None:
            raise RuntimeError("consumer bug")

        bridge = Bridge()
        broken = bridge.add(CallbackSink(explode, "broken"))
        bridge.add(CallbackSink(seen.append, "healthy"))
        bridge.handle(sample())

        assert len(seen) == 1
        assert broken.stats.errors == 1

    def test_tracks_channels_and_counts(self):
        bridge = Bridge()
        bridge.add(CallbackSink(lambda s: None))
        bridge.handle(sample("motion"))
        bridge.handle(sample("ar.face"))
        assert bridge.samples == 2
        assert bridge.channels == {"motion", "ar.face"}

    def test_summary_mentions_channels_and_delivery(self):
        bridge = Bridge()
        bridge.add(CallbackSink(lambda s: None, "sink1"))
        bridge.handle(sample("motion"))
        summary = bridge.summary()
        assert "motion" in summary and "sink1=1" in summary

    def test_close_is_safe_with_mixed_sinks(self, tmp_path):
        bridge = Bridge()
        bridge.add(JSONLSink(tmp_path / "a.jsonl"))
        bridge.add(UDPJSONSink("127.0.0.1", 9))
        bridge.add(CallbackSink(lambda s: None))
        bridge.handle(sample())
        bridge.close()
