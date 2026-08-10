"""Data channel parsing and maths tests."""

from __future__ import annotations

import json
import math

import pytest

from lostcam.datastream import (
    DataStats,
    NDJSONParser,
    Sample,
    pose_translation,
    quaternion_to_euler,
    to_sample,
)

IDENTITY_POSE = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.5, 1.4, -0.3, 1]


def line(**fields: object) -> bytes:
    return json.dumps(fields).encode("utf-8") + b"\n"


class TestNDJSONParser:
    def test_single_record(self):
        parser = NDJSONParser()
        assert parser.feed(line(t=1, seq=1, ch="attitude")) == [
            {"t": 1, "seq": 1, "ch": "attitude"}
        ]

    def test_several_records_in_one_read(self):
        parser = NDJSONParser()
        stream = line(t=1, seq=1, ch="a") + line(t=2, seq=2, ch="b")
        assert [r["ch"] for r in parser.feed(stream)] == ["a", "b"]

    def test_record_split_across_reads(self):
        stream = line(t=1, seq=1, ch="ar.face", blend={"jawOpen": 0.5})
        for split in range(1, len(stream)):
            parser = NDJSONParser()
            got = parser.feed(stream[:split]) + parser.feed(stream[split:])
            assert len(got) == 1
            assert got[0]["blend"]["jawOpen"] == 0.5

    def test_incomplete_line_is_withheld(self):
        parser = NDJSONParser()
        assert parser.feed(b'{"t":1,"seq":1,"ch":"x"') == []
        assert parser.feed(b"}\n") == [{"t": 1, "seq": 1, "ch": "x"}]

    def test_bad_line_is_skipped_not_fatal(self):
        """A corrupt line must not drop the connection."""
        parser = NDJSONParser()
        stream = b"{not json}\n" + line(t=2, seq=2, ch="ok")
        records = parser.feed(stream)
        assert [r["ch"] for r in records] == ["ok"]
        assert parser.bad_lines == 1

    def test_non_object_line_is_rejected(self):
        parser = NDJSONParser()
        assert parser.feed(b"[1,2,3]\n") == []
        assert parser.bad_lines == 1

    def test_blank_lines_ignored_without_counting_as_bad(self):
        parser = NDJSONParser()
        assert parser.feed(b"\n\n  \n") == []
        assert parser.bad_lines == 0

    def test_unterminated_giant_line_is_dropped(self):
        parser = NDJSONParser(max_line_bytes=256)
        parser.feed(b'{"pad":"' + b"x" * 4096)
        assert parser.buffered == 0
        assert parser.bad_lines == 1

    def test_unicode_survives(self):
        parser = NDJSONParser()
        (record,) = parser.feed(line(t=1, seq=1, ch="x", device="iPhone – Pat's"))
        assert record["device"] == "iPhone – Pat's"


class TestToSample:
    def test_builds_sample(self):
        sample = to_sample({"t": 100, "seq": 7, "ch": "motion", "accel": [0, 0, 1]})
        assert sample is not None
        assert (sample.t, sample.seq, sample.channel) == (100, 7, "motion")
        assert sample.get("accel") == [0, 0, 1]

    def test_keeps_unknown_fields(self):
        """Unknown fields must survive so a bridge can forward them."""
        sample = to_sample({"t": 1, "seq": 1, "ch": "future", "somethingNew": 42})
        assert sample is not None and sample.raw["somethingNew"] == 42

    def test_float_timestamp_is_coerced(self):
        sample = to_sample({"t": 100.7, "seq": 1, "ch": "x"})
        assert sample is not None and sample.t == 100

    def test_missing_seq_defaults_to_zero(self):
        sample = to_sample({"t": 1, "ch": "x"})
        assert sample is not None and sample.seq == 0

    @pytest.mark.parametrize(
        "record",
        [
            {},
            {"t": 1, "seq": 1},
            {"t": 1, "seq": 1, "ch": ""},
            {"t": 1, "seq": 1, "ch": 5},
            {"seq": 1, "ch": "x"},
            {"t": "soon", "seq": 1, "ch": "x"},
            {"t": True, "seq": 1, "ch": "x"},
        ],
    )
    def test_rejects_malformed(self, record):
        assert to_sample(record) is None

    def test_is_ar_flag(self):
        assert to_sample({"t": 1, "seq": 1, "ch": "ar.face"}).is_ar
        assert not to_sample({"t": 1, "seq": 1, "ch": "motion"}).is_ar


class TestFlatten:
    def test_flattens_nested_dict(self):
        sample = Sample(1, 1, "ar.face", {"blend": {"jawOpen": 0.4}})
        assert sample.flatten()["blend.jawOpen"] == 0.4

    def test_flattens_list_by_index(self):
        sample = Sample(1, 1, "attitude", {"q": [0.1, 0.2, 0.3, 0.4]})
        flat = sample.flatten()
        assert flat["q.0"] == 0.1
        assert flat["q.3"] == 0.4

    def test_flattens_pose_translation_indices(self):
        sample = Sample(1, 1, "ar.world", {"pose": IDENTITY_POSE})
        flat = sample.flatten()
        assert flat["pose.12"] == 0.5
        assert flat["pose.14"] == -0.3

    def test_keeps_scalars_and_bools(self):
        sample = Sample(1, 1, "battery", {"level": 0.8, "charging": False, "th": "ok"})
        flat = sample.flatten()
        assert flat["level"] == 0.8
        assert flat["charging"] is False
        assert flat["th"] == "ok"

    def test_deeply_nested(self):
        sample = Sample(1, 1, "x", {"a": {"b": {"c": [1, 2]}}})
        flat = sample.flatten()
        assert flat["a.b.c.0"] == 1


class TestDataStats:
    def test_counts_per_channel(self):
        stats = DataStats()
        stats.observe(Sample(1, 1, "motion", {}))
        stats.observe(Sample(2, 2, "motion", {}))
        stats.observe(Sample(3, 3, "ar.face", {}))
        assert stats.samples == 3
        assert stats.per_channel == {"motion": 2, "ar.face": 1}

    def test_detects_seq_gap_as_loss(self):
        stats = DataStats()
        stats.observe(Sample(1, 1, "x", {}))
        stats.observe(Sample(2, 5, "x", {}))  # 2,3,4 missing
        assert stats.dropped == 3

    def test_no_loss_when_contiguous(self):
        stats = DataStats()
        for seq in range(1, 20):
            stats.observe(Sample(seq, seq, "x", {}))
        assert stats.dropped == 0

    def test_zero_seq_does_not_trigger_loss(self):
        stats = DataStats()
        stats.observe(Sample(1, 0, "x", {}))
        stats.observe(Sample(2, 0, "x", {}))
        assert stats.dropped == 0


class TestQuaternionToEuler:
    def test_identity_is_level(self):
        pitch, yaw, roll = quaternion_to_euler([0, 0, 0, 1])
        assert (pitch, yaw, roll) == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)

    def test_90_degrees_about_y_is_yaw(self):
        """A 90° yaw is exactly gimbal lock, so the convention is pinned.

        This test previously passed on Linux and failed on Windows: pitch is
        mathematically ambiguous here, and the naive formula returned 0° or 180°
        depending on the platform's ``sin``. The implementation now defines roll
        as 0 and folds the remainder into pitch, which for this rotation gives 0.
        """
        half = math.radians(90) / 2
        pitch, yaw, roll = quaternion_to_euler([0, math.sin(half), 0, math.cos(half)])
        assert yaw == pytest.approx(90.0, abs=1e-4)
        assert pitch == pytest.approx(0.0, abs=1e-4)
        assert roll == pytest.approx(0.0, abs=1e-9)

    def test_gimbal_lock_is_deterministic_either_side_of_zero(self):
        """The exact case that differed between Linux and Windows.

        ``1 - 2(x² + y²)`` lands on either side of zero depending on the last bit
        of ``sin``. Both signs must produce the same answer.
        """
        import math as m

        for y in (m.sin(m.radians(45)), 0.7071067811865475, 0.7071067811865476):
            pitch, yaw, roll = quaternion_to_euler([0.0, y, 0.0, y])
            assert yaw == pytest.approx(90.0, abs=1e-3), f"y={y!r}"
            assert pitch == pytest.approx(0.0, abs=1e-6), f"y={y!r} gave pitch={pitch}"
            assert roll == 0.0

    def test_negative_pole_is_also_deterministic(self):
        half = math.radians(-90) / 2
        pitch, yaw, roll = quaternion_to_euler([0, math.sin(half), 0, math.cos(half)])
        assert yaw == pytest.approx(-90.0, abs=1e-3)
        assert pitch == pytest.approx(0.0, abs=1e-6)
        assert roll == 0.0

    def test_denormalised_quaternion_near_the_pole_does_not_nan(self):
        # A slightly over-unit quaternion would push asin past 1.
        for value in (0.7072, 0.71, 0.75):
            pitch, yaw, roll = quaternion_to_euler([0.0, value, 0.0, value])
            for angle in (pitch, yaw, roll):
                assert not math.isnan(angle), f"NaN for {value}"

    def test_just_below_the_pole_still_uses_the_general_formula(self):
        """A rotation near but not at the pole must not be snapped to the pole."""
        # 80° yaw: sin(yaw) = 0.985, below the threshold.
        half = math.radians(80) / 2
        _, yaw, _ = quaternion_to_euler([0, math.sin(half), 0, math.cos(half)])
        assert yaw == pytest.approx(80.0, abs=1e-3)

    def test_90_degrees_about_x_is_pitch(self):
        half = math.radians(90) / 2
        pitch, _, _ = quaternion_to_euler([math.sin(half), 0, 0, math.cos(half)])
        assert pitch == pytest.approx(90.0, abs=1e-4)

    def test_gimbal_lock_clamps_instead_of_nan(self):
        """Straight up must not produce a NaN."""
        half = math.radians(90) / 2
        pitch, yaw, roll = quaternion_to_euler([0, math.sin(half), 0, math.cos(half)])
        for value in (pitch, yaw, roll):
            assert not math.isnan(value)

    def test_slightly_denormalised_input_still_works(self):
        pitch, yaw, roll = quaternion_to_euler([0, 0.7072, 0, 0.7072])
        assert not math.isnan(yaw)

    def test_wrong_length_rejected(self):
        with pytest.raises(ValueError, match="4 components"):
            quaternion_to_euler([0, 0, 1])


class TestPoseTranslation:
    def test_reads_column_major_translation(self):
        """Elements 12,13,14 — the column-major convention from §6.2."""
        assert pose_translation(IDENTITY_POSE) == pytest.approx((0.5, 1.4, -0.3))

    def test_wrong_length_rejected(self):
        with pytest.raises(ValueError, match="16 elements"):
            pose_translation([1, 0, 0, 1])
