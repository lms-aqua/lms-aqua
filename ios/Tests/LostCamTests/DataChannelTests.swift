import XCTest
@testable import LostCam

/// Data channel tests, focused on the invariants the desktop client relies on:
/// contiguous per-subscriber sequence numbers, sparse blendshapes, honest
/// channel filtering and rate capping.
final class SampleEncoderTests: XCTestCase {
    private func encoder() -> SampleEncoder {
        // Fixed clock so records are byte-comparable.
        SampleEncoder(clock: { 1000 })
    }

    private func decode(_ data: Data) throws -> [String: Any] {
        let object = try JSONSerialization.jsonObject(with: data)
        return try XCTUnwrap(object as? [String: Any])
    }

    func testEveryRecordCarriesRequiredFields() throws {
        let record = try decode(encoder().encode(.battery(
            BatterySample(level: 0.5, charging: false, thermal: "nominal"))))
        XCTAssertEqual(record["t"] as? Int, 1000)
        XCTAssertEqual(record["seq"] as? Int, 1)
        XCTAssertEqual(record["ch"] as? String, "battery")
    }

    func testSequenceIncrementsPerRecord() throws {
        let subject = encoder()
        let sample = SensorSample.battery(
            BatterySample(level: 1, charging: true, thermal: "fair"))
        let first = try decode(subject.encode(sample))
        let second = try decode(subject.encode(sample))
        let third = try decode(subject.encode(sample))
        XCTAssertEqual(first["seq"] as? Int, 1)
        XCTAssertEqual(second["seq"] as? Int, 2)
        XCTAssertEqual(third["seq"] as? Int, 3)
    }

    func testEachEncoderNumbersIndependently() throws {
        // Two subscribers must each see a contiguous stream, which is why they
        // do not share a counter.
        let a = encoder()
        let b = encoder()
        let sample = SensorSample.light(LightSample(lumens: 900, kelvin: nil))
        _ = a.encode(sample)
        _ = a.encode(sample)
        let fromB = try decode(b.encode(sample))
        XCTAssertEqual(fromB["seq"] as? Int, 1)
    }

    func testAttitudeIncludesQuaternionAndEuler() throws {
        let half = (Double.pi / 2) / 2
        let record = try decode(encoder().encode(.attitude(AttitudeSample(
            x: 0, y: sin(half), z: 0, w: cos(half),
            reference: "magnetic", accuracy: "high"))))
        let quaternion = try XCTUnwrap(record["q"] as? [Double])
        XCTAssertEqual(quaternion.count, 4)
        // Scalar last, per the spec.
        XCTAssertEqual(quaternion[3], cos(half), accuracy: 1e-5)
        let euler = try XCTUnwrap(record["euler"] as? [Double])
        XCTAssertEqual(euler[1], 90, accuracy: 1e-2)
        XCTAssertEqual(record["ref"] as? String, "magnetic")
    }

    func testWorldPoseIsSixteenElements() throws {
        let record = try decode(encoder().encode(.world(WorldSample(
            pose: Array(repeating: 0.5, count: 16), state: "normal", reason: nil,
            features: 800, intrinsics: [1, 2, 3, 4], resolution: [1920, 1080]))))
        XCTAssertEqual((record["pose"] as? [Double])?.count, 16)
        XCTAssertEqual(record["state"] as? String, "normal")
        XCTAssertEqual(record["features"] as? Int, 800)
        XCTAssertNil(record["reason"], "absent optional fields must be omitted")
    }

    func testFaceOmitsAbsentOptionalFields() throws {
        let record = try decode(encoder().encode(.face(FaceSample(
            tracked: true, blendShapes: ["jawOpen": 0.4], transform: nil,
            leftEye: nil, rightEye: nil, look: nil))))
        XCTAssertEqual(record["tracked"] as? Bool, true)
        XCTAssertNotNil(record["blend"])
        XCTAssertNil(record["transform"])
        XCTAssertNil(record["leftEye"])
    }

    func testBlendShapeCoefficientsSurviveRoundTrip() throws {
        let record = try decode(encoder().encode(.face(FaceSample(
            tracked: true, blendShapes: ["jawOpen": 0.4213, "eyeBlinkLeft": 0.0625],
            transform: nil, leftEye: nil, rightEye: nil, look: nil))))
        let blend = try XCTUnwrap(record["blend"] as? [String: Double])
        XCTAssertEqual(blend["jawOpen"] ?? 0, 0.4213, accuracy: 1e-4)
        XCTAssertEqual(blend["eyeBlinkLeft"] ?? 0, 0.0625, accuracy: 1e-4)
    }

    func testRemovedPlaneCarriesOnlyIdentityAndEvent() throws {
        let record = try decode(encoder().encode(.plane(PlaneSample(
            id: "abc", event: "removed", center: [1, 2, 3], extent: [1, 1],
            alignment: "horizontal", classification: "table"))))
        XCTAssertEqual(record["event"] as? String, "removed")
        XCTAssertEqual(record["id"] as? String, "abc")
        // There is no geometry left to describe, so none is sent.
        XCTAssertNil(record["center"])
        XCTAssertNil(record["extent"])
    }

    func testChannelNamesMatchTheSpec() throws {
        let cases: [(SensorSample, String)] = [
            (.attitude(AttitudeSample(x: 0, y: 0, z: 0, w: 1,
                                      reference: "arbitrary", accuracy: nil)),
             "attitude"),
            (.motion(MotionSample(userAcceleration: [0, 0, 0], gravity: [0, 0, -1],
                                  rotationRate: [0, 0, 0], magneticField: nil,
                                  magneticAccuracy: nil)), "motion"),
            (.world(WorldSample(pose: Array(repeating: 0, count: 16),
                                state: "normal", reason: nil, features: nil,
                                intrinsics: nil, resolution: nil)), "ar.world"),
            (.face(FaceSample(tracked: true, blendShapes: [:], transform: nil,
                              leftEye: nil, rightEye: nil, look: nil)), "ar.face"),
            (.plane(PlaneSample(id: "x", event: "added", center: nil, extent: nil,
                                alignment: nil, classification: nil)), "ar.planes"),
            (.light(LightSample(lumens: 1, kelvin: nil)), "light"),
            (.barometer(BarometerSample(kilopascals: 101, relativeAltitude: nil)),
             "barometer"),
            (.battery(BatterySample(level: 1, charging: false, thermal: "nominal")),
             "battery"),
            (.location(LocationSample(latitude: 1, longitude: 2, accuracy: 3,
                                      altitude: nil, speed: nil, heading: nil)),
             "location"),
        ]
        for (sample, expected) in cases {
            XCTAssertEqual(sample.channel.rawValue, expected)
            let record = try decode(encoder().encode(sample))
            XCTAssertEqual(record["ch"] as? String, expected)
        }
    }

    func testLocationPrecisionIsPreserved() throws {
        let record = try decode(encoder().encode(.location(LocationSample(
            latitude: 51.5074123, longitude: -0.1278456, accuracy: 5,
            altitude: nil, speed: nil, heading: nil))))
        // Seven decimals is roughly a centimetre; fewer would quietly degrade it.
        XCTAssertEqual(record["lat"] as? Double ?? 0, 51.5074123, accuracy: 1e-7)
        XCTAssertEqual(record["lon"] as? Double ?? 0, -0.1278456, accuracy: 1e-7)
    }
}

final class ChannelFilterTests: XCTestCase {
    func testEmptyFilterMeansEverything() {
        XCTAssertTrue(parseChannelFilter(nil).isEmpty)
        XCTAssertTrue(parseChannelFilter("").isEmpty)
    }

    func testParsesCommaSeparatedList() {
        XCTAssertEqual(parseChannelFilter("ar.face,attitude"),
                       Set(["ar.face", "attitude"]))
    }

    func testTrimsWhitespaceAndIgnoresBlanks() {
        XCTAssertEqual(parseChannelFilter(" ar.face , , attitude "),
                       Set(["ar.face", "attitude"]))
    }
}

final class DataSubscriberTests: XCTestCase {
    private func subscriber(channels: Set<String>, hz: Int,
                            into sink: @escaping (Data) -> Void) -> DataSubscriber {
        DataSubscriber(channels: channels, hz: hz,
                       encoder: SampleEncoder(clock: { 0 }), send: sink)
    }

    private let batterySample = SensorSample.battery(
        BatterySample(level: 1, charging: false, thermal: "nominal"))
    private let lightSample = SensorSample.light(
        LightSample(lumens: 900, kelvin: nil))

    func testAcceptsEverythingWhenUnfiltered() {
        let subject = subscriber(channels: [], hz: 0) { _ in }
        XCTAssertTrue(subject.accepts(channel: "battery"))
        XCTAssertTrue(subject.accepts(channel: "anything.new"))
    }

    func testRejectsChannelsOutsideTheFilter() {
        let subject = subscriber(channels: ["battery"], hz: 0) { _ in }
        XCTAssertTrue(subject.accepts(channel: "battery"))
        XCTAssertFalse(subject.accepts(channel: "light"))
    }

    func testFilteredSubscriberStillSeesContiguousSequence() throws {
        // The important one: a channel subset must not look like packet loss.
        var lines: [Data] = []
        let subject = subscriber(channels: ["battery"], hz: 0) { lines.append($0) }
        for index in 0..<5 {
            subject.offer(batterySample, nowMillis: index)
            subject.offer(lightSample, nowMillis: index)
        }
        XCTAssertEqual(lines.count, 5, "only the requested channel should be sent")

        let sequences = try lines.map { line -> Int in
            let record = try JSONSerialization.jsonObject(with: line) as? [String: Any]
            return record?["seq"] as? Int ?? -1
        }
        XCTAssertEqual(sequences, [1, 2, 3, 4, 5])
    }

    func testRateCapDecimates() {
        var count = 0
        // 10 Hz means one sample per 100 ms.
        let subject = subscriber(channels: [], hz: 10) { _ in count += 1 }
        for millis in stride(from: 0, to: 100, by: 10) {
            subject.offer(batterySample, nowMillis: millis)
        }
        XCTAssertEqual(count, 1, "ten offers inside one interval should emit once")
    }

    func testRateCapIsPerChannel() {
        var count = 0
        let subject = subscriber(channels: [], hz: 10) { _ in count += 1 }
        subject.offer(batterySample, nowMillis: 0)
        subject.offer(lightSample, nowMillis: 0)
        // Different channels do not compete for the same budget.
        XCTAssertEqual(count, 2)
    }

    func testUncappedRateEmitsEverything() {
        var count = 0
        let subject = subscriber(channels: [], hz: 0) { _ in count += 1 }
        for millis in 0..<10 {
            subject.offer(batterySample, nowMillis: millis)
        }
        XCTAssertEqual(count, 10)
    }
}

final class RateLimiterTests: XCTestCase {
    func testZeroHzMeansNoLimit() {
        var limiter = RateLimiter(hz: 0)
        XCTAssertTrue(limiter.shouldEmit(channel: "a", nowMillis: 0))
        XCTAssertTrue(limiter.shouldEmit(channel: "a", nowMillis: 0))
    }

    func testEmitsAgainAfterTheInterval() {
        var limiter = RateLimiter(hz: 10)
        XCTAssertTrue(limiter.shouldEmit(channel: "a", nowMillis: 0))
        XCTAssertFalse(limiter.shouldEmit(channel: "a", nowMillis: 50))
        XCTAssertTrue(limiter.shouldEmit(channel: "a", nowMillis: 100))
    }

    func testVeryHighRateStillHasAMinimumInterval() {
        // 1000/2000 would floor to zero; it must not become a divide-by-nothing.
        let limiter = RateLimiter(hz: 2000)
        XCTAssertGreaterThanOrEqual(limiter.intervalMillis, 1)
    }
}

final class ChannelMetadataTests: XCTestCase {
    func testARChannelsAreMarkedAsNeedingASession() {
        XCTAssertTrue(Channel.arWorld.requiresARSession)
        XCTAssertTrue(Channel.arFace.requiresARSession)
        XCTAssertTrue(Channel.arFace.requiresFaceTracking)
        XCTAssertFalse(Channel.motion.requiresARSession)
    }

    func testLocationIsTheSensitiveChannel() {
        XCTAssertTrue(Channel.location.isSensitive)
        for channel in Channel.allCases where channel != .location {
            XCTAssertFalse(channel.isSensitive, "\(channel.rawValue) marked sensitive")
        }
    }

    func testEveryChannelHasHumanReadableText() {
        for channel in Channel.allCases {
            XCTAssertFalse(channel.label.isEmpty)
            XCTAssertFalse(channel.detail.isEmpty)
        }
    }
}
