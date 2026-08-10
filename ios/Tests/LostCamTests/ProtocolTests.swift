import XCTest
@testable import LostCam

/// Tests for the wire format: request parsing, MJPEG framing, JSON writing and
/// the maths. These are the bugs that are painful to find on a device, so they
/// are the ones covered here.
final class HTTPRequestTests: XCTestCase {
    func testParsesSimpleRequest() {
        let request = HTTPRequest.parse("GET /video HTTP/1.1\r\nHost: phone:4747")
        XCTAssertEqual(request?.method, "GET")
        XCTAssertEqual(request?.path, "/video")
        XCTAssertEqual(request?.headers["host"], "phone:4747")
    }

    func testHeaderNamesAreCaseInsensitive() {
        let request = HTTPRequest.parse("GET / HTTP/1.1\r\nX-LostCam-Token: abc")
        XCTAssertEqual(request?.headers["x-lostcam-token"], "abc")
        XCTAssertEqual(request?.token, "abc")
    }

    func testParsesQueryParameters() {
        let request = HTTPRequest.parse("GET /video?w=1280&h=720&q=70 HTTP/1.1")
        XCTAssertEqual(request?.path, "/video")
        XCTAssertEqual(request?.query["w"], "1280")
        XCTAssertEqual(request?.query["h"], "720")
        XCTAssertEqual(request?.query["q"], "70")
    }

    func testTokenFromQueryTakesPrecedence() {
        let request = HTTPRequest.parse("GET /data?token=fromquery HTTP/1.1")
        XCTAssertEqual(request?.token, "fromquery")
    }

    func testPercentDecodesQueryValues() {
        let request = HTTPRequest.parse("GET /data?ch=ar.face%2Cattitude HTTP/1.1")
        XCTAssertEqual(request?.query["ch"], "ar.face,attitude")
    }

    func testTrailingSlashIsNormalised() {
        XCTAssertEqual(HTTPRequest.parse("GET /video/ HTTP/1.1")?.path, "/video")
    }

    func testRootPathIsPreserved() {
        XCTAssertEqual(HTTPRequest.parse("GET / HTTP/1.1")?.path, "/")
    }

    func testToleratesBareNewlines() {
        let request = HTTPRequest.parse("GET /info HTTP/1.1\nHost: x")
        XCTAssertEqual(request?.path, "/info")
        XCTAssertEqual(request?.headers["host"], "x")
    }

    func testRejectsGarbage() {
        XCTAssertNil(HTTPRequest.parse(""))
        XCTAssertNil(HTTPRequest.parse("nonsense"))
    }

    func testIntClampsIntoRange() {
        let request = HTTPRequest.parse("GET /data?hz=9999 HTTP/1.1")!
        XCTAssertEqual(request.int("hz", default: 60, min: 1, max: 240), 240)
    }

    func testIntFallsBackWhenUnparseable() {
        let request = HTTPRequest.parse("GET /data?hz=fast HTTP/1.1")!
        XCTAssertEqual(request.int("hz", default: 60, min: 1, max: 240), 60)
    }

    func testRoutingCoversEveryEndpoint() {
        XCTAssertEqual(LostCamServer.route(for: "/"), .status)
        XCTAssertEqual(LostCamServer.route(for: "/info"), .info)
        XCTAssertEqual(LostCamServer.route(for: "/video"), .video)
        XCTAssertEqual(LostCamServer.route(for: "/audio"), .audio)
        XCTAssertEqual(LostCamServer.route(for: "/data"), .data)
        XCTAssertEqual(LostCamServer.route(for: "/depth"), .depth)
        XCTAssertEqual(LostCamServer.route(for: "/nope"), .notFound)
    }
}

final class MJPEGFramingTests: XCTestCase {
    func testStreamContentTypeCarriesBoundary() {
        XCTAssertEqual(MJPEGFraming.streamContentType(),
                       "multipart/x-mixed-replace; boundary=lostcamframe")
    }

    func testPartHeaderIncludesLengthAndTimestamp() {
        let header = String(decoding: MJPEGFraming.partHeader(byteCount: 1234,
                                                             timestampMillis: 5678),
                           as: UTF8.self)
        XCTAssertTrue(header.hasPrefix("--lostcamframe\r\n"))
        XCTAssertTrue(header.contains("Content-Length: 1234\r\n"))
        XCTAssertTrue(header.contains("X-LostCam-Timestamp: 5678\r\n"))
        XCTAssertTrue(header.hasSuffix("\r\n\r\n"))
    }

    func testPartWrapsPayloadWithTrailingCRLF() {
        let jpeg = Data([0xFF, 0xD8, 0x01, 0xFF, 0xD9])
        let part = MJPEGFraming.part(jpeg, timestampMillis: 1)
        XCTAssertTrue(part.suffix(2).elementsEqual(Data("\r\n".utf8)))
        // The declared length must match the payload actually sent, or every
        // consumer using Content-Length desynchronises.
        let header = String(decoding: part.prefix(120), as: UTF8.self)
        XCTAssertTrue(header.contains("Content-Length: 5"))
    }

    func testDepthHeaderDescribesRasterAndIntrinsics() {
        let header = String(decoding: MJPEGFraming.depthPartHeader(
            byteCount: 320 * 240 * 2, width: 320, height: 240,
            timestampMillis: 42, intrinsics: (100, 101, 160, 120)),
                            as: UTF8.self)
        XCTAssertTrue(header.contains("X-LostCam-Depth: 320x240; format=u16mm"))
        XCTAssertTrue(header.contains("X-LostCam-Intrinsics: 100.0,101.0,160.0,120.0"))
        XCTAssertTrue(header.contains("Content-Length: 153600"))
    }
}

final class HTTPResponseTests: XCTestCase {
    func testStreamHeadHasNoContentLength() {
        let head = String(decoding: HTTPResponse.streamHead(contentType: "x/y"),
                          as: UTF8.self)
        XCTAssertTrue(head.hasPrefix("HTTP/1.1 200 OK\r\n"))
        XCTAssertFalse(head.contains("Content-Length"))
        XCTAssertTrue(head.contains("Cache-Control: no-store"))
    }

    func testTextResponseLengthMatchesBody() {
        let response = String(decoding: HTTPResponse.text(status: 404,
                                                          reason: "Not Found",
                                                          body: "nope"),
                              as: UTF8.self)
        XCTAssertTrue(response.contains("Content-Length: 4"))
        XCTAssertTrue(response.hasSuffix("nope"))
    }

    func testUnauthorizedIs401() {
        let response = String(decoding: HTTPResponse.unauthorized, as: UTF8.self)
        XCTAssertTrue(response.hasPrefix("HTTP/1.1 401 Unauthorized"))
    }
}

final class JSONWriterTests: XCTestCase {
    func testWritesFlatObject() {
        var writer = JSONWriter()
        writer.add("a", 1)
        writer.add("b", "two")
        writer.add("c", true)
        XCTAssertEqual(writer.jsonString, #"{"a":1,"b":"two","c":true}"#)
    }

    func testFinishAppendsNewlineForNDJSON() {
        var writer = JSONWriter()
        writer.add("a", 1)
        XCTAssertEqual(String(decoding: writer.finish(), as: UTF8.self), "{\"a\":1}\n")
    }

    func testFloatArrayIsWrittenInOrder() {
        var writer = JSONWriter()
        writer.add("q", floats: [0.5, -0.25], decimals: 3)
        XCTAssertEqual(writer.jsonString, #"{"q":[0.500,-0.250]}"#)
    }

    func testDictionaryKeysAreSorted() {
        var writer = JSONWriter()
        writer.add("blend", dictionary: ["jawOpen": 0.5, "browInnerUp": 0.25])
        XCTAssertEqual(writer.jsonString,
                       #"{"blend":{"browInnerUp":0.2500,"jawOpen":0.5000}}"#)
    }

    func testEscapesQuotesAndControlCharacters() {
        var writer = JSONWriter()
        writer.add("device", "a \"quoted\"\nname")
        XCTAssertTrue(writer.jsonString.contains(#"a \"quoted\"\nname"#))
    }

    func testNonFiniteNumbersBecomeZeroNotInvalidJSON() {
        // JSON has no NaN or Infinity; emitting them would produce a document no
        // consumer can parse.
        XCTAssertEqual(JSONWriter.number(Double.nan, decimals: 3), "0")
        XCTAssertEqual(JSONWriter.number(Double.infinity, decimals: 3), "0")
    }

    func testOutputParsesAsJSON() throws {
        var writer = JSONWriter()
        writer.add("t", 1000)
        writer.add("ch", "ar.face")
        writer.add("blend", dictionary: ["jawOpen": 0.42])
        writer.add("pose", floats: Array(repeating: 1.0, count: 16))
        let parsed = try JSONSerialization.jsonObject(
            with: writer.finish()) as? [String: Any]
        XCTAssertEqual(parsed?["ch"] as? String, "ar.face")
        XCTAssertEqual((parsed?["pose"] as? [Any])?.count, 16)
    }
}

final class MathsTests: XCTestCase {
    func testIdentityQuaternionIsLevel() {
        let angles = Maths.euler(x: 0, y: 0, z: 0, w: 1)
        XCTAssertEqual(angles.pitch, 0, accuracy: 1e-6)
        XCTAssertEqual(angles.yaw, 0, accuracy: 1e-6)
        XCTAssertEqual(angles.roll, 0, accuracy: 1e-6)
    }

    func testNinetyDegreesAboutYIsYaw() {
        let half = (Double.pi / 2) / 2
        let angles = Maths.euler(x: 0, y: sin(half), z: 0, w: cos(half))
        XCTAssertEqual(angles.yaw, 90, accuracy: 1e-4)
    }

    func testGimbalLockClampsInsteadOfNaN() {
        let half = (Double.pi / 2) / 2
        let angles = Maths.euler(x: 0, y: sin(half), z: 0, w: cos(half))
        XCTAssertFalse(angles.pitch.isNaN)
        XCTAssertFalse(angles.yaw.isNaN)
        XCTAssertFalse(angles.roll.isNaN)
    }

    func testGimbalLockResolvesToTheSharedConvention() {
        // At yaw = ±90 only pitch+roll is determined, so the convention is
        // pinned: roll is 0 and the remainder goes into pitch. The desktop
        // client and the Android sender must produce the same numbers.
        let half = (Double.pi / 2) / 2
        let angles = Maths.euler(x: 0, y: sin(half), z: 0, w: cos(half))
        XCTAssertEqual(angles.pitch, 0, accuracy: 1e-6)
        XCTAssertEqual(angles.yaw, 90, accuracy: 1e-3)
        XCTAssertEqual(angles.roll, 0, accuracy: 1e-12)
    }

    func testGimbalLockIsStableEitherSideOfTheDegenerateTerm() {
        // 1 - 2(x² + y²) straddles zero depending on the platform's sin, which
        // is what made the equivalent Python test pass on Linux and fail on
        // Windows.
        for value in [0.7071067811865475, 0.7071067811865476] {
            let angles = Maths.euler(x: 0, y: value, z: 0, w: value)
            XCTAssertEqual(angles.pitch, 0, accuracy: 1e-6)
            XCTAssertEqual(angles.yaw, 90, accuracy: 1e-3)
        }
    }

    func testJustBelowThePoleUsesTheGeneralFormula() {
        let half = (80.0 * Double.pi / 180.0) / 2
        let angles = Maths.euler(x: 0, y: sin(half), z: 0, w: cos(half))
        XCTAssertEqual(angles.yaw, 80, accuracy: 1e-3)
    }

    func testDepthConversionToMillimetres() {
        XCTAssertEqual(Maths.depthMillimetres(metres: 1.0), 1000)
        // Deliberately not a value that lands on a half-millimetre: 0.2345 m is
        // not representable in Float and comes out a hair *above* 234.5 mm, so
        // asserting either 234 or 235 would be testing the binary
        // representation of the literal rather than the conversion. The contract
        // is "nearest millimetre", so the test uses values that have one.
        XCTAssertEqual(Maths.depthMillimetres(metres: 0.2344), 234)
        XCTAssertEqual(Maths.depthMillimetres(metres: 0.2346), 235)
    }

    func testDepthZeroMeansNoMeasurement() {
        XCTAssertEqual(Maths.depthMillimetres(metres: 0), 0)
        XCTAssertEqual(Maths.depthMillimetres(metres: -1), 0)
        XCTAssertEqual(Maths.depthMillimetres(metres: .nan), 0)
        XCTAssertEqual(Maths.depthMillimetres(metres: .infinity), 0)
    }

    func testDepthBeyondRangeReportsNoMeasurementRatherThanWrapping() {
        // 100 m exceeds u16 millimetres; a wrapped value would read as a
        // believable short distance, which is worse than admitting nothing.
        XCTAssertEqual(Maths.depthMillimetres(metres: 100), 0)
    }
}
