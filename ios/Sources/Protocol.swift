import Foundation

/// Wire-format helpers for docs/PROTOCOL.md.
///
/// Everything in this file is deliberately free of UIKit, ARKit and Network
/// imports, and free of side effects, so it can be unit tested on its own. The
/// framing and parsing bugs are the ones that cost hours to find on a device,
/// so they are the ones pushed into testable code.

// MARK: - Constants

enum LostCam {
    static let product = "LostCam"
    static let protocolVersion = 2
    static let defaultPort: UInt16 = 4747
    static let discoveryPort: UInt16 = 4748
    static let videoBoundary = "lostcamframe"
    static let depthBoundary = "lostcamdepth"
    static let discoveryProbe = "LOSTCAM_DISCOVER_V1"
}

// MARK: - HTTP request parsing

/// A parsed HTTP request line plus headers.
struct HTTPRequest: Equatable {
    let method: String
    let path: String
    let query: [String: String]
    let headers: [String: String]

    /// Parse a request head (everything before the blank line).
    ///
    /// Returns nil for anything that is not a plausible HTTP request, rather
    /// than trapping — this is fed straight from a socket, so malformed input
    /// is expected traffic, not a programming error.
    static func parse(_ raw: String) -> HTTPRequest? {
        // Tolerate bare \n line endings as well as CRLF; some tools send them.
        let lines = raw.replacingOccurrences(of: "\r\n", with: "\n")
            .split(separator: "\n", omittingEmptySubsequences: false)
        guard let requestLine = lines.first, !requestLine.isEmpty else { return nil }

        let parts = requestLine.split(separator: " ", omittingEmptySubsequences: true)
        guard parts.count >= 2 else { return nil }
        let method = String(parts[0]).uppercased()
        let target = String(parts[1])

        var path = target
        var query: [String: String] = [:]
        if let questionMark = target.firstIndex(of: "?") {
            path = String(target[target.startIndex..<questionMark])
            let queryString = String(target[target.index(after: questionMark)...])
            query = Self.parseQuery(queryString)
        }

        // Normalise a trailing slash so "/video/" and "/video" route alike,
        // while keeping "/" itself intact.
        if path.count > 1 && path.hasSuffix("/") {
            path.removeLast()
        }

        var headers: [String: String] = [:]
        for line in lines.dropFirst() {
            if line.isEmpty { break }
            guard let colon = line.firstIndex(of: ":") else { continue }
            let name = String(line[line.startIndex..<colon])
                .trimmingCharacters(in: .whitespaces)
                .lowercased()
            let value = String(line[line.index(after: colon)...])
                .trimmingCharacters(in: .whitespaces)
            guard !name.isEmpty else { continue }
            headers[name] = value
        }

        return HTTPRequest(method: method, path: path, query: query, headers: headers)
    }

    static func parseQuery(_ queryString: String) -> [String: String] {
        var query: [String: String] = [:]
        for pair in queryString.split(separator: "&", omittingEmptySubsequences: true) {
            let bits = pair.split(separator: "=", maxSplits: 1,
                                  omittingEmptySubsequences: false)
            let key = percentDecode(String(bits[0]))
            guard !key.isEmpty else { continue }
            let value = bits.count > 1 ? percentDecode(String(bits[1])) : ""
            query[key] = value
        }
        return query
    }

    private static func percentDecode(_ value: String) -> String {
        let plusesResolved = value.replacingOccurrences(of: "+", with: " ")
        return plusesResolved.removingPercentEncoding ?? plusesResolved
    }

    /// An integer query parameter, clamped into a sane range.
    func int(_ key: String, default fallback: Int, min lower: Int, max upper: Int) -> Int {
        guard let raw = query[key], let value = Int(raw) else { return fallback }
        return Swift.min(upper, Swift.max(lower, value))
    }

    /// The token supplied by query parameter or header, if any.
    var token: String? {
        query["token"] ?? headers["x-lostcam-token"]
    }
}

// MARK: - Responses

enum HTTPResponse {
    /// A complete, small response with a body.
    static func head(status: Int, reason: String, contentType: String,
                     contentLength: Int, extraHeaders: [String: String] = [:]) -> Data {
        var text = "HTTP/1.1 \(status) \(reason)\r\n"
        text += "Content-Type: \(contentType)\r\n"
        text += "Content-Length: \(contentLength)\r\n"
        for (name, value) in extraHeaders.sorted(by: { $0.key < $1.key }) {
            text += "\(name): \(value)\r\n"
        }
        text += "Connection: close\r\n\r\n"
        return Data(text.utf8)
    }

    /// A streaming response head: no Content-Length, the body never ends.
    static func streamHead(contentType: String,
                           extraHeaders: [String: String] = [:]) -> Data {
        var text = "HTTP/1.1 200 OK\r\n"
        text += "Content-Type: \(contentType)\r\n"
        text += "Cache-Control: no-store\r\n"
        text += "Pragma: no-cache\r\n"
        for (name, value) in extraHeaders.sorted(by: { $0.key < $1.key }) {
            text += "\(name): \(value)\r\n"
        }
        text += "Connection: close\r\n\r\n"
        return Data(text.utf8)
    }

    static func text(status: Int, reason: String, body: String,
                     contentType: String = "text/plain; charset=utf-8") -> Data {
        let payload = Data(body.utf8)
        var response = head(status: status, reason: reason, contentType: contentType,
                            contentLength: payload.count)
        response.append(payload)
        return response
    }

    static func json(status: Int = 200, reason: String = "OK", body: Data) -> Data {
        var response = head(status: status, reason: reason,
                            contentType: "application/json", contentLength: body.count)
        response.append(body)
        return response
    }

    static let unauthorized = text(status: 401, reason: "Unauthorized",
                                   body: "unauthorized: bad or missing token")
    static let notFound = text(status: 404, reason: "Not Found", body: "not found")
}

// MARK: - MJPEG framing

enum MJPEGFraming {
    static func streamContentType(boundary: String = LostCam.videoBoundary) -> String {
        "multipart/x-mixed-replace; boundary=\(boundary)"
    }

    /// The per-part header preceding each JPEG payload.
    ///
    /// Content-Length is always emitted: consumers are required to cope without
    /// it, but making them scan for markers when the length is known is rude.
    static func partHeader(byteCount: Int, timestampMillis: Int,
                           boundary: String = LostCam.videoBoundary) -> Data {
        var text = "--\(boundary)\r\n"
        text += "Content-Type: image/jpeg\r\n"
        text += "Content-Length: \(byteCount)\r\n"
        text += "X-LostCam-Timestamp: \(timestampMillis)\r\n\r\n"
        return Data(text.utf8)
    }

    /// A complete part: header, payload, then the CRLF the next boundary needs.
    static func part(_ jpeg: Data, timestampMillis: Int,
                     boundary: String = LostCam.videoBoundary) -> Data {
        var out = partHeader(byteCount: jpeg.count, timestampMillis: timestampMillis,
                             boundary: boundary)
        out.append(jpeg)
        out.append(Data("\r\n".utf8))
        return out
    }

    /// Depth part header, carrying the raster size and the depth intrinsics.
    static func depthPartHeader(byteCount: Int, width: Int, height: Int,
                                timestampMillis: Int,
                                intrinsics: (Float, Float, Float, Float)) -> Data {
        var text = "--\(LostCam.depthBoundary)\r\n"
        text += "Content-Type: application/octet-stream\r\n"
        text += "Content-Length: \(byteCount)\r\n"
        text += "X-LostCam-Timestamp: \(timestampMillis)\r\n"
        text += "X-LostCam-Depth: \(width)x\(height); format=u16mm\r\n"
        text += "X-LostCam-Intrinsics: \(intrinsics.0),\(intrinsics.1),"
        text += "\(intrinsics.2),\(intrinsics.3)\r\n\r\n"
        return Data(text.utf8)
    }
}

// MARK: - JSON writing

/// A tiny JSON writer used for the data channel.
///
/// JSONSerialization would work, but it allocates a dictionary per sample and
/// gives no control over float formatting; at 60 Hz across several channels
/// that churn is worth avoiding. It also lets numbers be written with a fixed
/// number of decimals, which keeps lines short.
struct JSONWriter {
    private var body = ""
    private var needsComma = false

    init() {
        body = "{"
    }

    mutating func add(_ key: String, _ value: Int) {
        prefix(key)
        body += String(value)
    }

    mutating func add(_ key: String, _ value: Bool) {
        prefix(key)
        body += value ? "true" : "false"
    }

    mutating func add(_ key: String, _ value: Double, decimals: Int = 5) {
        prefix(key)
        body += JSONWriter.number(value, decimals: decimals)
    }

    mutating func add(_ key: String, _ value: String) {
        prefix(key)
        body += "\"\(JSONWriter.escape(value))\""
    }

    mutating func add(_ key: String, floats: [Float], decimals: Int = 5) {
        prefix(key)
        body += "["
        for (index, value) in floats.enumerated() {
            if index > 0 { body += "," }
            body += JSONWriter.number(Double(value), decimals: decimals)
        }
        body += "]"
    }

    mutating func add(_ key: String, dictionary: [String: Float], decimals: Int = 4) {
        prefix(key)
        body += "{"
        // Sorted so a diff of two captured streams is readable.
        for (index, pair) in dictionary.sorted(by: { $0.key < $1.key }).enumerated() {
            if index > 0 { body += "," }
            body += "\"\(JSONWriter.escape(pair.key))\":"
            body += JSONWriter.number(Double(pair.value), decimals: decimals)
        }
        body += "}"
    }

    private mutating func prefix(_ key: String) {
        if needsComma { body += "," }
        body += "\"\(JSONWriter.escape(key))\":"
        needsComma = true
    }

    /// Finish the object and append the newline that makes it NDJSON.
    func finish() -> Data {
        Data((body + "}\n").utf8)
    }

    var jsonString: String { body + "}" }

    static func number(_ value: Double, decimals: Int) -> String {
        guard value.isFinite else { return "0" }  // JSON has no NaN or Infinity
        if value == value.rounded() && abs(value) < 1e15 {
            return String(format: "%.1f", value)
        }
        return String(format: "%.\(decimals)f", value)
    }

    static func escape(_ value: String) -> String {
        var out = ""
        out.reserveCapacity(value.count + 2)
        for character in value.unicodeScalars {
            switch character {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default:
                if character.value < 0x20 {
                    out += String(format: "\\u%04x", character.value)
                } else {
                    out.unicodeScalars.append(character)
                }
            }
        }
        return out
    }
}

// MARK: - Maths

enum Maths {
    /// Above this |sin(yaw)| the rotation is at gimbal lock and pitch/roll are no
    /// longer independently determined.
    static let gimbalLockThreshold = 0.99999

    /// Quaternion (x, y, z, w) to (pitch, yaw, roll) in degrees.
    ///
    /// **Gimbal lock is handled explicitly.** At yaw = ±90° only the *sum* of
    /// pitch and roll is determined, and the naive formula computes
    /// `atan2(0, 1 - 2(x² + y²))` whose second argument lands on either side of
    /// zero depending on the platform's `sin` — 0° on one machine and 180° on
    /// another for identical input. Both are valid, which is the problem.
    ///
    /// So at the pole roll is defined as 0 and the remaining rotation goes into
    /// pitch. The desktop client and the Android sender implement the same rule,
    /// which is the point: three implementations that must agree.
    static func euler(x: Double, y: Double, z: Double, w: Double)
        -> (pitch: Double, yaw: Double, roll: Double) {
        // Clamp before asin: a denormalised quaternion would otherwise yield NaN.
        let sinYaw = Swift.max(-1.0, Swift.min(1.0, 2.0 * (w * y - z * x)))
        let yaw = asin(sinYaw)

        var pitch: Double
        var roll: Double
        if abs(sinYaw) >= gimbalLockThreshold {
            pitch = 2.0 * atan2(x, w)
            // Keep pitch in (-180, 180] rather than wrapping to ±360.
            if pitch > Double.pi {
                pitch -= 2.0 * Double.pi
            } else if pitch < -Double.pi {
                pitch += 2.0 * Double.pi
            }
            roll = 0.0
        } else {
            pitch = atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
            roll = atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        }

        let toDegrees = 180.0 / Double.pi
        return (pitch * toDegrees, yaw * toDegrees, roll * toDegrees)
    }

    /// Convert metres to the u16 millimetres the depth stream carries.
    ///
    /// Zero means "no measurement", so a non-finite or non-positive reading maps
    /// to zero rather than wrapping around into a plausible-looking distance.
    static func depthMillimetres(metres: Float) -> UInt16 {
        guard metres.isFinite, metres > 0 else { return 0 }
        let millimetres = (metres * 1000.0).rounded()
        // Anything past 65.535 m is outside what the sensor can actually
        // measure, so report "no measurement" rather than a wrapped value that
        // would read as a believable short distance.
        guard millimetres >= 1, millimetres <= Float(UInt16.max) else { return 0 }
        return UInt16(millimetres)
    }
}
