import Foundation

/// The sensor and AR data channel (docs/PROTOCOL.md §6, docs/SENSORS.md).
///
/// Channels are opt-in, and sample encoding is centralised here so the field
/// names match the spec in exactly one place.
///
/// Note the shape this is built around: sources emit *typed* samples, and each
/// subscriber encodes them itself. That is not indirection for its own sake —
/// `seq` must be contiguous per subscriber, and two subscribers asking for
/// different channel subsets cannot share one counter without one of them
/// seeing gaps, which the spec defines as dropped samples.

// MARK: - Channels

enum Channel: String, CaseIterable, Identifiable {
    case attitude
    case motion
    case arWorld = "ar.world"
    case arFace = "ar.face"
    case arPlanes = "ar.planes"
    case light
    case barometer
    case battery
    case location

    var id: String { rawValue }

    /// What the user sees in the app's channel list.
    var label: String {
        switch self {
        case .attitude: return "Orientation"
        case .motion: return "Motion & IMU"
        case .arWorld: return "AR world pose (6DoF)"
        case .arFace: return "AR face & blendshapes"
        case .arPlanes: return "AR planes"
        case .light: return "Light estimate"
        case .barometer: return "Barometer"
        case .battery: return "Battery & thermal"
        case .location: return "Location"
        }
    }

    var detail: String {
        switch self {
        case .attitude: return "Quaternion and euler angles. Cheap."
        case .motion: return "Acceleration, rotation rate, gravity, magnetometer."
        case .arWorld: return "Camera pose from ARKit. Costs battery and heat."
        case .arFace: return "52 ARKit blendshapes. Front camera only."
        case .arPlanes: return "Detected planes — a partial map of the room."
        case .light: return "Ambient intensity and colour temperature."
        case .barometer: return "Pressure and relative altitude."
        case .battery: return "Explains why the frame rate dropped."
        case .location: return "Off by default. Requires separate permission."
        }
    }

    /// Channels needing an ARKit session, which is why they are not free.
    var requiresARSession: Bool {
        self == .arWorld || self == .arFace || self == .arPlanes || self == .light
    }

    /// Face tracking uses a different ARKit configuration to world tracking.
    var requiresFaceTracking: Bool { self == .arFace }

    /// Treated separately everywhere: it never turns on implicitly.
    var isSensitive: Bool { self == .location }
}

// MARK: - Typed samples

struct AttitudeSample {
    var x: Double
    var y: Double
    var z: Double
    var w: Double
    var reference: String
    var accuracy: String?
}

struct MotionSample {
    var userAcceleration: [Float]
    var gravity: [Float]
    var rotationRate: [Float]
    var magneticField: [Float]?
    var magneticAccuracy: String?
}

struct WorldSample {
    var pose: [Float]
    var state: String
    var reason: String?
    var features: Int?
    var intrinsics: [Float]?
    var resolution: [Int]?
}

struct FaceSample {
    var tracked: Bool
    /// Only non-zero coefficients, per the spec: an absent key means zero, so
    /// sending zeros would be pure waste at 60 Hz.
    var blendShapes: [String: Float]
    var transform: [Float]?
    var leftEye: [Float]?
    var rightEye: [Float]?
    var look: [Float]?
}

struct PlaneSample {
    var id: String
    var event: String
    var center: [Float]?
    var extent: [Float]?
    var alignment: String?
    var classification: String?
}

struct LightSample {
    var lumens: Double
    var kelvin: Double?
}

struct BarometerSample {
    var kilopascals: Double
    var relativeAltitude: Double?
}

struct BatterySample {
    var level: Double
    var charging: Bool
    var thermal: String
}

struct LocationSample {
    var latitude: Double
    var longitude: Double
    var accuracy: Double
    var altitude: Double?
    var speed: Double?
    var heading: Double?
}

enum SensorSample {
    case attitude(AttitudeSample)
    case motion(MotionSample)
    case world(WorldSample)
    case face(FaceSample)
    case plane(PlaneSample)
    case light(LightSample)
    case barometer(BarometerSample)
    case battery(BatterySample)
    case location(LocationSample)

    var channel: Channel {
        switch self {
        case .attitude: return .attitude
        case .motion: return .motion
        case .world: return .arWorld
        case .face: return .arFace
        case .plane: return .arPlanes
        case .light: return .light
        case .barometer: return .barometer
        case .battery: return .battery
        case .location: return .location
        }
    }
}

// MARK: - Sample encoding

/// Encodes typed samples as NDJSON lines, numbering its own stream.
///
/// One instance per subscriber, which is what keeps `seq` contiguous.
final class SampleEncoder {
    private var sequence = 0
    private let clock: () -> Int

    /// - Parameter clock: monotonic milliseconds. Injected so tests are stable.
    init(clock: @escaping () -> Int = { MonotonicClock.milliseconds() }) {
        self.clock = clock
    }

    func reset() { sequence = 0 }

    var emitted: Int { sequence }

    func encode(_ sample: SensorSample) -> Data {
        var writer = begin(sample.channel)
        switch sample {
        case .attitude(let value):
            writer.add("q", floats: [Float(value.x), Float(value.y),
                                     Float(value.z), Float(value.w)], decimals: 6)
            let angles = Maths.euler(x: value.x, y: value.y, z: value.z, w: value.w)
            writer.add("euler", floats: [Float(angles.pitch), Float(angles.yaw),
                                         Float(angles.roll)], decimals: 3)
            writer.add("ref", value.reference)
            if let accuracy = value.accuracy { writer.add("accuracy", accuracy) }

        case .motion(let value):
            writer.add("accel", floats: value.userAcceleration)
            writer.add("gravity", floats: value.gravity)
            writer.add("rot", floats: value.rotationRate)
            if let field = value.magneticField {
                writer.add("mag", floats: field, decimals: 3)
            }
            if let accuracy = value.magneticAccuracy {
                writer.add("magAccuracy", accuracy)
            }

        case .world(let value):
            writer.add("pose", floats: value.pose)
            writer.add("state", value.state)
            if let reason = value.reason { writer.add("reason", reason) }
            if let features = value.features { writer.add("features", features) }
            if let intrinsics = value.intrinsics {
                writer.add("intrinsics", floats: intrinsics, decimals: 2)
            }
            if let resolution = value.resolution, resolution.count == 2 {
                writer.add("resolution",
                           floats: [Float(resolution[0]), Float(resolution[1])],
                           decimals: 0)
            }

        case .face(let value):
            writer.add("tracked", value.tracked)
            writer.add("blend", dictionary: value.blendShapes)
            if let transform = value.transform {
                writer.add("transform", floats: transform)
            }
            if let leftEye = value.leftEye { writer.add("leftEye", floats: leftEye) }
            if let rightEye = value.rightEye { writer.add("rightEye", floats: rightEye) }
            if let look = value.look { writer.add("look", floats: look) }

        case .plane(let value):
            writer.add("id", value.id)
            writer.add("event", value.event)
            // A removal carries only id and event; there is no geometry left.
            if value.event != "removed" {
                if let center = value.center { writer.add("center", floats: center) }
                if let extent = value.extent { writer.add("extent", floats: extent) }
                if let alignment = value.alignment { writer.add("align", alignment) }
                if let classification = value.classification {
                    writer.add("classification", classification)
                }
            }

        case .light(let value):
            writer.add("lumens", value.lumens, decimals: 2)
            if let kelvin = value.kelvin { writer.add("kelvin", kelvin, decimals: 1) }

        case .barometer(let value):
            writer.add("kpa", value.kilopascals, decimals: 4)
            if let altitude = value.relativeAltitude {
                writer.add("relAltitude", altitude, decimals: 3)
            }

        case .battery(let value):
            writer.add("level", value.level, decimals: 3)
            writer.add("charging", value.charging)
            writer.add("thermal", value.thermal)

        case .location(let value):
            writer.add("lat", value.latitude, decimals: 7)
            writer.add("lon", value.longitude, decimals: 7)
            writer.add("accuracy", value.accuracy, decimals: 2)
            if let altitude = value.altitude {
                writer.add("altitude", altitude, decimals: 2)
            }
            if let speed = value.speed { writer.add("speed", speed, decimals: 3) }
            if let heading = value.heading { writer.add("heading", heading, decimals: 2) }
        }
        return writer.finish()
    }

    /// Start a record, stamping the required t/seq/ch fields.
    private func begin(_ channel: Channel) -> JSONWriter {
        sequence += 1
        var writer = JSONWriter()
        writer.add("t", clock())
        writer.add("seq", sequence)
        writer.add("ch", channel.rawValue)
        return writer
    }
}

// MARK: - Clock

enum MonotonicClock {
    /// Monotonic milliseconds, matching the definition in PROTOCOL.md §6.3.
    ///
    /// Deliberately not wall time: the useful questions are "how far apart" and
    /// "which video frame", and both survive a clock that never jumps.
    static func milliseconds() -> Int {
        Int(ProcessInfo.processInfo.systemUptime * 1000.0)
    }
}

// MARK: - Rate limiting

/// Per-channel decimation for the `hz` query parameter.
///
/// Decimates rather than buffers: a late sample is worthless, so dropping it is
/// the correct response to a rate cap.
struct RateLimiter {
    private var lastEmitted: [String: Int] = [:]
    let intervalMillis: Int

    init(hz: Int) {
        intervalMillis = hz > 0 ? max(1, 1000 / hz) : 0
    }

    mutating func shouldEmit(channel: String, nowMillis: Int) -> Bool {
        guard intervalMillis > 0 else { return true }
        if let last = lastEmitted[channel], nowMillis - last < intervalMillis {
            return false
        }
        lastEmitted[channel] = nowMillis
        return true
    }
}

// MARK: - Subscribers

/// One `/data` subscriber: its channel filter, its rate cap, its own numbering.
final class DataSubscriber {
    let id: UUID
    private let wanted: Set<String>
    private var limiter: RateLimiter
    private let encoder: SampleEncoder
    private let send: (Data) -> Void

    init(id: UUID = UUID(), channels: Set<String>, hz: Int,
         encoder: SampleEncoder = SampleEncoder(),
         send: @escaping (Data) -> Void) {
        self.id = id
        self.wanted = channels
        self.limiter = RateLimiter(hz: hz)
        self.encoder = encoder
        self.send = send
    }

    /// An empty channel set means "everything available".
    func accepts(channel: String) -> Bool {
        wanted.isEmpty || wanted.contains(channel)
    }

    var emitted: Int { encoder.emitted }

    /// Encode and deliver, if this subscriber wants it and is not saturated.
    ///
    /// Encoding happens after the filter, so a subscriber asking for one channel
    /// costs nothing for the others.
    func offer(_ sample: SensorSample, nowMillis: Int) {
        let channel = sample.channel.rawValue
        guard accepts(channel: channel) else { return }
        guard limiter.shouldEmit(channel: channel, nowMillis: nowMillis) else { return }
        send(encoder.encode(sample))
    }
}

/// Parse the `ch` query parameter into a channel set.
func parseChannelFilter(_ raw: String?) -> Set<String> {
    guard let raw, !raw.isEmpty else { return [] }
    let names = raw.split(separator: ",").map {
        $0.trimmingCharacters(in: .whitespaces)
    }
    return Set(names.filter { !$0.isEmpty })
}
