import ARKit
import CoreImage
import CoreVideo
import Foundation
import simd

/// ARKit bridge: world pose, face blendshapes, planes, light estimate and LiDAR
/// depth, converted to the wire schema in docs/SENSORS.md.
///
/// Two configurations, never both: `ARWorldTrackingConfiguration` for the rear
/// camera (pose, planes, LiDAR depth) and `ARFaceTrackingConfiguration` for the
/// front camera (blendshapes). ARKit cannot serve both at once on most devices,
/// so the app picks one from the enabled channels.

protocol ARBridgeDelegate: AnyObject {
    func arBridge(didProduce sample: SensorSample)
    func arBridge(didProduceDepth millimetres: Data, width: Int, height: Int,
                  timestampMillis: Int, intrinsics: (Float, Float, Float, Float))
    func arBridge(didProduceJPEG jpeg: Data, timestampMillis: Int)
    func arBridge(didChangeState message: String)
}

final class ARBridge: NSObject {
    enum Mode {
        case off
        case world
        case face
    }

    weak var delegate: ARBridgeDelegate?

    let session = ARSession()
    private(set) var mode: Mode = .off
    private var knownPlanes: Set<UUID> = []

    /// Depth is heavy — a 256x192 float32 buffer per frame — so it is only
    /// converted when a client is actually reading /depth.
    var isDepthEnabled = false
    var wantsPlanes = false
    var wantsLight = false
    /// When true, ARKit's frames also feed the video stream, so a single AR
    /// session serves both instead of running a second capture session.
    var providesVideoFrames = false
    var jpegQuality: CGFloat = 0.8
    var frameRateCap = 30

    private var lastFrameMillis = 0
    private let context = CIContext(options: [.useSoftwareRenderer: false])

    // MARK: Capability probes

    static var supportsWorldTracking: Bool {
        ARWorldTrackingConfiguration.isSupported
    }

    static var supportsFaceTracking: Bool {
        ARFaceTrackingConfiguration.isSupported
    }

    /// LiDAR. Present on iPhone Pro and iPad Pro models, absent elsewhere.
    static var supportsSceneDepth: Bool {
        ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
    }

    static var depthSourceName: String {
        supportsSceneDepth ? "lidar" : "none"
    }

    override init() {
        super.init()
        session.delegate = self
    }

    // MARK: Lifecycle

    func start(mode: Mode) {
        guard mode != .off else {
            stop()
            return
        }
        self.mode = mode
        knownPlanes.removeAll()

        switch mode {
        case .face:
            guard Self.supportsFaceTracking else {
                delegate?.arBridge(didChangeState:
                    "face tracking is not supported on this device")
                self.mode = .off
                return
            }
            let configuration = ARFaceTrackingConfiguration()
            configuration.isLightEstimationEnabled = wantsLight
            session.run(configuration, options: [.resetTracking, .removeExistingAnchors])

        case .world:
            guard Self.supportsWorldTracking else {
                delegate?.arBridge(didChangeState:
                    "world tracking is not supported on this device")
                self.mode = .off
                return
            }
            let configuration = ARWorldTrackingConfiguration()
            configuration.isLightEstimationEnabled = wantsLight
            configuration.planeDetection = wantsPlanes ? [.horizontal, .vertical] : []
            if isDepthEnabled && Self.supportsSceneDepth {
                configuration.frameSemantics.insert(.sceneDepth)
            }
            session.run(configuration, options: [.resetTracking, .removeExistingAnchors])

        case .off:
            break
        }
        delegate?.arBridge(didChangeState: "AR session running (\(mode))")
    }

    func stop() {
        session.pause()
        mode = .off
        knownPlanes.removeAll()
    }

    // MARK: Conversion helpers

    /// Flatten a 4x4 to the 16-element column-major array the spec requires.
    ///
    /// simd stores columns already, so this is a straight read — but the order
    /// is the thing consumers get wrong, so it lives in one tested function.
    static func flatten(_ matrix: simd_float4x4) -> [Float] {
        let columns = [matrix.columns.0, matrix.columns.1,
                       matrix.columns.2, matrix.columns.3]
        var out: [Float] = []
        out.reserveCapacity(16)
        for column in columns {
            out.append(column.x)
            out.append(column.y)
            out.append(column.z)
            out.append(column.w)
        }
        return out
    }

    static func trackingStateName(_ state: ARCamera.TrackingState) -> String {
        switch state {
        case .normal: return "normal"
        case .notAvailable: return "unavailable"
        case .limited: return "limited"
        }
    }

    static func trackingReasonName(_ state: ARCamera.TrackingState) -> String? {
        guard case .limited(let reason) = state else { return nil }
        switch reason {
        case .initializing: return "initializing"
        case .excessiveMotion: return "excessiveMotion"
        case .insufficientFeatures: return "insufficientFeatures"
        case .relocalizing: return "relocalizing"
        @unknown default: return "unknown"
        }
    }

    static func alignmentName(_ alignment: ARPlaneAnchor.Alignment) -> String {
        switch alignment {
        case .horizontal: return "horizontal"
        case .vertical: return "vertical"
        @unknown default: return "any"
        }
    }

    static func classificationName(
        _ classification: ARPlaneAnchor.Classification) -> String {
        switch classification {
        case .wall: return "wall"
        case .floor: return "floor"
        case .ceiling: return "ceiling"
        case .table: return "table"
        case .seat: return "seat"
        case .door: return "door"
        case .window: return "window"
        case .none: return "none"
        @unknown default: return "none"
        }
    }

    /// Convert an ARKit depth map to the u16-millimetre raster the spec defines.
    ///
    /// ARKit gives float32 metres; the wire format is u16 millimetres, which
    /// halves the bandwidth at a precision the sensor does not exceed anyway.
    /// Confidence, when present, is used to zero out low-confidence pixels —
    /// zero means "no measurement", which is far more useful to a consumer than
    /// a confidently wrong distance.
    static func depthMillimetres(from depthMap: CVPixelBuffer,
                                 confidence: CVPixelBuffer?) -> Data? {
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }

        let width = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)
        guard width > 0, height > 0,
              let base = CVPixelBufferGetBaseAddress(depthMap) else { return nil }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(depthMap)

        var confidenceBase: UnsafeMutableRawPointer?
        var confidenceRow = 0
        if let confidence {
            CVPixelBufferLockBaseAddress(confidence, .readOnly)
            confidenceBase = CVPixelBufferGetBaseAddress(confidence)
            confidenceRow = CVPixelBufferGetBytesPerRow(confidence)
        }
        defer {
            if let confidence { CVPixelBufferUnlockBaseAddress(confidence, .readOnly) }
        }

        var out = Data(count: width * height * 2)
        out.withUnsafeMutableBytes { rawOut in
            guard let destination = rawOut.bindMemory(to: UInt16.self).baseAddress else {
                return
            }
            for row in 0..<height {
                let sourceRow = base.advanced(by: row * bytesPerRow)
                    .assumingMemoryBound(to: Float32.self)
                let confidenceRowPointer = confidenceBase?
                    .advanced(by: row * confidenceRow)
                    .assumingMemoryBound(to: UInt8.self)
                for column in 0..<width {
                    var millimetres = Maths.depthMillimetres(metres: sourceRow[column])
                    if let confidenceRowPointer,
                       Int(confidenceRowPointer[column])
                        < ARConfidenceLevel.medium.rawValue {
                        millimetres = 0
                    }
                    destination[row * width + column] = millimetres.littleEndian
                }
            }
        }
        return out
    }
}

// MARK: - ARSessionDelegate

extension ARBridge: ARSessionDelegate {
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let now = MonotonicClock.milliseconds()

        emitWorld(frame: frame, now: now)
        emitLight(frame: frame)
        emitFace(frame: frame)
        emitDepth(frame: frame, now: now)
        emitVideo(frame: frame, now: now)
    }

    private func emitWorld(frame: ARFrame, now: Int) {
        guard mode == .world else { return }
        let camera = frame.camera
        let resolution = camera.imageResolution
        let intrinsics = camera.intrinsics
        let sample = WorldSample(
            pose: Self.flatten(camera.transform),
            state: Self.trackingStateName(camera.trackingState),
            reason: Self.trackingReasonName(camera.trackingState),
            features: frame.rawFeaturePoints?.points.count,
            // [fx, fy, cx, cy] — the layout consumers expect.
            intrinsics: [intrinsics.columns.0.x, intrinsics.columns.1.y,
                         intrinsics.columns.2.x, intrinsics.columns.2.y],
            resolution: [Int(resolution.width), Int(resolution.height)])
        delegate?.arBridge(didProduce: .world(sample))
    }

    private func emitLight(frame: ARFrame) {
        guard wantsLight, let estimate = frame.lightEstimate else { return }
        delegate?.arBridge(didProduce: .light(LightSample(
            lumens: estimate.ambientIntensity,
            kelvin: estimate.ambientColorTemperature)))
    }

    private func emitFace(frame: ARFrame) {
        guard mode == .face else { return }
        for anchor in frame.anchors {
            guard let face = anchor as? ARFaceAnchor else { continue }

            // Only non-zero coefficients: an absent key means zero, and at 60 Hz
            // sending 52 mostly-zero floats is pure waste.
            var blendShapes: [String: Float] = [:]
            for (location, value) in face.blendShapes {
                let coefficient = value.floatValue
                if coefficient > 0.001 {
                    blendShapes[location.rawValue] = coefficient
                }
            }

            let look = [face.lookAtPoint.x, face.lookAtPoint.y, face.lookAtPoint.z]
            let sample = FaceSample(
                tracked: face.isTracked,
                blendShapes: blendShapes,
                transform: Self.flatten(face.transform),
                leftEye: Self.flatten(face.leftEyeTransform),
                rightEye: Self.flatten(face.rightEyeTransform),
                look: look)
            delegate?.arBridge(didProduce: .face(sample))
        }
    }

    private func emitDepth(frame: ARFrame, now: Int) {
        guard isDepthEnabled, let depth = frame.sceneDepth else { return }
        let map = depth.depthMap
        let width = CVPixelBufferGetWidth(map)
        let height = CVPixelBufferGetHeight(map)
        guard let payload = Self.depthMillimetres(
            from: map, confidence: depth.confidenceMap) else { return }

        // Depth has its own intrinsics: the raster is much smaller than the
        // colour frame, so the colour camera's numbers would be wrong here.
        let camera = frame.camera
        let colourResolution = camera.imageResolution
        let scaleX = Float(width) / Float(colourResolution.width)
        let scaleY = Float(height) / Float(colourResolution.height)
        let intrinsics = camera.intrinsics
        let scaled = (intrinsics.columns.0.x * scaleX,
                      intrinsics.columns.1.y * scaleY,
                      intrinsics.columns.2.x * scaleX,
                      intrinsics.columns.2.y * scaleY)

        delegate?.arBridge(didProduceDepth: payload, width: width, height: height,
                           timestampMillis: now, intrinsics: scaled)
    }

    private func emitVideo(frame: ARFrame, now: Int) {
        guard providesVideoFrames else { return }
        let minimumInterval = frameRateCap > 0 ? 1000 / frameRateCap : 0
        guard now - lastFrameMillis >= minimumInterval else { return }
        lastFrameMillis = now

        guard let jpeg = CameraController.encodeJPEG(
            pixelBuffer: frame.capturedImage, context: context,
            quality: jpegQuality) else { return }
        delegate?.arBridge(didProduceJPEG: jpeg, timestampMillis: now)
    }

    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        emitPlanes(anchors, event: "added")
    }

    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        emitPlanes(anchors, event: "updated")
    }

    func session(_ session: ARSession, didRemove anchors: [ARAnchor]) {
        for anchor in anchors {
            guard anchor is ARPlaneAnchor else { continue }
            knownPlanes.remove(anchor.identifier)
            delegate?.arBridge(didProduce: .plane(PlaneSample(
                id: anchor.identifier.uuidString, event: "removed",
                center: nil, extent: nil, alignment: nil, classification: nil)))
        }
    }

    private func emitPlanes(_ anchors: [ARAnchor], event: String) {
        guard wantsPlanes else { return }
        for anchor in anchors {
            guard let plane = anchor as? ARPlaneAnchor else { continue }
            // An anchor can be reported as updated before we ever saw it added.
            let resolvedEvent = knownPlanes.contains(plane.identifier)
                ? event : "added"
            knownPlanes.insert(plane.identifier)

            let sample = PlaneSample(
                id: plane.identifier.uuidString,
                event: resolvedEvent,
                center: [plane.center.x, plane.center.y, plane.center.z],
                extent: [plane.planeExtent.width, plane.planeExtent.height],
                alignment: Self.alignmentName(plane.alignment),
                classification: Self.classificationName(plane.classification))
            delegate?.arBridge(didProduce: .plane(sample))
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        delegate?.arBridge(didChangeState: "AR failed: \(error.localizedDescription)")
    }

    func sessionWasInterrupted(_ session: ARSession) {
        // iOS interrupts AR and camera capture when the app is backgrounded.
        // Reporting it is the honest thing; claiming to survive it is not.
        delegate?.arBridge(didChangeState:
            "AR interrupted — iOS suspends capture when the app is not in front")
    }

    func sessionInterruptionEnded(_ session: ARSession) {
        delegate?.arBridge(didChangeState: "AR resumed")
    }
}
