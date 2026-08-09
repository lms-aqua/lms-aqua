import AVFoundation
import Network
import SwiftUI
import UIKit

@main
struct LostCamApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

// MARK: - View model

@MainActor
final class SenderModel: ObservableObject {
    @Published var isStreaming = false
    @Published var status = "Idle"
    @Published var clientCount = 0
    @Published var addresses: [String] = []
    @Published var enabledChannels: Set<Channel> = []
    @Published var capturePosition: CameraController.Position = .back
    @Published var isLocked = false
    @Published var lockStatus = "auto exposure / focus"
    @Published var depthEnabled = false
    @Published var audioEnabled = false
    @Published var token = ""
    @Published var problems: [String] = []

    let server = LostCamServer()
    private let camera = CameraController()
    private let ar = ARBridge()
    private let sensors = SensorHub()
    private let discovery = DiscoveryResponder()

    var supportsDepth: Bool { ARBridge.supportsSceneDepth }

    init() {
        camera.delegate = self
        ar.delegate = self
        sensors.delegate = self

        server.infoProvider = { [weak self] in
            self?.infoJSON() ?? Data("{}".utf8)
        }
        discovery.infoProvider = { [weak self] in
            self?.infoJSON(includePort: true) ?? Data("{}".utf8)
        }
        server.onStateChange = { [weak self] message in
            Task { @MainActor in self?.status = message }
        }
        server.onClientCountChange = { [weak self] count in
            Task { @MainActor in self?.clientCount = count }
        }
        addresses = Self.localAddresses()
    }

    // MARK: Streaming

    func start() {
        CameraController.requestCameraAccess { [weak self] granted in
            guard let self else { return }
            guard granted else {
                self.status = "Camera permission denied — enable it in Settings"
                return
            }
            self.beginStreaming()
        }
    }

    private func beginStreaming() {
        // The screen must stay awake: a locked phone stops capturing.
        UIApplication.shared.isIdleTimerDisabled = true

        server.token = token.isEmpty ? nil : token
        do {
            try server.start()
        } catch {
            status = "Could not start the server: \(error.localizedDescription)"
            return
        }
        discovery.start()

        let arMode = resolveARMode()
        if arMode == .off {
            camera.isAudioEnabled = audioEnabled
            camera.configure(position: capturePosition)
            camera.start()
        } else {
            // One AR session serves both pixels and tracking; running a separate
            // capture session alongside it would double the power draw.
            ar.wantsPlanes = enabledChannels.contains(.arPlanes)
            ar.wantsLight = enabledChannels.contains(.light)
            ar.isDepthEnabled = depthEnabled && ARBridge.supportsSceneDepth
            ar.providesVideoFrames = true
            ar.start(mode: arMode)
        }

        sensors.start(channels: enabledChannels)
        isStreaming = true
        addresses = Self.localAddresses()
        status = "Streaming on port \(server.boundPort)"
    }

    func stop() {
        UIApplication.shared.isIdleTimerDisabled = false
        camera.stop()
        ar.stop()
        sensors.stopAll()
        discovery.stop()
        server.stop()
        isStreaming = false
        status = "Idle"
    }

    /// Decide which ARKit configuration the enabled channels require.
    ///
    /// Face tracking and world tracking are mutually exclusive, so face wins
    /// when both are asked for — it is the more specific request.
    private func resolveARMode() -> ARBridge.Mode {
        if enabledChannels.contains(.arFace) && ARBridge.supportsFaceTracking {
            return .face
        }
        let needsWorld = enabledChannels.contains(.arWorld)
            || enabledChannels.contains(.arPlanes)
            || (depthEnabled && ARBridge.supportsSceneDepth)
        if needsWorld && ARBridge.supportsWorldTracking {
            return .world
        }
        // The light estimate alone is not worth an AR session's battery cost.
        return .off
    }

    func toggle(_ channel: Channel) {
        if enabledChannels.contains(channel) {
            enabledChannels.remove(channel)
        } else {
            enabledChannels.insert(channel)
        }
        if isStreaming {
            // Changing AR channels needs the session reconfigured.
            restartIfNeeded()
        }
    }

    private func restartIfNeeded() {
        stop()
        beginStreaming()
    }

    func switchCamera() {
        capturePosition = capturePosition == .back ? .front : .back
        if isStreaming && ar.mode == .off {
            camera.switchCamera()
        } else if isStreaming {
            restartIfNeeded()
        }
    }

    // MARK: Capture locks

    func toggleLock() {
        let target = !isLocked
        camera.lockCaptureSettings(target) { [weak self] message in
            Task { @MainActor in
                self?.isLocked = target
                self?.lockStatus = message
            }
        }
    }

    /// Frame the plate, tap it, and hold those settings for the whole run.
    func focusOnPlate() {
        // Centre of the frame in the normalised device space AVFoundation uses.
        camera.focusAndExpose(at: CGPoint(x: 0.5, y: 0.5), thenLock: true) {
            [weak self] message in
            Task { @MainActor in
                self?.isLocked = true
                self?.lockStatus = message
            }
        }
    }

    // MARK: Info

    /// Build the `/info` capability document (docs/PROTOCOL.md §1).
    ///
    /// Assembled with `JSONSerialization` rather than the streaming JSONWriter:
    /// this is a nested document requested once per connection, so clarity beats
    /// the per-sample efficiency the writer exists for.
    func infoJSON(includePort: Bool = false) -> Data {
        let width = camera.captureWidth > 0 ? camera.captureWidth : 1280
        let height = camera.captureHeight > 0 ? camera.captureHeight : 720

        var document: [String: Any] = [
            "product": LostCam.product,
            "protocol": LostCam.protocolVersion,
            "device": UIDevice.current.model,
            "platform": "ios",
            "os": UIDevice.current.systemVersion,
            "cameras": ["back", "front"],
            "video": ["width": width, "height": height, "fps": camera.frameRateCap],
            "audio": ["rate": 44100, "channels": 1, "format": "s16le"],
            // The capability contract: what is actually available right now,
            // not what the enum could theoretically offer.
            "channels": availableChannelNames(),
            // Recorded so a dataset can prove whether it was captured with the
            // camera locked — an unlocked run is a different kind of data.
            "capture": [
                "locks": camera.lockStateDescription(),
                "position": capturePosition == .back ? "back" : "front",
            ],
        ]

        if includePort {
            document["port"] = Int(server.boundPort)
        }

        // Depth capability, described honestly: absent hardware means absent.
        if supportsDepth {
            document["depth"] = [
                "available": depthEnabled,
                "format": "u16mm",
                "source": ARBridge.depthSourceName,
            ]
        } else {
            document["depth"] = ["available": false, "source": "none"]
        }

        // Lets a consumer map the monotonic clock onto wall time, as accurately
        // as one request's round trip allows and no more.
        document["clock"] = [
            "mono": MonotonicClock.milliseconds(),
            "unix": Int(Date().timeIntervalSince1970 * 1000),
        ]

        guard let data = try? JSONSerialization.data(withJSONObject: document,
                                                    options: [.sortedKeys]) else {
            return Data("{\"product\":\"LostCam\",\"protocol\":2}".utf8)
        }
        return data
    }

    private func availableChannelNames() -> [String] {
        var names: [String] = []
        for channel in Channel.allCases {
            guard enabledChannels.contains(channel) else { continue }
            if channel.requiresARSession {
                let supported = channel == .arFace
                    ? ARBridge.supportsFaceTracking : ARBridge.supportsWorldTracking
                if supported { names.append(channel.rawValue) }
            } else if SensorHub.isAvailable(channel) {
                names.append(channel.rawValue)
            }
        }
        return names
    }

    /// This device's LAN addresses, so the app can print the URL to type.
    static func localAddresses() -> [String] {
        var addresses: [String] = []
        var pointer: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&pointer) == 0, let first = pointer else { return [] }
        defer { freeifaddrs(pointer) }

        var current: UnsafeMutablePointer<ifaddrs>? = first
        while let entry = current {
            let interface = entry.pointee
            let family = interface.ifa_addr.pointee.sa_family
            if family == UInt8(AF_INET) {
                let name = String(cString: interface.ifa_name)
                // en0 is Wi-Fi; bridge/utun are not useful to show a user.
                if name.hasPrefix("en") {
                    var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
                    if getnameinfo(interface.ifa_addr,
                                   socklen_t(interface.ifa_addr.pointee.sa_len),
                                   &host, socklen_t(host.count), nil, 0,
                                   NI_NUMERICHOST) == 0 {
                        let address = String(cString: host)
                        if !address.isEmpty, !address.hasPrefix("127.") {
                            addresses.append(address)
                        }
                    }
                }
            }
            current = interface.ifa_next
        }
        return Array(Set(addresses)).sorted()
    }

    fileprivate func note(problem: String) {
        problems.insert(problem, at: 0)
        if problems.count > 5 { problems.removeLast() }
    }
}

// MARK: - Capture delegates

extension SenderModel: CaptureDelegate {
    nonisolated func capture(didProduceJPEG jpeg: Data, timestampMillis: Int) {
        server.broadcast(videoFrame: jpeg, timestampMillis: timestampMillis)
    }

    nonisolated func capture(didProduceAudio pcm: Data) {
        server.broadcast(audio: pcm)
    }

    nonisolated func capture(didFailWith message: String) {
        Task { @MainActor in
            self.status = message
            self.note(problem: message)
        }
    }
}

extension SenderModel: ARBridgeDelegate {
    nonisolated func arBridge(didProduce sample: SensorSample) {
        server.broadcast(sample: sample)
    }

    nonisolated func arBridge(didProduceDepth millimetres: Data, width: Int,
                              height: Int, timestampMillis: Int,
                              intrinsics: (Float, Float, Float, Float)) {
        server.broadcast(depth: millimetres, width: width, height: height,
                         timestampMillis: timestampMillis, intrinsics: intrinsics)
    }

    nonisolated func arBridge(didProduceJPEG jpeg: Data, timestampMillis: Int) {
        server.broadcast(videoFrame: jpeg, timestampMillis: timestampMillis)
    }

    nonisolated func arBridge(didChangeState message: String) {
        Task { @MainActor in self.status = message }
    }
}

extension SenderModel: SensorHubDelegate {
    nonisolated func sensorHub(didProduce sample: SensorSample) {
        server.broadcast(sample: sample)
    }

    nonisolated func sensorHub(didReportProblem message: String) {
        Task { @MainActor in self.note(problem: message) }
    }
}

// MARK: - View

struct ContentView: View {
    @StateObject private var model = SenderModel()

    var body: some View {
        NavigationStack {
            List {
                statusSection
                lockSection
                channelSection
                if model.supportsDepth { depthSection }
                securitySection
                notesSection
            }
            .navigationTitle("LostCam")
            .safeAreaInset(edge: .bottom) { startButton }
        }
    }

    private var statusSection: some View {
        Section("Status") {
            LabeledContent("State", value: model.status)
            LabeledContent("Clients", value: "\(model.clientCount)")
            if model.addresses.isEmpty {
                Text("No Wi-Fi address — connect to Wi-Fi, or use USB.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(model.addresses, id: \.self) { address in
                    VStack(alignment: .leading, spacing: 2) {
                        Text("http://\(address):\(String(model.server.boundPort))")
                            .font(.system(.footnote, design: .monospaced))
                        Text("lostcam pull \(address)")
                            .font(.system(.caption2, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                }
            }
            Button("Switch camera") { model.switchCamera() }
        }
    }

    private var lockSection: some View {
        Section {
            Button {
                model.focusOnPlate()
            } label: {
                Label("Focus on centre and lock", systemImage: "viewfinder")
            }
            Toggle("Lock exposure, white balance, focus", isOn: Binding(
                get: { model.isLocked },
                set: { _ in model.toggleLock() }))
            Text(model.lockStatus)
                .font(.caption)
                .foregroundStyle(.secondary)
        } header: {
            Text("Capture consistency")
        } footer: {
            Text("For a fixed rig watching a printer, lock these once the shot is "
                 + "framed. Auto exposure and auto focus drift over hours, and a "
                 + "model then learns the camera's reaction instead of the scene.")
        }
    }

    private var channelSection: some View {
        Section {
            ForEach(Channel.allCases) { channel in
                Toggle(isOn: Binding(
                    get: { model.enabledChannels.contains(channel) },
                    set: { _ in model.toggle(channel) })) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(channel.label)
                        Text(channel.detail)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .disabled(!isSupported(channel))
            }
        } header: {
            Text("Data channels")
        } footer: {
            Text("Every channel is off until you turn it on. Location needs its "
                 + "own permission and is never enabled implicitly.")
        }
    }

    private var depthSection: some View {
        Section {
            Toggle("Stream LiDAR depth", isOn: $model.depthEnabled)
        } header: {
            Text("Depth")
        } footer: {
            Text("u16 millimetres at the sensor's own resolution. Good for gross "
                 + "shape and failure detection; it resolves centimetres, not "
                 + "print layer heights.")
        }
    }

    private var securitySection: some View {
        Section {
            SecureField("Optional token", text: $model.token)
                .textInputAutocapitalization(.never)
                .disableAutocorrection(true)
            Toggle("Stream microphone", isOn: $model.audioEnabled)
        } header: {
            Text("Access")
        } footer: {
            Text("Anyone who can reach this phone on the network can watch the "
                 + "stream. Set a token, or use USB, on a network you do not "
                 + "control. Never port-forward this to the internet.")
        }
    }

    @ViewBuilder
    private var notesSection: some View {
        if !model.problems.isEmpty {
            Section("Recent problems") {
                ForEach(model.problems, id: \.self) { problem in
                    Text(problem).font(.caption)
                }
            }
        }
    }

    private var startButton: some View {
        Button {
            model.isStreaming ? model.stop() : model.start()
        } label: {
            Text(model.isStreaming ? "Stop streaming" : "Start streaming")
                .font(.headline)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 14)
        }
        .buttonStyle(.borderedProminent)
        .tint(model.isStreaming ? .red : .accentColor)
        .padding()
        .background(.bar)
    }

    private func isSupported(_ channel: Channel) -> Bool {
        if channel.requiresFaceTracking { return ARBridge.supportsFaceTracking }
        if channel.requiresARSession { return ARBridge.supportsWorldTracking }
        return SensorHub.isAvailable(channel)
    }
}
