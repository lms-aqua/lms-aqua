import Foundation
import Network

/// The phone-side HTTP server (docs/PROTOCOL.md §1, §6, §7).
///
/// The phone is the server and the desktop connects out, matching DroidCam, so
/// `/video` is reachable from a browser, VLC or ffmpeg as well as from the
/// LostCam client.
///
/// Built on Network.framework rather than a third-party HTTP library: the whole
/// surface is four streaming routes and three small ones, and NWListener handles
/// the Wi-Fi/USB interface details for free.

final class LostCamServer {
    // MARK: Types

    /// A connected streaming client.
    private final class Client {
        let id = UUID()
        let connection: NWConnection
        /// Backpressure guard. If a send has not completed, the next frame is
        /// dropped rather than queued — a webcam a second behind is worse than
        /// one that skipped a frame.
        var sendInFlight = false
        var subscriber: DataSubscriber?

        init(connection: NWConnection) {
            self.connection = connection
        }
    }

    enum Route {
        case status
        case info
        case video
        case audio
        case data
        case depth
        case notFound
    }

    // MARK: State

    private let port: NWEndpoint.Port
    private var listener: NWListener?
    private let queue = DispatchQueue(label: "com.lostmediastudios.lostcam.server")
    /// Serialises client bookkeeping. Frames arrive on capture queues, so this
    /// has to be safe from several threads.
    private let clientsLock = NSLock()
    private var videoClients: [UUID: Client] = [:]
    private var audioClients: [UUID: Client] = [:]
    private var dataClients: [UUID: Client] = [:]
    private var depthClients: [UUID: Client] = [:]

    /// Supplied by the app so the server never reaches into capture state.
    var infoProvider: () -> Data = { Data("{}".utf8) }
    var token: String?
    var onStateChange: ((String) -> Void)?
    var onClientCountChange: ((Int) -> Void)?

    private(set) var isRunning = false
    private(set) var framesSent = 0
    private(set) var samplesSent = 0

    init(port: UInt16 = LostCam.defaultPort) {
        self.port = NWEndpoint.Port(rawValue: port) ?? .init(integerLiteral: 4747)
    }

    var boundPort: UInt16 {
        listener?.port?.rawValue ?? port.rawValue
    }

    // MARK: Lifecycle

    func start() throws {
        guard listener == nil else { return }
        let parameters = NWParameters.tcp
        parameters.allowLocalEndpointReuse = true
        // Required for the adb/USB path, where the connection arrives on
        // loopback rather than the Wi-Fi interface.
        parameters.includePeerToPeer = true

        let listener = try NWListener(using: parameters, on: port)
        listener.newConnectionHandler = { [weak self] connection in
            self?.accept(connection)
        }
        listener.stateUpdateHandler = { [weak self] state in
            guard let self else { return }
            switch state {
            case .ready:
                self.isRunning = true
                self.onStateChange?("listening on port \(self.boundPort)")
            case .failed(let error):
                self.isRunning = false
                self.onStateChange?("failed: \(error.localizedDescription)")
            case .cancelled:
                self.isRunning = false
                self.onStateChange?("stopped")
            default:
                break
            }
        }
        self.listener = listener
        listener.start(queue: queue)
    }

    func stop() {
        listener?.cancel()
        listener = nil
        isRunning = false

        clientsLock.lock()
        let all = Array(videoClients.values) + Array(audioClients.values)
            + Array(dataClients.values) + Array(depthClients.values)
        videoClients.removeAll()
        audioClients.removeAll()
        dataClients.removeAll()
        depthClients.removeAll()
        clientsLock.unlock()

        for client in all { client.connection.cancel() }
        notifyClientCount()
    }

    var clientCount: Int {
        clientsLock.lock()
        defer { clientsLock.unlock() }
        return videoClients.count + audioClients.count + dataClients.count
            + depthClients.count
    }

    private func notifyClientCount() {
        let count = clientCount
        DispatchQueue.main.async { [weak self] in
            self?.onClientCountChange?(count)
        }
    }

    // MARK: Accepting

    private func accept(_ connection: NWConnection) {
        connection.stateUpdateHandler = { [weak self] state in
            switch state {
            case .failed, .cancelled:
                self?.forget(connection)
            default:
                break
            }
        }
        connection.start(queue: queue)
        receiveRequest(on: connection, accumulated: Data())
    }

    /// Read until the end of the request head, tolerating a split across reads.
    private func receiveRequest(on connection: NWConnection, accumulated: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 8192) {
            [weak self] chunk, _, isComplete, error in
            guard let self else { return }
            if error != nil {
                connection.cancel()
                return
            }
            var buffer = accumulated
            if let chunk { buffer.append(chunk) }

            // Cap the head: a client that never sends a blank line is not one.
            guard buffer.count <= 64 * 1024 else {
                self.send(HTTPResponse.text(status: 431,
                                            reason: "Request Header Fields Too Large",
                                            body: "request head too large"),
                          on: connection, thenClose: true)
                return
            }

            if let range = buffer.range(of: Data("\r\n\r\n".utf8))
                ?? buffer.range(of: Data("\n\n".utf8)) {
                let head = String(decoding: buffer[buffer.startIndex..<range.lowerBound],
                                  as: UTF8.self)
                self.handle(head: head, on: connection)
                return
            }
            if isComplete {
                connection.cancel()
                return
            }
            self.receiveRequest(on: connection, accumulated: buffer)
        }
    }

    static func route(for path: String) -> Route {
        switch path {
        case "", "/": return .status
        case "/info": return .info
        case "/video": return .video
        case "/audio": return .audio
        case "/data": return .data
        case "/depth": return .depth
        default: return .notFound
        }
    }

    private func handle(head: String, on connection: NWConnection) {
        guard let request = HTTPRequest.parse(head) else {
            send(HTTPResponse.text(status: 400, reason: "Bad Request",
                                   body: "malformed request"),
                 on: connection, thenClose: true)
            return
        }

        guard request.method == "GET" || request.method == "HEAD" else {
            send(HTTPResponse.text(status: 405, reason: "Method Not Allowed",
                                   body: "only GET is supported"),
                 on: connection, thenClose: true)
            return
        }

        guard authorized(request) else {
            send(HTTPResponse.unauthorized, on: connection, thenClose: true)
            return
        }

        switch Self.route(for: request.path) {
        case .status:
            send(HTTPResponse.text(status: 200, reason: "OK", body: statusPage(),
                                   contentType: "text/html; charset=utf-8"),
                 on: connection, thenClose: true)
        case .info:
            send(HTTPResponse.json(body: infoProvider()), on: connection,
                 thenClose: true)
        case .video:
            startVideoStream(request, on: connection)
        case .audio:
            startAudioStream(on: connection)
        case .data:
            startDataStream(request, on: connection)
        case .depth:
            startDepthStream(on: connection)
        case .notFound:
            send(HTTPResponse.notFound, on: connection, thenClose: true)
        }
    }

    /// Constant-time token comparison, so a wrong token leaks no timing signal.
    private func authorized(_ request: HTTPRequest) -> Bool {
        guard let token, !token.isEmpty else { return true }
        guard let supplied = request.token else { return false }
        let expected = Array(token.utf8)
        let actual = Array(supplied.utf8)
        guard expected.count == actual.count else { return false }
        var difference: UInt8 = 0
        for index in 0..<expected.count {
            difference |= expected[index] ^ actual[index]
        }
        return difference == 0
    }

    private func statusPage() -> String {
        """
        <!doctype html><meta charset=utf-8>
        <meta name=viewport content="width=device-width,initial-scale=1">
        <title>LostCam</title>
        <style>body{font:16px system-ui;margin:2rem;line-height:1.5}
        code{background:#eee;padding:.1em .3em;border-radius:3px}</style>
        <h1>LostCam is streaming</h1>
        <ul>
          <li><a href="/video">/video</a> — MJPEG video</li>
          <li><a href="/info">/info</a> — capabilities</li>
          <li><code>/data</code> — sensor &amp; AR channel (NDJSON)</li>
          <li><code>/audio</code> — PCM microphone</li>
        </ul>
        <p>On your computer: <code>lostcam pull &lt;this-phone-ip&gt;</code></p>
        """
    }

    // MARK: Streams

    private func startVideoStream(_ request: HTTPRequest, on connection: NWConnection) {
        let client = Client(connection: connection)
        clientsLock.lock()
        videoClients[client.id] = client
        clientsLock.unlock()
        notifyClientCount()

        let head = HTTPResponse.streamHead(
            contentType: MJPEGFraming.streamContentType())
        connection.send(content: head, completion: .contentProcessed { _ in })
    }

    private func startAudioStream(on connection: NWConnection) {
        let client = Client(connection: connection)
        clientsLock.lock()
        audioClients[client.id] = client
        clientsLock.unlock()
        notifyClientCount()

        let head = HTTPResponse.streamHead(
            contentType: "audio/L16; rate=44100; channels=1")
        connection.send(content: head, completion: .contentProcessed { _ in })
    }

    private func startDepthStream(on connection: NWConnection) {
        let client = Client(connection: connection)
        clientsLock.lock()
        depthClients[client.id] = client
        clientsLock.unlock()
        notifyClientCount()

        let head = HTTPResponse.streamHead(
            contentType: "multipart/x-mixed-replace; boundary=\(LostCam.depthBoundary)")
        connection.send(content: head, completion: .contentProcessed { _ in })
    }

    private func startDataStream(_ request: HTTPRequest, on connection: NWConnection) {
        let client = Client(connection: connection)
        let channels = parseChannelFilter(request.query["ch"])
        let hz = request.int("hz", default: 60, min: 1, max: 240)
        client.subscriber = DataSubscriber(channels: channels, hz: hz) {
            [weak self, weak client] line in
            guard let self, let client else { return }
            self.deliver(line, to: client, countingAs: .data)
        }

        clientsLock.lock()
        dataClients[client.id] = client
        clientsLock.unlock()
        notifyClientCount()

        let head = HTTPResponse.streamHead(contentType: "application/x-ndjson")
        connection.send(content: head, completion: .contentProcessed { _ in })
    }

    // MARK: Broadcasting

    private enum Kind { case video, audio, data, depth }

    /// Send one JPEG frame to every video client.
    func broadcast(videoFrame jpeg: Data, timestampMillis: Int) {
        clientsLock.lock()
        let clients = Array(videoClients.values)
        clientsLock.unlock()
        guard !clients.isEmpty else { return }

        let payload = MJPEGFraming.part(jpeg, timestampMillis: timestampMillis)
        for client in clients {
            deliver(payload, to: client, countingAs: .video)
        }
    }

    func broadcast(audio pcm: Data) {
        clientsLock.lock()
        let clients = Array(audioClients.values)
        clientsLock.unlock()
        for client in clients {
            deliver(pcm, to: client, countingAs: .audio)
        }
    }

    /// Offer a sample to every data client; each encodes and numbers its own.
    func broadcast(sample: SensorSample) {
        clientsLock.lock()
        let clients = Array(dataClients.values)
        clientsLock.unlock()
        guard !clients.isEmpty else { return }

        let now = MonotonicClock.milliseconds()
        for client in clients {
            client.subscriber?.offer(sample, nowMillis: now)
        }
    }

    func broadcast(depth millimetres: Data, width: Int, height: Int,
                   timestampMillis: Int,
                   intrinsics: (Float, Float, Float, Float)) {
        clientsLock.lock()
        let clients = Array(depthClients.values)
        clientsLock.unlock()
        guard !clients.isEmpty else { return }

        var payload = MJPEGFraming.depthPartHeader(
            byteCount: millimetres.count, width: width, height: height,
            timestampMillis: timestampMillis, intrinsics: intrinsics)
        payload.append(millimetres)
        payload.append(Data("\r\n".utf8))
        for client in clients {
            deliver(payload, to: client, countingAs: .depth)
        }
    }

    /// Whether anything is actually listening, so capture can idle when not.
    var hasVideoClients: Bool {
        clientsLock.lock()
        defer { clientsLock.unlock() }
        return !videoClients.isEmpty
    }

    var hasDataClients: Bool {
        clientsLock.lock()
        defer { clientsLock.unlock() }
        return !dataClients.isEmpty
    }

    var hasDepthClients: Bool {
        clientsLock.lock()
        defer { clientsLock.unlock() }
        return !depthClients.isEmpty
    }

    private func deliver(_ payload: Data, to client: Client, countingAs kind: Kind) {
        // Drop rather than queue when the previous send is still outstanding.
        clientsLock.lock()
        if client.sendInFlight {
            clientsLock.unlock()
            return
        }
        client.sendInFlight = true
        clientsLock.unlock()

        client.connection.send(content: payload, completion: .contentProcessed {
            [weak self, weak client] error in
            guard let self, let client else { return }
            self.clientsLock.lock()
            client.sendInFlight = false
            self.clientsLock.unlock()

            if error != nil {
                client.connection.cancel()
                self.forget(client.connection)
                return
            }
            switch kind {
            case .video: self.framesSent += 1
            case .data: self.samplesSent += 1
            case .audio, .depth: break
            }
        })
    }

    private func send(_ payload: Data, on connection: NWConnection, thenClose: Bool) {
        connection.send(content: payload, completion: .contentProcessed { _ in
            if thenClose { connection.cancel() }
        })
    }

    private func forget(_ connection: NWConnection) {
        clientsLock.lock()
        for (id, client) in videoClients where client.connection === connection {
            videoClients.removeValue(forKey: id)
        }
        for (id, client) in audioClients where client.connection === connection {
            audioClients.removeValue(forKey: id)
        }
        for (id, client) in dataClients where client.connection === connection {
            dataClients.removeValue(forKey: id)
        }
        for (id, client) in depthClients where client.connection === connection {
            depthClients.removeValue(forKey: id)
        }
        clientsLock.unlock()
        notifyClientCount()
    }
}

// MARK: - Discovery

/// Answers the desktop's UDP discovery probes (docs/PROTOCOL.md §4).
///
/// Advisory: an explicit IP always works, and discovery failing must never stop
/// a manual connection. Many guest networks block broadcast entirely.
final class DiscoveryResponder {
    private var listener: NWListener?
    private let queue = DispatchQueue(label: "com.lostmediastudios.lostcam.discovery")
    var infoProvider: () -> Data = { Data("{}".utf8) }

    func start(port: UInt16 = LostCam.discoveryPort) {
        guard listener == nil else { return }
        guard let nwPort = NWEndpoint.Port(rawValue: port) else { return }
        let parameters = NWParameters.udp
        parameters.allowLocalEndpointReuse = true
        do {
            let listener = try NWListener(using: parameters, on: nwPort)
            listener.newConnectionHandler = { [weak self] connection in
                self?.handle(connection)
            }
            self.listener = listener
            listener.start(queue: queue)
        } catch {
            // Discovery is optional; a busy port is not worth surfacing.
            listener = nil
        }
    }

    func stop() {
        listener?.cancel()
        listener = nil
    }

    private func handle(_ connection: NWConnection) {
        connection.start(queue: queue)
        connection.receiveMessage { [weak self] payload, _, _, _ in
            guard let self, let payload else {
                connection.cancel()
                return
            }
            let text = String(decoding: payload, as: UTF8.self)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard text == LostCam.discoveryProbe else {
                connection.cancel()
                return
            }
            connection.send(content: self.infoProvider(),
                            completion: .contentProcessed { _ in
                connection.cancel()
            })
        }
    }
}
