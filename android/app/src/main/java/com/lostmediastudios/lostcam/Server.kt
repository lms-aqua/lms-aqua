package com.lostmediastudios.lostcam

import java.io.BufferedOutputStream
import java.io.IOException
import java.io.InputStream
import java.net.InetAddress
import java.net.NetworkInterface
import java.net.ServerSocket
import java.net.Socket
import java.util.Collections
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/**
 * The phone-side HTTP server. The phone listens and the desktop connects out,
 * matching DroidCam, so `/video` is reachable from a browser, VLC or ffmpeg as
 * well as from the LostCam client.
 *
 * Plain `ServerSocket` on a thread pool rather than a framework: the surface is
 * three streaming routes and three small ones, and a dependency-free server is
 * one less thing to break on an odd device.
 */
class LostCamServer(private val port: Int = LostCam.DEFAULT_PORT) {

    /** A connected streaming client. */
    private class Client(val socket: Socket) {
        val output: BufferedOutputStream = BufferedOutputStream(socket.getOutputStream())
        val alive = AtomicBoolean(true)
        var subscriber: DataSubscriber? = null
        private val lock = Any()

        /**
         * Write, dropping the payload if a write is already in flight.
         *
         * A webcam a second behind is worse than one that skipped a frame, so
         * back-pressure is resolved by discarding rather than queueing.
         */
        fun write(payload: ByteArray): Boolean {
            if (!alive.get()) return false
            synchronized(lock) {
                return try {
                    output.write(payload)
                    output.flush()
                    true
                } catch (_: IOException) {
                    alive.set(false)
                    close()
                    false
                }
            }
        }

        fun close() {
            alive.set(false)
            try {
                socket.close()
            } catch (_: IOException) {
                // Already gone; nothing useful left to do.
            }
        }
    }

    enum class Route { STATUS, INFO, VIDEO, AUDIO, DATA, NOT_FOUND }

    private var serverSocket: ServerSocket? = null
    private val pool = Executors.newCachedThreadPool()
    private val videoClients = CopyOnWriteArrayList<Client>()
    private val audioClients = CopyOnWriteArrayList<Client>()
    private val dataClients = CopyOnWriteArrayList<Client>()
    private val running = AtomicBoolean(false)

    val framesSent = AtomicLong(0)
    val samplesSent = AtomicLong(0)

    /** Supplied by the app so the server never reaches into capture state. */
    var infoProvider: () -> String = { "{}" }
    var token: String? = null
    var onStateChange: ((String) -> Unit)? = null
    var onClientCountChange: ((Int) -> Unit)? = null

    val isRunning: Boolean get() = running.get()

    val boundPort: Int get() = serverSocket?.localPort ?: port

    val clientCount: Int
        get() = videoClients.size + audioClients.size + dataClients.size

    val hasVideoClients: Boolean get() = videoClients.isNotEmpty()
    val hasDataClients: Boolean get() = dataClients.isNotEmpty()

    // MARK: Lifecycle

    fun start() {
        if (running.get()) return
        val socket = ServerSocket(port)
        socket.reuseAddress = true
        serverSocket = socket
        running.set(true)
        onStateChange?.invoke("listening on port ${socket.localPort}")

        pool.execute {
            while (running.get()) {
                val client = try {
                    socket.accept()
                } catch (_: IOException) {
                    // Expected on shutdown, when the socket is closed underneath.
                    break
                }
                pool.execute { handle(client) }
            }
        }
    }

    fun stop() {
        running.set(false)
        try {
            serverSocket?.close()
        } catch (_: IOException) {
            // Nothing to do; we are tearing down anyway.
        }
        serverSocket = null

        (videoClients + audioClients + dataClients).forEach { it.close() }
        videoClients.clear()
        audioClients.clear()
        dataClients.clear()
        notifyClientCount()
        onStateChange?.invoke("stopped")
    }

    // MARK: Request handling

    private fun handle(socket: Socket) {
        try {
            socket.tcpNoDelay = true
            val head = readHead(socket.getInputStream()) ?: run {
                socket.close()
                return
            }
            val request = HttpRequest.parse(head)
            if (request == null) {
                sendAndClose(socket, HttpResponse.text(400, "Bad Request", "malformed"))
                return
            }
            if (request.method != "GET" && request.method != "HEAD") {
                sendAndClose(
                    socket,
                    HttpResponse.text(405, "Method Not Allowed", "only GET"),
                )
                return
            }
            if (!authorized(request)) {
                sendAndClose(socket, HttpResponse.unauthorized)
                return
            }

            when (routeFor(request.path)) {
                Route.STATUS -> sendAndClose(
                    socket,
                    HttpResponse.text(200, "OK", statusPage(), "text/html; charset=utf-8"),
                )
                Route.INFO -> sendAndClose(socket, HttpResponse.json(infoProvider()))
                Route.VIDEO -> startStream(socket, videoClients, null)
                Route.AUDIO -> startStream(
                    socket, audioClients, "audio/L16; rate=44100; channels=1",
                )
                Route.DATA -> startDataStream(socket, request)
                Route.NOT_FOUND -> sendAndClose(socket, HttpResponse.notFound)
            }
        } catch (_: IOException) {
            try {
                socket.close()
            } catch (_: IOException) {
                // Nothing further to do.
            }
        }
    }

    /** Read until the end of the request head, tolerating a split across reads. */
    private fun readHead(input: InputStream, limit: Int = 64 * 1024): String? {
        val buffer = StringBuilder()
        var consecutiveNewlines = 0
        while (buffer.length < limit) {
            val byte = input.read()
            if (byte < 0) return null
            val character = byte.toChar()
            if (character == '\r') continue
            buffer.append(character)
            consecutiveNewlines = if (character == '\n') consecutiveNewlines + 1 else 0
            if (consecutiveNewlines == 2) return buffer.toString()
        }
        return null
    }

    /** Constant-time token comparison, so a wrong token leaks no timing signal. */
    private fun authorized(request: HttpRequest): Boolean {
        val expected = token
        if (expected.isNullOrEmpty()) return true
        val supplied = request.token ?: return false
        val a = expected.toByteArray()
        val b = supplied.toByteArray()
        if (a.size != b.size) return false
        var difference = 0
        for (index in a.indices) difference = difference or (a[index].toInt() xor b[index].toInt())
        return difference == 0
    }

    private fun startStream(
        socket: Socket,
        clients: CopyOnWriteArrayList<Client>,
        contentType: String?,
    ) {
        val client = Client(socket)
        val head = HttpResponse.streamHead(
            contentType ?: MjpegFraming.streamContentType(),
        )
        if (!client.write(head)) return
        clients.add(client)
        notifyClientCount()
    }

    private fun startDataStream(socket: Socket, request: HttpRequest) {
        val client = Client(socket)
        if (!client.write(HttpResponse.streamHead("application/x-ndjson"))) return
        val channels = parseChannelFilter(request.query["ch"])
        val hz = request.int("hz", default = 60, minimum = 1, maximum = 240)
        client.subscriber = DataSubscriber(channels, hz) { line ->
            if (client.write(line)) samplesSent.incrementAndGet() else remove(client)
        }
        dataClients.add(client)
        notifyClientCount()
    }

    private fun sendAndClose(socket: Socket, payload: ByteArray) {
        try {
            socket.getOutputStream().apply {
                write(payload)
                flush()
            }
        } catch (_: IOException) {
            // The peer went away mid-response; nothing to report.
        } finally {
            try {
                socket.close()
            } catch (_: IOException) {
                // Already closed.
            }
        }
    }

    // MARK: Broadcasting

    fun broadcastVideo(jpeg: ByteArray, timestampMillis: Long) {
        if (videoClients.isEmpty()) return
        val payload = MjpegFraming.part(jpeg, timestampMillis)
        for (client in videoClients) {
            if (client.write(payload)) framesSent.incrementAndGet() else remove(client)
        }
    }

    fun broadcastAudio(pcm: ByteArray) {
        for (client in audioClients) {
            if (!client.write(pcm)) remove(client)
        }
    }

    /** Offer a sample to every data client; each encodes and numbers its own. */
    fun broadcastSample(sample: SensorSample) {
        if (dataClients.isEmpty()) return
        val now = MonotonicClock.millis()
        for (client in dataClients) {
            client.subscriber?.offer(sample, now)
        }
    }

    private fun remove(client: Client) {
        client.close()
        videoClients.remove(client)
        audioClients.remove(client)
        dataClients.remove(client)
        notifyClientCount()
    }

    private fun notifyClientCount() {
        onClientCountChange?.invoke(clientCount)
    }

    private fun statusPage(): String = """
        <!doctype html><meta charset=utf-8>
        <meta name=viewport content="width=device-width,initial-scale=1">
        <title>LostCam</title>
        <style>body{font:16px system-ui;margin:2rem;line-height:1.5}
        code{background:#eee;padding:.1em .3em;border-radius:3px}</style>
        <h1>LostCam is streaming</h1>
        <ul>
          <li><a href="/video">/video</a> &mdash; MJPEG video</li>
          <li><a href="/info">/info</a> &mdash; capabilities</li>
          <li><code>/data</code> &mdash; sensor channel (NDJSON)</li>
        </ul>
        <p>On your computer: <code>lostcam pull &lt;this-phone-ip&gt;</code></p>
    """.trimIndent()

    companion object {
        fun routeFor(path: String): Route = when (path) {
            "", "/" -> Route.STATUS
            "/info" -> Route.INFO
            "/video" -> Route.VIDEO
            "/audio" -> Route.AUDIO
            "/data" -> Route.DATA
            else -> Route.NOT_FOUND
        }

        /** This device's LAN addresses, so the app can show the URL to type. */
        fun localAddresses(): List<String> {
            val out = mutableListOf<String>()
            try {
                for (nic in Collections.list(NetworkInterface.getNetworkInterfaces())) {
                    if (!nic.isUp || nic.isLoopback) continue
                    for (address in Collections.list(nic.inetAddresses)) {
                        val text = address.hostAddress ?: continue
                        // IPv4 only: a link-local IPv6 address is not something a
                        // user can usefully type into a desktop client.
                        if (address.isLoopbackAddress) continue
                        if (text.contains(':')) continue
                        out.add(text)
                    }
                }
            } catch (_: Exception) {
                return emptyList()
            }
            return out.sorted()
        }

        fun broadcastAddress(): InetAddress? = try {
            InetAddress.getByName("255.255.255.255")
        } catch (_: Exception) {
            null
        }
    }
}
