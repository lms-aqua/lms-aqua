package com.lostmediastudios.lostcam

import java.io.IOException
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Answers the desktop's UDP discovery probes (docs/PROTOCOL.md §4).
 *
 * Advisory only: an explicit IP always works, and discovery failing must never
 * stop a manual connection. Plenty of networks — most guest and corporate Wi-Fi —
 * drop broadcast traffic entirely, which is why this is a convenience and not the
 * connection mechanism.
 */
class DiscoveryResponder(private val port: Int = LostCam.DISCOVERY_PORT) {
    private val executor = Executors.newSingleThreadExecutor()
    private val running = AtomicBoolean(false)
    private var socket: DatagramSocket? = null

    var infoProvider: () -> String = { "{}" }

    fun start() {
        if (running.get()) return
        running.set(true)
        executor.execute {
            try {
                val bound = DatagramSocket(port).apply {
                    reuseAddress = true
                    broadcast = true
                    soTimeout = 500
                }
                socket = bound
                serve(bound)
            } catch (_: IOException) {
                // A busy port is not worth surfacing: discovery is optional.
                running.set(false)
            }
        }
    }

    private fun serve(bound: DatagramSocket) {
        val buffer = ByteArray(2048)
        while (running.get()) {
            val packet = DatagramPacket(buffer, buffer.size)
            try {
                bound.receive(packet)
            } catch (_: IOException) {
                continue  // timeout, or shutting down
            }
            val text = String(packet.data, 0, packet.length).trim()
            if (text != LostCam.DISCOVERY_PROBE) continue

            val payload = infoProvider().toByteArray()
            try {
                bound.send(
                    DatagramPacket(payload, payload.size, packet.address, packet.port),
                )
            } catch (_: IOException) {
                // The prober went away; nothing to do.
            }
        }
    }

    fun stop() {
        running.set(false)
        try {
            socket?.close()
        } catch (_: Exception) {
            // Already closed.
        }
        socket = null
    }
}
