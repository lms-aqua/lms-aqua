package com.lostmediastudios.lostcam

import kotlin.math.abs
import kotlin.math.asin
import kotlin.math.atan2
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.roundToLong

/**
 * Wire format for docs/PROTOCOL.md, kept free of Android imports so the whole
 * file runs under plain JVM unit tests. The framing and parsing mistakes are the
 * expensive ones to find on a phone, so they live here where CI can catch them.
 */
object LostCam {
    const val PRODUCT = "LostCam"
    const val PROTOCOL_VERSION = 2
    const val DEFAULT_PORT = 4747
    const val DISCOVERY_PORT = 4748
    const val VIDEO_BOUNDARY = "lostcamframe"
    const val DISCOVERY_PROBE = "LOSTCAM_DISCOVER_V1"
}

/** A parsed request line plus headers. */
data class HttpRequest(
    val method: String,
    val path: String,
    val query: Map<String, String>,
    val headers: Map<String, String>,
) {
    /** The token supplied by query parameter or header, if any. */
    val token: String?
        get() = query["token"] ?: headers["x-lostcam-token"]

    /** An integer query parameter, clamped into a sane range. */
    fun int(key: String, default: Int, minimum: Int, maximum: Int): Int {
        val raw = query[key] ?: return default
        val value = raw.toIntOrNull() ?: return default
        return min(maximum, max(minimum, value))
    }

    companion object {
        /**
         * Parse a request head. Returns null for anything implausible rather than
         * throwing: this is fed from a socket, so malformed input is expected
         * traffic and not a programming error.
         */
        fun parse(raw: String): HttpRequest? {
            val lines = raw.replace("\r\n", "\n").split("\n")
            val requestLine = lines.firstOrNull()?.takeIf { it.isNotBlank() }
                ?: return null

            val parts = requestLine.split(" ").filter { it.isNotEmpty() }
            if (parts.size < 2) return null
            val method = parts[0].uppercase()
            val target = parts[1]

            var path = target
            var query = emptyMap<String, String>()
            val questionMark = target.indexOf('?')
            if (questionMark >= 0) {
                path = target.substring(0, questionMark)
                query = parseQuery(target.substring(questionMark + 1))
            }

            // Normalise a trailing slash so "/video/" routes like "/video",
            // while leaving "/" itself alone.
            if (path.length > 1 && path.endsWith("/")) {
                path = path.dropLast(1)
            }

            val headers = mutableMapOf<String, String>()
            for (line in lines.drop(1)) {
                if (line.isEmpty()) break
                val colon = line.indexOf(':')
                if (colon <= 0) continue
                val name = line.substring(0, colon).trim().lowercase()
                val value = line.substring(colon + 1).trim()
                if (name.isNotEmpty()) headers[name] = value
            }

            return HttpRequest(method, path, query, headers)
        }

        fun parseQuery(queryString: String): Map<String, String> {
            val out = mutableMapOf<String, String>()
            for (pair in queryString.split("&")) {
                if (pair.isEmpty()) continue
                val equals = pair.indexOf('=')
                val key: String
                val value: String
                if (equals < 0) {
                    key = decode(pair)
                    value = ""
                } else {
                    key = decode(pair.substring(0, equals))
                    value = decode(pair.substring(equals + 1))
                }
                if (key.isNotEmpty()) out[key] = value
            }
            return out
        }

        /**
         * Minimal percent-decoding. Hand-rolled rather than using URLDecoder so
         * this file stays testable on a bare JVM and behaves identically to the
         * iOS implementation.
         */
        private fun decode(value: String): String {
            if (!value.contains('%') && !value.contains('+')) return value
            val bytes = ArrayList<Byte>(value.length)
            var index = 0
            while (index < value.length) {
                when (val character = value[index]) {
                    '+' -> {
                        bytes.add(' '.code.toByte())
                        index++
                    }
                    '%' -> {
                        val hex = value.substring(
                            index + 1,
                            min(index + 3, value.length),
                        )
                        val parsed = hex.toIntOrNull(16)
                        if (hex.length == 2 && parsed != null) {
                            bytes.add(parsed.toByte())
                            index += 3
                        } else {
                            bytes.add(character.code.toByte())
                            index++
                        }
                    }
                    else -> {
                        for (byte in character.toString().toByteArray()) bytes.add(byte)
                        index++
                    }
                }
            }
            return String(bytes.toByteArray())
        }
    }
}

/** Response builders. */
object HttpResponse {
    fun streamHead(contentType: String): ByteArray = buildString {
        append("HTTP/1.1 200 OK\r\n")
        append("Content-Type: $contentType\r\n")
        append("Cache-Control: no-store\r\n")
        append("Pragma: no-cache\r\n")
        append("Connection: close\r\n\r\n")
    }.toByteArray()

    fun text(
        status: Int,
        reason: String,
        body: String,
        contentType: String = "text/plain; charset=utf-8",
    ): ByteArray {
        val payload = body.toByteArray()
        val head = buildString {
            append("HTTP/1.1 $status $reason\r\n")
            append("Content-Type: $contentType\r\n")
            append("Content-Length: ${payload.size}\r\n")
            append("Connection: close\r\n\r\n")
        }.toByteArray()
        return head + payload
    }

    fun json(body: String, status: Int = 200, reason: String = "OK"): ByteArray =
        text(status, reason, body, "application/json")

    val unauthorized: ByteArray
        get() = text(401, "Unauthorized", "unauthorized: bad or missing token")

    val notFound: ByteArray get() = text(404, "Not Found", "not found")
}

/** MJPEG framing. */
object MjpegFraming {
    fun streamContentType(boundary: String = LostCam.VIDEO_BOUNDARY): String =
        "multipart/x-mixed-replace; boundary=$boundary"

    /**
     * The per-part header preceding each JPEG. Content-Length is always sent:
     * consumers must cope without it, but making them scan when the length is
     * known is rude.
     */
    fun partHeader(
        byteCount: Int,
        timestampMillis: Long,
        boundary: String = LostCam.VIDEO_BOUNDARY,
    ): ByteArray = buildString {
        append("--$boundary\r\n")
        append("Content-Type: image/jpeg\r\n")
        append("Content-Length: $byteCount\r\n")
        append("X-LostCam-Timestamp: $timestampMillis\r\n\r\n")
    }.toByteArray()

    /** A complete part: header, payload, then the CRLF the next boundary needs. */
    fun part(
        jpeg: ByteArray,
        timestampMillis: Long,
        boundary: String = LostCam.VIDEO_BOUNDARY,
    ): ByteArray = partHeader(jpeg.size, timestampMillis, boundary) +
        jpeg + "\r\n".toByteArray()
}

/**
 * A small JSON writer for the data channel.
 *
 * Hand-rolled for the same reason as the iOS one: org.json would allocate a
 * map per sample and give no control over float formatting, and at 60 Hz across
 * several channels that churn is worth avoiding.
 */
class JsonWriter {
    private val body = StringBuilder("{")
    private var needsComma = false

    fun add(key: String, value: Int): JsonWriter = apply {
        prefix(key); body.append(value)
    }

    fun add(key: String, value: Long): JsonWriter = apply {
        prefix(key); body.append(value)
    }

    fun add(key: String, value: Boolean): JsonWriter = apply {
        prefix(key); body.append(if (value) "true" else "false")
    }

    fun add(key: String, value: String): JsonWriter = apply {
        prefix(key); body.append('"').append(escape(value)).append('"')
    }

    fun add(key: String, value: Double, decimals: Int = 5): JsonWriter = apply {
        prefix(key); body.append(number(value, decimals))
    }

    fun addFloats(key: String, values: FloatArray, decimals: Int = 5): JsonWriter =
        apply {
            prefix(key)
            body.append('[')
            values.forEachIndexed { index, value ->
                if (index > 0) body.append(',')
                body.append(number(value.toDouble(), decimals))
            }
            body.append(']')
        }

    fun addMap(key: String, values: Map<String, Float>, decimals: Int = 4): JsonWriter =
        apply {
            prefix(key)
            body.append('{')
            // Sorted so a diff of two captured streams is readable.
            values.entries.sortedBy { it.key }.forEachIndexed { index, entry ->
                if (index > 0) body.append(',')
                body.append('"').append(escape(entry.key)).append("\":")
                body.append(number(entry.value.toDouble(), decimals))
            }
            body.append('}')
        }

    private fun prefix(key: String) {
        if (needsComma) body.append(',')
        body.append('"').append(escape(key)).append("\":")
        needsComma = true
    }

    /** Finish the object and append the newline that makes it NDJSON. */
    fun finish(): ByteArray = (body.toString() + "}\n").toByteArray()

    fun jsonString(): String = "$body}"

    companion object {
        fun number(value: Double, decimals: Int): String {
            // JSON has no NaN or Infinity; emitting them yields an unparseable
            // document, so they become 0.
            if (value.isNaN() || value.isInfinite()) return "0"
            if (value == value.roundToLong().toDouble() && abs(value) < 1e15) {
                return String.format("%.1f", value)
            }
            return String.format("%.${decimals}f", value)
        }

        fun escape(value: String): String {
            val out = StringBuilder(value.length + 2)
            for (character in value) {
                when (character) {
                    '"' -> out.append("\\\"")
                    '\\' -> out.append("\\\\")
                    '\n' -> out.append("\\n")
                    '\r' -> out.append("\\r")
                    '\t' -> out.append("\\t")
                    else ->
                        if (character.code < 0x20) {
                            out.append(String.format("\\u%04x", character.code))
                        } else {
                            out.append(character)
                        }
                }
            }
            return out.toString()
        }
    }
}

/** Maths shared with the iOS implementation, kept identical on purpose. */
object Maths {
    /**
     * Above this |sin(yaw)| the rotation is at gimbal lock and pitch/roll are no
     * longer independently determined.
     */
    const val GIMBAL_LOCK_THRESHOLD = 0.99999

    /**
     * Quaternion (x, y, z, w) to (pitch, yaw, roll) in degrees.
     *
     * **Gimbal lock is handled explicitly.** At yaw = ±90° only the *sum* of
     * pitch and roll is determined, and the naive formula computes
     * `atan2(0, 1 - 2(x² + y²))` whose second argument falls on either side of
     * zero depending on the platform's `sin` — giving 0° on one machine and 180°
     * on another for identical input. Both are valid, which is exactly why it
     * cannot be left to chance.
     *
     * At the pole, roll is defined as 0 and the remaining rotation goes into
     * pitch. The desktop client and the iOS sender implement the same rule.
     */
    fun euler(x: Double, y: Double, z: Double, w: Double): DoubleArray {
        // Clamp before asin: a denormalised quaternion would otherwise yield NaN.
        val sinYaw = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
        val yaw = asin(sinYaw)

        var pitch: Double
        val roll: Double
        if (abs(sinYaw) >= GIMBAL_LOCK_THRESHOLD) {
            pitch = 2.0 * atan2(x, w)
            // Keep pitch in (-180, 180] rather than wrapping to ±360.
            if (pitch > Math.PI) {
                pitch -= 2.0 * Math.PI
            } else if (pitch < -Math.PI) {
                pitch += 2.0 * Math.PI
            }
            roll = 0.0
        } else {
            pitch = atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
            roll = atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        }

        val toDegrees = 180.0 / Math.PI
        return doubleArrayOf(pitch * toDegrees, yaw * toDegrees, roll * toDegrees)
    }

    /**
     * Android reports acceleration in m/s^2 but the wire format is in g, matching
     * what both platforms report natively on the iOS side. Converting in exactly
     * one place is what stops an off-by-9.81 appearing in one app only.
     */
    const val GRAVITY = 9.80665f

    fun toG(metresPerSecondSquared: Float): Float = metresPerSecondSquared / GRAVITY

    /** Millibar/hPa to kilopascals, the unit the spec uses. */
    fun hectopascalsToKilopascals(hPa: Float): Double = hPa / 10.0

    fun clampFps(value: Int): Int = min(60, max(1, value))

    fun jpegQuality(value: Int): Int = min(100, max(10, value))

    fun frameIntervalMillis(fps: Int): Long =
        if (fps > 0) (1000.0 / fps).roundToInt().toLong() else 0L
}
