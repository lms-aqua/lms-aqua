package com.lostmediastudios.lostcam

import java.util.UUID

/**
 * The data channel (docs/PROTOCOL.md §6, docs/SENSORS.md).
 *
 * Same architecture as the iOS app, and for the same reason: sources emit typed
 * samples and each subscriber encodes them itself, because `seq` must be
 * contiguous per subscriber and two subscribers with different channel filters
 * cannot share a counter without one of them seeing gaps — which the spec defines
 * as dropped samples.
 *
 * What Android does *not* offer is as important as what it does. The `ar.*`
 * channels are absent in this version: ARCore's Augmented Faces API produces a
 * mesh and region poses but no blendshape coefficients, so an `ar.face` channel
 * here could never mean what it means on iOS, and inventing coefficients would be
 * worse than absence. `/info` advertises only what is real.
 */
enum class Channel(val wireName: String, val label: String, val detail: String) {
    ATTITUDE("attitude", "Orientation", "Quaternion and euler angles. Cheap."),
    MOTION("motion", "Motion & IMU", "Acceleration, rotation rate, gravity, magnetometer."),
    BAROMETER("barometer", "Barometer", "Pressure and relative altitude."),
    BATTERY("battery", "Battery & thermal", "Explains why the frame rate dropped."),
    LOCATION("location", "Location", "Off by default. Needs its own permission."),
    ;

    /** Treated separately everywhere; never enabled implicitly. */
    val isSensitive: Boolean get() = this == LOCATION

    companion object {
        fun fromWireName(name: String): Channel? =
            entries.firstOrNull { it.wireName == name }
    }
}

// MARK: - Typed samples

sealed interface SensorSample {
    val channel: Channel

    data class Attitude(
        val x: Double,
        val y: Double,
        val z: Double,
        val w: Double,
        val reference: String,
        val accuracy: String?,
    ) : SensorSample {
        override val channel get() = Channel.ATTITUDE
    }

    data class Motion(
        /** In g, not m/s^2 — see Maths.toG. */
        val userAcceleration: FloatArray,
        val gravity: FloatArray,
        val rotationRate: FloatArray,
        val magneticField: FloatArray?,
        val magneticAccuracy: String?,
    ) : SensorSample {
        override val channel get() = Channel.MOTION
    }

    data class Barometer(
        val kilopascals: Double,
        val relativeAltitude: Double?,
    ) : SensorSample {
        override val channel get() = Channel.BAROMETER
    }

    data class Battery(
        val level: Double,
        val charging: Boolean,
        val thermal: String,
    ) : SensorSample {
        override val channel get() = Channel.BATTERY
    }

    data class Location(
        val latitude: Double,
        val longitude: Double,
        val accuracy: Double,
        val altitude: Double?,
        val speed: Double?,
        val heading: Double?,
    ) : SensorSample {
        override val channel get() = Channel.LOCATION
    }
}

// MARK: - Encoding

/**
 * Encodes typed samples as NDJSON lines, numbering its own stream.
 * One instance per subscriber, which is what keeps `seq` contiguous.
 */
class SampleEncoder(private val clock: () -> Long = { MonotonicClock.millis() }) {
    private var sequence = 0

    val emitted: Int get() = sequence

    fun reset() {
        sequence = 0
    }

    fun encode(sample: SensorSample): ByteArray {
        val writer = begin(sample.channel)
        when (sample) {
            is SensorSample.Attitude -> {
                writer.addFloats(
                    "q",
                    floatArrayOf(
                        sample.x.toFloat(), sample.y.toFloat(),
                        sample.z.toFloat(), sample.w.toFloat(),
                    ),
                    decimals = 6,
                )
                val angles = Maths.euler(sample.x, sample.y, sample.z, sample.w)
                writer.addFloats(
                    "euler",
                    floatArrayOf(
                        angles[0].toFloat(), angles[1].toFloat(), angles[2].toFloat(),
                    ),
                    decimals = 3,
                )
                writer.add("ref", sample.reference)
                sample.accuracy?.let { writer.add("accuracy", it) }
            }

            is SensorSample.Motion -> {
                writer.addFloats("accel", sample.userAcceleration)
                writer.addFloats("gravity", sample.gravity)
                writer.addFloats("rot", sample.rotationRate)
                sample.magneticField?.let { writer.addFloats("mag", it, decimals = 3) }
                sample.magneticAccuracy?.let { writer.add("magAccuracy", it) }
            }

            is SensorSample.Barometer -> {
                writer.add("kpa", sample.kilopascals, decimals = 4)
                sample.relativeAltitude?.let {
                    writer.add("relAltitude", it, decimals = 3)
                }
            }

            is SensorSample.Battery -> {
                writer.add("level", sample.level, decimals = 3)
                writer.add("charging", sample.charging)
                writer.add("thermal", sample.thermal)
            }

            is SensorSample.Location -> {
                writer.add("lat", sample.latitude, decimals = 7)
                writer.add("lon", sample.longitude, decimals = 7)
                writer.add("accuracy", sample.accuracy, decimals = 2)
                sample.altitude?.let { writer.add("altitude", it, decimals = 2) }
                sample.speed?.let { writer.add("speed", it, decimals = 3) }
                sample.heading?.let { writer.add("heading", it, decimals = 2) }
            }
        }
        return writer.finish()
    }

    /** Start a record, stamping the required t/seq/ch fields. */
    private fun begin(channel: Channel): JsonWriter {
        sequence += 1
        return JsonWriter()
            .add("t", clock())
            .add("seq", sequence)
            .add("ch", channel.wireName)
    }
}

object MonotonicClock {
    /**
     * Monotonic milliseconds, matching PROTOCOL.md §6.3. `nanoTime` rather than
     * `currentTimeMillis` deliberately: the useful questions are "how far apart"
     * and "which video frame", and both survive a clock that never jumps.
     */
    fun millis(): Long = System.nanoTime() / 1_000_000L
}

// MARK: - Rate limiting

/**
 * Per-channel decimation for the `hz` query parameter. Decimates rather than
 * buffers: a late sample is worthless, so dropping it is the right response.
 */
class RateLimiter(hz: Int) {
    val intervalMillis: Long = if (hz > 0) maxOf(1L, 1000L / hz) else 0L
    private val lastEmitted = mutableMapOf<String, Long>()

    fun shouldEmit(channel: String, nowMillis: Long): Boolean {
        if (intervalMillis <= 0) return true
        val last = lastEmitted[channel]
        if (last != null && nowMillis - last < intervalMillis) return false
        lastEmitted[channel] = nowMillis
        return true
    }
}

// MARK: - Subscribers

/** One `/data` subscriber: channel filter, rate cap, and its own numbering. */
class DataSubscriber(
    private val wanted: Set<String>,
    hz: Int,
    private val encoder: SampleEncoder = SampleEncoder(),
    private val send: (ByteArray) -> Unit,
) {
    val id: UUID = UUID.randomUUID()
    private val limiter = RateLimiter(hz)

    /** An empty channel set means "everything available". */
    fun accepts(channel: String): Boolean = wanted.isEmpty() || wanted.contains(channel)

    val emitted: Int get() = encoder.emitted

    /**
     * Encode and deliver, if this subscriber wants it and is not saturated.
     * Encoding happens after the filter, so asking for one channel costs nothing
     * for the others.
     */
    fun offer(sample: SensorSample, nowMillis: Long) {
        val channel = sample.channel.wireName
        if (!accepts(channel)) return
        if (!limiter.shouldEmit(channel, nowMillis)) return
        send(encoder.encode(sample))
    }
}

/** Parse the `ch` query parameter into a channel set. */
fun parseChannelFilter(raw: String?): Set<String> {
    if (raw.isNullOrEmpty()) return emptySet()
    return raw.split(",").map { it.trim() }.filter { it.isNotEmpty() }.toSet()
}
