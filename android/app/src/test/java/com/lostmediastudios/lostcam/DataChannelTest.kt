package com.lostmediastudios.lostcam

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Data channel tests. The invariant that matters most is contiguous per-subscriber
 * sequence numbering: a channel subset must not look like packet loss to a
 * consumer that reads `seq` gaps as drops, which the spec says it should.
 */
class SampleEncoderTest {
    private fun encoder() = SampleEncoder(clock = { 1000L })

    private fun decode(bytes: ByteArray) = JSONObject(String(bytes).trim())

    private val battery = SensorSample.Battery(0.5, false, "nominal")

    @Test
    fun `every record carries the required fields`() {
        val record = decode(encoder().encode(battery))
        assertEquals(1000L, record.getLong("t"))
        assertEquals(1, record.getInt("seq"))
        assertEquals("battery", record.getString("ch"))
    }

    @Test
    fun `sequence increments per record`() {
        val subject = encoder()
        assertEquals(1, decode(subject.encode(battery)).getInt("seq"))
        assertEquals(2, decode(subject.encode(battery)).getInt("seq"))
        assertEquals(3, decode(subject.encode(battery)).getInt("seq"))
    }

    @Test
    fun `each encoder numbers independently`() {
        val a = encoder()
        val b = encoder()
        a.encode(battery)
        a.encode(battery)
        assertEquals(1, decode(b.encode(battery)).getInt("seq"))
    }

    @Test
    fun `attitude carries quaternion scalar last plus euler`() {
        // A properly normalised 90° rotation about y. The truncated literal
        // 0.7071 is NOT normalised and yields yaw 89.685°, which is what made
        // this test fail in CI while passing a casual eyeball.
        val half = Math.toRadians(90.0) / 2.0
        val component = Math.sin(half)
        val record = decode(
            encoder().encode(
                SensorSample.Attitude(
                    0.0, component, 0.0, Math.cos(half), "magnetic", "high",
                ),
            ),
        )
        val quaternion = record.getJSONArray("q")
        assertEquals(4, quaternion.length())
        // Scalar last, per the spec.
        assertEquals(Math.cos(half), quaternion.getDouble(3), 1e-5)
        assertEquals(90.0, record.getJSONArray("euler").getDouble(1), 0.01)
        assertEquals("magnetic", record.getString("ref"))
    }

    @Test
    fun `gimbal lock resolves the same way as the other senders`() {
        // At yaw = 90 the convention is roll = 0 with the remainder in pitch, so
        // an iOS and an Android stream describe the same rotation identically.
        val half = Math.toRadians(90.0) / 2.0
        val angles = Maths.euler(0.0, Math.sin(half), 0.0, Math.cos(half))
        assertEquals(0.0, angles[0], 1e-6)   // pitch
        assertEquals(90.0, angles[1], 1e-3)  // yaw
        assertEquals(0.0, angles[2], 1e-12)  // roll, defined as zero
    }

    @Test
    fun `gimbal lock is stable either side of the degenerate term`() {
        // 1 - 2(x^2 + y^2) straddles zero depending on the platform's sin.
        for (y in listOf(0.7071067811865475, 0.7071067811865476)) {
            val angles = Maths.euler(0.0, y, 0.0, y)
            assertEquals(0.0, angles[0], 1e-6)
            assertEquals(90.0, angles[1], 1e-3)
        }
    }

    @Test
    fun `motion omits the magnetometer when absent`() {
        val record = decode(
            encoder().encode(
                SensorSample.Motion(
                    userAcceleration = floatArrayOf(0f, 0f, 0.1f),
                    gravity = floatArrayOf(0f, 0f, -1f),
                    rotationRate = floatArrayOf(0f, 0f, 0f),
                    magneticField = null,
                    magneticAccuracy = null,
                ),
            ),
        )
        assertNotNull(record.getJSONArray("accel"))
        assertFalse(record.has("mag"))
        assertFalse(record.has("magAccuracy"))
    }

    @Test
    fun `barometer omits altitude when absent`() {
        val record = decode(encoder().encode(SensorSample.Barometer(101.3, null)))
        assertEquals(101.3, record.getDouble("kpa"), 1e-3)
        assertFalse(record.has("relAltitude"))
    }

    @Test
    fun `location precision is preserved`() {
        val record = decode(
            encoder().encode(
                SensorSample.Location(51.5074123, -0.1278456, 5.0, null, null, null),
            ),
        )
        // Seven decimals is roughly a centimetre; fewer would quietly degrade it.
        assertEquals(51.5074123, record.getDouble("lat"), 1e-7)
        assertEquals(-0.1278456, record.getDouble("lon"), 1e-7)
    }

    @Test
    fun `channel wire names match the spec`() {
        assertEquals("attitude", Channel.ATTITUDE.wireName)
        assertEquals("motion", Channel.MOTION.wireName)
        assertEquals("barometer", Channel.BAROMETER.wireName)
        assertEquals("battery", Channel.BATTERY.wireName)
        assertEquals("location", Channel.LOCATION.wireName)
    }

    @Test
    fun `channels can be looked up by wire name`() {
        assertEquals(Channel.MOTION, Channel.fromWireName("motion"))
        assertNull(Channel.fromWireName("ar.face"))
    }

    @Test
    fun `location is the sensitive channel`() {
        assertTrue(Channel.LOCATION.isSensitive)
        for (channel in Channel.entries.filter { it != Channel.LOCATION }) {
            assertFalse(channel.isSensitive)
        }
    }
}

class ChannelFilterTest {
    @Test
    fun `empty filter means everything`() {
        assertTrue(parseChannelFilter(null).isEmpty())
        assertTrue(parseChannelFilter("").isEmpty())
    }

    @Test
    fun `parses a comma separated list`() {
        assertEquals(setOf("attitude", "motion"), parseChannelFilter("attitude,motion"))
    }

    @Test
    fun `trims whitespace and ignores blanks`() {
        assertEquals(
            setOf("attitude", "motion"),
            parseChannelFilter(" attitude , , motion "),
        )
    }
}

class DataSubscriberTest {
    private val battery = SensorSample.Battery(1.0, false, "nominal")
    private val barometer = SensorSample.Barometer(101.0, null)

    private fun subscriber(
        channels: Set<String>,
        hz: Int,
        sink: (ByteArray) -> Unit,
    ) = DataSubscriber(channels, hz, SampleEncoder(clock = { 0L }), sink)

    @Test
    fun `accepts everything when unfiltered`() {
        val subject = subscriber(emptySet(), 0) {}
        assertTrue(subject.accepts("battery"))
        assertTrue(subject.accepts("anything.new"))
    }

    @Test
    fun `rejects channels outside the filter`() {
        val subject = subscriber(setOf("battery"), 0) {}
        assertTrue(subject.accepts("battery"))
        assertFalse(subject.accepts("barometer"))
    }

    @Test
    fun `filtered subscriber still sees a contiguous sequence`() {
        // The important one: a channel subset must not look like packet loss.
        val lines = mutableListOf<ByteArray>()
        val subject = subscriber(setOf("battery"), 0) { lines.add(it) }
        for (index in 0 until 5) {
            subject.offer(battery, index.toLong())
            subject.offer(barometer, index.toLong())
        }
        assertEquals(5, lines.size)
        val sequences = lines.map { JSONObject(String(it).trim()).getInt("seq") }
        assertEquals(listOf(1, 2, 3, 4, 5), sequences)
    }

    @Test
    fun `rate cap decimates`() {
        var count = 0
        // 10 Hz means one sample per 100 ms.
        val subject = subscriber(emptySet(), 10) { count += 1 }
        for (millis in 0 until 100 step 10) subject.offer(battery, millis.toLong())
        assertEquals(1, count)
    }

    @Test
    fun `rate cap is per channel`() {
        var count = 0
        val subject = subscriber(emptySet(), 10) { count += 1 }
        subject.offer(battery, 0)
        subject.offer(barometer, 0)
        // Different channels do not compete for the same budget.
        assertEquals(2, count)
    }

    @Test
    fun `uncapped rate emits everything`() {
        var count = 0
        val subject = subscriber(emptySet(), 0) { count += 1 }
        for (millis in 0 until 10) subject.offer(battery, millis.toLong())
        assertEquals(10, count)
    }
}

class RateLimiterTest {
    @Test
    fun `zero hz means no limit`() {
        val limiter = RateLimiter(0)
        assertTrue(limiter.shouldEmit("a", 0))
        assertTrue(limiter.shouldEmit("a", 0))
    }

    @Test
    fun `emits again after the interval`() {
        val limiter = RateLimiter(10)
        assertTrue(limiter.shouldEmit("a", 0))
        assertFalse(limiter.shouldEmit("a", 50))
        assertTrue(limiter.shouldEmit("a", 100))
    }

    @Test
    fun `very high rate still has a minimum interval`() {
        // 1000/2000 would floor to zero and become a divide-by-nothing.
        assertTrue(RateLimiter(2000).intervalMillis >= 1)
    }
}

class ThermalMappingTest {
    @Test
    fun `battery temperature maps onto the spec thermal names`() {
        // Android has no direct equivalent of iOS' thermal state, so temperature
        // is mapped onto the same four names to keep the field comparable.
        assertEquals("nominal", SensorHub.thermalNameFromTemperature(300))
        assertEquals("fair", SensorHub.thermalNameFromTemperature(370))
        assertEquals("serious", SensorHub.thermalNameFromTemperature(420))
        assertEquals("critical", SensorHub.thermalNameFromTemperature(480))
    }

    @Test
    fun `unknown temperature is reported as nominal not invented`() {
        assertEquals("nominal", SensorHub.thermalNameFromTemperature(-1))
    }
}
