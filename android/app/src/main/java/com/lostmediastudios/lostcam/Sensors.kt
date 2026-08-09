package com.lostmediastudios.lostcam

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.BatteryManager
import android.os.SystemClock
import kotlin.math.abs

/**
 * Android sensors mapped onto the wire schema in docs/SENSORS.md.
 *
 * The mapping is where the care is needed. Android and iOS report the same
 * physical quantities in different units and frames, and the spec says the
 * *sender* converts, once, so no consumer has to know which phone it is talking
 * to:
 *
 *  - acceleration: Android gives m/s^2, the wire format is g
 *  - pressure: Android gives hPa, the wire format is kPa
 *  - orientation: Android's rotation vector is already (x, y, z, w) but its
 *    reference frame is y-north/z-up, so yaw is reported with the same meaning
 *    as Core Motion's magnetic-north reference
 */
class SensorHub(
    private val context: Context,
    private val onSample: (SensorSample) -> Unit,
    private val onProblem: (String) -> Unit = {},
) : SensorEventListener {

    private val manager =
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private var enabled: Set<Channel> = emptySet()

    // Latest raw readings, combined into one `motion` record when they are all in.
    private var linearAcceleration: FloatArray? = null
    private var gravity: FloatArray? = null
    private var rotationRate: FloatArray? = null
    private var magneticField: FloatArray? = null
    private var magneticAccuracy: String? = null
    private var pressureBaseline: Float? = null
    private var lastMotionEmit = 0L
    private var lastBatteryEmit = 0L

    /** Emit at most this often per record type, matching the iOS default. */
    var motionHz: Int = 60

    private var batteryReceiver: BroadcastReceiver? = null

    // MARK: Availability

    fun isAvailable(channel: Channel): Boolean = when (channel) {
        Channel.ATTITUDE -> manager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR) != null
        Channel.MOTION -> manager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER) != null
        Channel.BAROMETER -> manager.getDefaultSensor(Sensor.TYPE_PRESSURE) != null
        Channel.BATTERY -> true
        // Deliberately not reported as available: enabling it requires the
        // runtime permission to have been granted, which the UI checks.
        Channel.LOCATION -> false
    }

    fun availableChannels(): List<Channel> =
        Channel.entries.filter { isAvailable(it) }

    // MARK: Lifecycle

    fun start(channels: Set<Channel>) {
        stop()
        enabled = channels
        // SENSOR_DELAY_GAME is ~50 Hz, which is the right order for this and far
        // cheaper than FASTEST.
        val delay = SensorManager.SENSOR_DELAY_GAME

        if (Channel.ATTITUDE in channels) {
            register(Sensor.TYPE_ROTATION_VECTOR, delay)
        }
        if (Channel.MOTION in channels) {
            register(Sensor.TYPE_LINEAR_ACCELERATION, delay)
            register(Sensor.TYPE_GRAVITY, delay)
            register(Sensor.TYPE_GYROSCOPE, delay)
            register(Sensor.TYPE_MAGNETIC_FIELD, delay)
        }
        if (Channel.BAROMETER in channels) {
            register(Sensor.TYPE_PRESSURE, SensorManager.SENSOR_DELAY_NORMAL)
        }
        if (Channel.BATTERY in channels) {
            startBattery()
        }
    }

    private fun register(type: Int, delay: Int) {
        val sensor = manager.getDefaultSensor(type)
        if (sensor == null) {
            onProblem("this device has no sensor of type $type")
            return
        }
        manager.registerListener(this, sensor, delay)
    }

    fun stop() {
        manager.unregisterListener(this)
        batteryReceiver?.let {
            try {
                context.unregisterReceiver(it)
            } catch (_: IllegalArgumentException) {
                // Not registered; nothing to undo.
            }
        }
        batteryReceiver = null
        enabled = emptySet()
        pressureBaseline = null
        lastMotionEmit = 0
        lastBatteryEmit = 0
    }

    // MARK: SensorEventListener

    override fun onSensorChanged(event: SensorEvent) {
        val now = MonotonicClock.millis()
        when (event.sensor.type) {
            Sensor.TYPE_ROTATION_VECTOR -> emitAttitude(event)
            Sensor.TYPE_LINEAR_ACCELERATION -> {
                linearAcceleration = event.values.copyOf()
                maybeEmitMotion(now)
            }
            Sensor.TYPE_GRAVITY -> {
                gravity = event.values.copyOf()
                maybeEmitMotion(now)
            }
            Sensor.TYPE_GYROSCOPE -> {
                rotationRate = event.values.copyOf()
                maybeEmitMotion(now)
            }
            Sensor.TYPE_MAGNETIC_FIELD -> {
                magneticField = event.values.copyOf()
                maybeEmitMotion(now)
            }
            Sensor.TYPE_PRESSURE -> emitPressure(event.values.firstOrNull() ?: return)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        if (sensor?.type == Sensor.TYPE_MAGNETIC_FIELD) {
            magneticAccuracy = accuracyName(accuracy)
        }
    }

    private fun emitAttitude(event: SensorEvent) {
        if (Channel.ATTITUDE !in enabled) return
        val values = event.values
        if (values.size < 4) return
        // Android's rotation vector is (x, y, z) with w at index 3 — already the
        // scalar-last order the spec requires, so no reordering.
        onSample(
            SensorSample.Attitude(
                x = values[0].toDouble(),
                y = values[1].toDouble(),
                z = values[2].toDouble(),
                w = values[3].toDouble(),
                reference = "magnetic",
                accuracy = accuracyName(event.accuracy),
            ),
        )
    }

    private fun maybeEmitMotion(now: Long) {
        if (Channel.MOTION !in enabled) return
        val interval = Maths.frameIntervalMillis(motionHz)
        if (now - lastMotionEmit < interval) return

        val acceleration = linearAcceleration ?: return
        val gravityVector = gravity ?: return
        val rotation = rotationRate ?: return
        lastMotionEmit = now

        onSample(
            SensorSample.Motion(
                // Converted to g here, once, so an iOS and an Android stream mean
                // the same thing to a consumer.
                userAcceleration = floatArrayOf(
                    Maths.toG(acceleration[0]),
                    Maths.toG(acceleration[1]),
                    Maths.toG(acceleration[2]),
                ),
                gravity = floatArrayOf(
                    Maths.toG(gravityVector[0]),
                    Maths.toG(gravityVector[1]),
                    Maths.toG(gravityVector[2]),
                ),
                rotationRate = rotation.copyOf(3),
                magneticField = magneticField?.copyOf(3),
                magneticAccuracy = magneticAccuracy,
            ),
        )
    }

    private fun emitPressure(hectopascals: Float) {
        if (Channel.BAROMETER !in enabled) return
        val baseline = pressureBaseline ?: hectopascals.also { pressureBaseline = it }
        // Relative, not absolute: absolute altitude needs a sea-level reference
        // the phone does not have.
        val relative = altitudeDifference(baseline, hectopascals)
        onSample(
            SensorSample.Barometer(
                kilopascals = Maths.hectopascalsToKilopascals(hectopascals),
                relativeAltitude = relative,
            ),
        )
    }

    private fun startBattery() {
        val receiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context?, intent: Intent?) {
                intent ?: return
                val now = MonotonicClock.millis()
                // 1 Hz is plenty: this channel exists to explain a frame rate
                // that collapsed nine minutes in.
                if (now - lastBatteryEmit < 1000) return
                lastBatteryEmit = now
                emitBattery(intent)
            }
        }
        val filter = IntentFilter(Intent.ACTION_BATTERY_CHANGED)
        val sticky = context.registerReceiver(receiver, filter)
        batteryReceiver = receiver
        sticky?.let { emitBattery(it) }
    }

    private fun emitBattery(intent: Intent) {
        if (Channel.BATTERY !in enabled) return
        val level = intent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
        val scale = intent.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
        val status = intent.getIntExtra(BatteryManager.EXTRA_STATUS, -1)
        val fraction = if (level >= 0 && scale > 0) level.toDouble() / scale else 0.0
        val charging = status == BatteryManager.BATTERY_STATUS_CHARGING ||
            status == BatteryManager.BATTERY_STATUS_FULL

        // Android exposes battery temperature in tenths of a degree Celsius, and
        // has no direct equivalent of iOS' thermal state. Mapping temperature
        // onto the same four names keeps the field comparable across platforms
        // rather than leaving it empty on Android.
        val tenths = intent.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1)
        val thermal = thermalNameFromTemperature(tenths)

        onSample(SensorSample.Battery(fraction, charging, thermal))
    }

    companion object {
        fun accuracyName(accuracy: Int): String = when (accuracy) {
            SensorManager.SENSOR_STATUS_ACCURACY_HIGH -> "high"
            SensorManager.SENSOR_STATUS_ACCURACY_MEDIUM -> "medium"
            SensorManager.SENSOR_STATUS_ACCURACY_LOW -> "low"
            else -> "unreliable"
        }

        /**
         * Approximate altitude change from a pressure change, via the standard
         * barometric formula the Android SDK itself uses.
         */
        fun altitudeDifference(baselineHPa: Float, currentHPa: Float): Double {
            if (baselineHPa <= 0f || currentHPa <= 0f) return 0.0
            val base = SensorManager.getAltitude(baselineHPa, baselineHPa)
            val now = SensorManager.getAltitude(baselineHPa, currentHPa)
            return (now - base).toDouble()
        }

        /**
         * Battery temperature in tenths of a degree Celsius, mapped onto the four
         * thermal names the spec uses. Thresholds are the points at which Android
         * devices in practice start throttling.
         */
        fun thermalNameFromTemperature(tenthsCelsius: Int): String {
            if (tenthsCelsius < 0) return "nominal"
            val celsius = tenthsCelsius / 10.0
            return when {
                celsius >= 45 -> "critical"
                celsius >= 40 -> "serious"
                celsius >= 35 -> "fair"
                else -> "nominal"
            }
        }

        /** Elapsed-time helper, kept here so callers do not reach for wall time. */
        fun elapsedMillis(): Long = SystemClock.elapsedRealtime()

        /** True when two vectors differ enough to be worth re-sending. */
        fun changedMeaningfully(a: FloatArray?, b: FloatArray?, epsilon: Float): Boolean {
            if (a == null || b == null || a.size != b.size) return true
            for (index in a.indices) {
                if (abs(a[index] - b[index]) > epsilon) return true
            }
            return false
        }
    }
}
