package com.lostmediastudios.lostcam

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.os.PowerManager

/**
 * A foreground service, so a long capture is not killed by the system.
 *
 * This exists because of the actual use case: a rig watching a print for hours
 * needs the stream to survive the user putting the phone down, and Android will
 * happily stop a background app's camera. A foreground service with a visible
 * notification is the supported way to say "this is deliberate", and the
 * notification is a feature rather than a nuisance — a phone silently streaming
 * its camera is exactly what a user should be able to see at a glance.
 *
 * Note the honest limit: this keeps the *process* alive and the camera running
 * with the screen off, but the OS can still throttle it under thermal pressure.
 * The `battery` data channel is how you find out that happened.
 */
class StreamService : Service() {

    private var wakeLock: PowerManager.WakeLock? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val port = intent?.getIntExtra(EXTRA_PORT, LostCam.DEFAULT_PORT)
            ?: LostCam.DEFAULT_PORT
        startForeground(NOTIFICATION_ID, buildNotification(port))
        acquireWakeLock()
        // Restarting with the last intent would resume streaming the camera
        // without the user asking, which is not a decision a service should make.
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        releaseWakeLock()
        super.onDestroy()
    }

    private fun acquireWakeLock() {
        if (wakeLock != null) return
        val manager = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = manager.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK,
            "LostCam::streaming",
        ).apply {
            setReferenceCounted(false)
            // Bounded, so a forgotten lock cannot flatten the battery overnight.
            acquire(MAX_WAKE_LOCK_MILLIS)
        }
    }

    private fun releaseWakeLock() {
        wakeLock?.let {
            if (it.isHeld) it.release()
        }
        wakeLock = null
    }

    private fun buildNotification(port: Int): Notification {
        createChannel()
        val openApp = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )

        val builder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Notification.Builder(this, CHANNEL_ID)
        } else {
            @Suppress("DEPRECATION")
            Notification.Builder(this)
        }

        return builder
            .setContentTitle(getString(R.string.notification_title))
            .setContentText("Serving on port $port. Tap to stop or change channels.")
            .setSmallIcon(android.R.drawable.presence_video_online)
            .setContentIntent(openApp)
            .setOngoing(true)
            .build()
    }

    private fun createChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val manager = getSystemService(NotificationManager::class.java)
        if (manager.getNotificationChannel(CHANNEL_ID) != null) return
        manager.createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID,
                getString(R.string.channel_name),
                // Low: it must be visible, but it should not make noise.
                NotificationManager.IMPORTANCE_LOW,
            ),
        )
    }

    companion object {
        private const val CHANNEL_ID = "lostcam.streaming"
        private const val NOTIFICATION_ID = 1
        private const val MAX_WAKE_LOCK_MILLIS = 12L * 60L * 60L * 1000L
        const val EXTRA_PORT = "port"

        fun start(context: Context, port: Int) {
            val intent = Intent(context, StreamService::class.java)
                .putExtra(EXTRA_PORT, port)
            context.startForegroundService(intent)
        }

        fun stop(context: Context) {
            context.stopService(Intent(context, StreamService::class.java))
        }
    }
}
