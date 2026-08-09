package com.lostmediastudios.lostcam

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import org.json.JSONArray
import org.json.JSONObject

class MainActivity : ComponentActivity() {

    private val server = LostCamServer()
    private val discovery = DiscoveryResponder()
    private lateinit var camera: CameraSource
    private lateinit var sensors: SensorHub

    private var isStreaming by mutableStateOf(false)
    private var status by mutableStateOf("Idle")
    private var clientCount by mutableStateOf(0)
    private var lockStatus by mutableStateOf("auto exposure / focus")
    private var isLocked by mutableStateOf(false)
    private var token by mutableStateOf("")
    private val addresses = mutableStateListOf<String>()
    private val enabledChannels = mutableStateListOf<Channel>()
    private val problems = mutableStateListOf<String>()

    private val requestCamera = registerForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        if (granted) {
            beginStreaming()
        } else {
            status = "Camera permission denied — enable it in Settings"
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        camera = CameraSource(
            context = this,
            onJpeg = { jpeg, timestamp -> server.broadcastVideo(jpeg, timestamp) },
            onProblem = { note(it) },
        )
        sensors = SensorHub(
            context = this,
            onSample = { server.broadcastSample(it) },
            onProblem = { note(it) },
        )

        server.infoProvider = { infoJson(includePort = false) }
        discovery.infoProvider = { infoJson(includePort = true) }
        server.onStateChange = { message -> runOnUiThread { status = message } }
        server.onClientCountChange = { count -> runOnUiThread { clientCount = count } }
        addresses.addAll(LostCamServer.localAddresses())

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            // Only used for the foreground-service notification, so a refusal is
            // not fatal and is not treated as one.
            registerForActivityResult(ActivityResultContracts.RequestPermission()) {}
                .launch(Manifest.permission.POST_NOTIFICATIONS)
        }

        setContent { Screen() }
    }

    override fun onDestroy() {
        stopStreaming()
        super.onDestroy()
    }

    // MARK: Streaming

    private fun startStreaming() {
        if (checkSelfPermission(Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            requestCamera.launch(Manifest.permission.CAMERA)
            return
        }
        beginStreaming()
    }

    private fun beginStreaming() {
        // The screen staying on is the difference between a capture that runs for
        // hours and one that stops when the phone dims.
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        server.token = token.ifEmpty { null }
        try {
            server.start()
        } catch (error: Exception) {
            status = "Could not start the server: ${error.message}"
            return
        }
        discovery.start()
        StreamService.start(this, server.boundPort)

        camera.start(this)
        sensors.start(enabledChannels.toSet())

        addresses.clear()
        addresses.addAll(LostCamServer.localAddresses())
        isStreaming = true
        status = "Streaming on port ${server.boundPort}"
    }

    private fun stopStreaming() {
        window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        camera.stop()
        sensors.stop()
        discovery.stop()
        server.stop()
        StreamService.stop(this)
        isStreaming = false
        status = "Idle"
    }

    private fun toggleChannel(channel: Channel) {
        if (enabledChannels.contains(channel)) {
            enabledChannels.remove(channel)
        } else {
            enabledChannels.add(channel)
        }
        if (isStreaming) sensors.start(enabledChannels.toSet())
    }

    private fun note(message: String) {
        runOnUiThread {
            problems.add(0, message)
            while (problems.size > 5) problems.removeAt(problems.size - 1)
        }
    }

    // MARK: Info

    /**
     * The `/info` capability document.
     *
     * `channels` lists only what this device can actually deliver right now. That
     * is the whole point of the field: an Android build has no `ar.*` channels, and
     * saying so is better than a consumer discovering the absence at runtime.
     */
    private fun infoJson(includePort: Boolean): String {
        val document = JSONObject()
        document.put("product", LostCam.PRODUCT)
        document.put("protocol", LostCam.PROTOCOL_VERSION)
        document.put("device", "${Build.MANUFACTURER} ${Build.MODEL}")
        document.put("platform", "android")
        document.put("os", Build.VERSION.RELEASE ?: "")
        if (includePort) document.put("port", server.boundPort)

        document.put("cameras", JSONArray(listOf("back", "front")))
        document.put(
            "video",
            JSONObject()
                .put("width", if (camera.captureWidth > 0) camera.captureWidth else 1280)
                .put("height", if (camera.captureHeight > 0) camera.captureHeight else 720)
                .put("fps", camera.frameRateCap),
        )
        document.put(
            "audio",
            JSONObject().put("rate", 44100).put("channels", 1).put("format", "s16le"),
        )

        val channels = JSONArray()
        for (channel in enabledChannels) {
            if (sensors.isAvailable(channel)) channels.put(channel.wireName)
        }
        document.put("channels", channels)

        // Recorded so a dataset can prove whether it was captured with the camera
        // locked; an unlocked run is a different kind of data.
        document.put(
            "capture",
            JSONObject()
                .put("locks", camera.lockStateDescription())
                .put("position", if (camera.facingBack) "back" else "front"),
        )
        // No LiDAR equivalent in this build, stated rather than omitted.
        document.put(
            "depth",
            JSONObject().put("available", false).put("source", "none"),
        )
        document.put(
            "clock",
            JSONObject()
                .put("mono", MonotonicClock.millis())
                .put("unix", System.currentTimeMillis()),
        )
        return document.toString()
    }

    // MARK: UI

    @Composable
    private fun Screen() {
        MaterialTheme {
            Surface(modifier = Modifier.fillMaxSize()) {
                Column(
                    modifier = Modifier
                        .verticalScroll(rememberScrollState())
                        .padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    Text("LostCam", style = MaterialTheme.typography.headlineSmall)
                    StatusCard()
                    LockCard()
                    ChannelCard()
                    AccessCard()
                    if (problems.isNotEmpty()) ProblemCard()
                    Spacer(Modifier.height(8.dp))
                    Button(
                        onClick = { if (isStreaming) stopStreaming() else startStreaming() },
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        Text(if (isStreaming) "Stop streaming" else "Start streaming")
                    }
                }
            }
        }
    }

    @Composable
    private fun StatusCard() {
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(12.dp), Arrangement.spacedBy(4.dp)) {
                Text("Status", style = MaterialTheme.typography.titleMedium)
                Text(status)
                Text("Clients: $clientCount")
                if (addresses.isEmpty()) {
                    Text("No Wi-Fi address — connect to Wi-Fi, or use adb over USB.")
                } else {
                    for (address in addresses) {
                        Text(
                            "http://$address:${server.boundPort}",
                            fontFamily = FontFamily.Monospace,
                            style = MaterialTheme.typography.bodySmall,
                        )
                        Text(
                            "lostcam pull $address",
                            fontFamily = FontFamily.Monospace,
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                }
            }
        }
    }

    @Composable
    private fun LockCard() {
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(12.dp), Arrangement.spacedBy(6.dp)) {
                Text("Capture consistency", style = MaterialTheme.typography.titleMedium)
                Row(
                    Modifier.fillMaxWidth(),
                    Arrangement.SpaceBetween,
                    Alignment.CenterVertically,
                ) {
                    Text("Lock exposure, white balance, focus")
                    Switch(
                        checked = isLocked,
                        onCheckedChange = { wanted ->
                            camera.setLocked(wanted) { message ->
                                runOnUiThread {
                                    isLocked = wanted
                                    lockStatus = message
                                }
                            }
                        },
                    )
                }
                Text(lockStatus, style = MaterialTheme.typography.bodySmall)
                Text(
                    "For a fixed rig watching a printer, lock these once the shot " +
                        "is framed. Auto exposure and focus drift over hours, and a " +
                        "model then learns the camera's reaction instead of the scene.",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }
    }

    @Composable
    private fun ChannelCard() {
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(12.dp), Arrangement.spacedBy(6.dp)) {
                Text("Data channels", style = MaterialTheme.typography.titleMedium)
                for (channel in Channel.entries) {
                    val available = sensors.isAvailable(channel)
                    Row(
                        Modifier.fillMaxWidth(),
                        Arrangement.SpaceBetween,
                        Alignment.CenterVertically,
                    ) {
                        Column(Modifier.padding(end = 12.dp)) {
                            Text(channel.label)
                            Text(
                                if (available) channel.detail
                                else "${channel.detail} (unavailable on this device)",
                                style = MaterialTheme.typography.bodySmall,
                            )
                        }
                        Switch(
                            checked = enabledChannels.contains(channel),
                            enabled = available,
                            onCheckedChange = { toggleChannel(channel) },
                        )
                    }
                }
                Text(
                    "Every channel is off until you turn it on. AR pose, face " +
                        "blendshapes and LiDAR depth are iOS-only in this version — " +
                        "/info advertises only what this phone can really send.",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }
    }

    @Composable
    private fun AccessCard() {
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(12.dp), Arrangement.spacedBy(6.dp)) {
                Text("Access", style = MaterialTheme.typography.titleMedium)
                OutlinedTextField(
                    value = token,
                    onValueChange = { token = it },
                    label = { Text("Optional token") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
                Text(
                    "Anyone who can reach this phone on the network can watch the " +
                        "stream. Set a token, or use adb over USB, on a network you " +
                        "do not control. Never port-forward this to the internet.",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }
    }

    @Composable
    private fun ProblemCard() {
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(12.dp), Arrangement.spacedBy(4.dp)) {
                Text("Recent problems", style = MaterialTheme.typography.titleMedium)
                for (problem in problems) {
                    Text(problem, style = MaterialTheme.typography.bodySmall)
                }
            }
        }
    }
}
