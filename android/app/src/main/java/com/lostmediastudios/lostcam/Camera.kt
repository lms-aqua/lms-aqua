package com.lostmediastudios.lostcam

import android.content.Context
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.camera2.interop.Camera2CameraControl
import androidx.camera.camera2.interop.CaptureRequestOptions
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.lifecycle.LifecycleOwner
import android.util.Size
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors

/**
 * CameraX capture, encoding each frame to JPEG for the MJPEG stream.
 *
 * `ImageAnalysis` in `STRATEGY_KEEP_ONLY_LATEST` is the right primitive here:
 * the phone should hand over the newest frame and drop anything it could not keep
 * up with, which is exactly the latency-over-completeness trade the whole
 * protocol makes.
 */
class CameraSource(
    private val context: Context,
    private val onJpeg: (ByteArray, Long) -> Unit,
    private val onProblem: (String) -> Unit = {},
) {
    private val executor = Executors.newSingleThreadExecutor()
    private var provider: ProcessCameraProvider? = null
    private var camera: Camera? = null
    private var lastFrameMillis = 0L

    var quality: Int = 80
        set(value) { field = Maths.jpegQuality(value) }

    var frameRateCap: Int = 30
        set(value) { field = Maths.clampFps(value) }

    var isEncodingEnabled: Boolean = true
    var facingBack: Boolean = true

    var captureWidth: Int = 0
        private set
    var captureHeight: Int = 0
        private set
    var isLocked: Boolean = false
        private set

    fun start(owner: LifecycleOwner, width: Int = 1280, height: Int = 720) {
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            try {
                val cameraProvider = future.get()
                provider = cameraProvider
                bind(cameraProvider, owner, width, height)
            } catch (error: Exception) {
                onProblem("could not open the camera: ${error.message}")
            }
        }, executor)
    }

    private fun bind(
        cameraProvider: ProcessCameraProvider,
        owner: LifecycleOwner,
        width: Int,
        height: Int,
    ) {
        cameraProvider.unbindAll()

        val analysis = ImageAnalysis.Builder()
            .setResolutionSelector(
                ResolutionSelector.Builder()
                    .setResolutionStrategy(
                        ResolutionStrategy(
                            Size(width, height),
                            ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER,
                        ),
                    )
                    .build(),
            )
            // Newest frame wins; a backlog would be latency the viewer sees.
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
            .build()

        analysis.setAnalyzer(executor) { image -> handle(image) }

        val selector = if (facingBack) {
            CameraSelector.DEFAULT_BACK_CAMERA
        } else {
            CameraSelector.DEFAULT_FRONT_CAMERA
        }

        camera = try {
            cameraProvider.bindToLifecycle(owner, selector, analysis)
        } catch (error: Exception) {
            onProblem("could not bind the camera: ${error.message}")
            null
        }
    }

    fun stop() {
        provider?.unbindAll()
        camera = null
    }

    fun switchCamera(owner: LifecycleOwner) {
        facingBack = !facingBack
        provider?.let { bind(it, owner, 1280, 720) }
    }

    private fun handle(image: ImageProxy) {
        try {
            if (!isEncodingEnabled) return
            // Rate cap by dropping: the sensor runs at whatever rate it likes.
            val now = MonotonicClock.millis()
            val interval = Maths.frameIntervalMillis(frameRateCap)
            if (now - lastFrameMillis < interval) return
            lastFrameMillis = now

            captureWidth = image.width
            captureHeight = image.height
            val jpeg = encodeJpeg(image, quality) ?: return
            onJpeg(jpeg, now)
        } finally {
            // Always closed, or CameraX stops delivering frames after a handful.
            image.close()
        }
    }

    // MARK: Capture locks

    /**
     * Lock exposure, white balance and focus.
     *
     * This is the feature that matters for a fixed rig watching a printer for
     * hours: continuous auto-exposure and auto-white-balance make the same scene
     * drift in brightness and colour as the print grows, and auto-focus hunts and
     * produces intermittently soft frames. A model trained on those learns the
     * camera's reaction instead of the print.
     */
    @OptIn(ExperimentalCamera2Interop::class)
    fun setLocked(locked: Boolean, onResult: (String) -> Unit = {}) {
        val activeCamera = camera ?: run {
            onResult("no camera to lock")
            return
        }
        val control = Camera2CameraControl.from(activeCamera.cameraControl)
        val options = CaptureRequestOptions.Builder().apply {
            setCaptureRequestOption(
                android.hardware.camera2.CaptureRequest.CONTROL_AE_LOCK, locked,
            )
            setCaptureRequestOption(
                android.hardware.camera2.CaptureRequest.CONTROL_AWB_LOCK, locked,
            )
            setCaptureRequestOption(
                android.hardware.camera2.CaptureRequest.CONTROL_AF_MODE,
                if (locked) {
                    android.hardware.camera2.CameraMetadata.CONTROL_AF_MODE_OFF
                } else {
                    android.hardware.camera2.CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_VIDEO
                },
            )
        }.build()

        control.setCaptureRequestOptions(options)
        isLocked = locked
        onResult(
            if (locked) {
                "locked: exposure, white balance, focus"
            } else {
                "auto: exposure, white balance, focus"
            },
        )
    }

    fun lockStateDescription(): String = if (isLocked) "locked" else "auto"

    companion object {
        /**
         * YUV_420_888 to JPEG.
         *
         * `YuvImage` is used rather than a Bitmap round trip because it compresses
         * straight from NV21 without allocating an ARGB intermediate — at 30 fps
         * that intermediate is the difference between keeping up and not.
         */
        fun encodeJpeg(image: ImageProxy, quality: Int): ByteArray? {
            if (image.format != ImageFormat.YUV_420_888) return null
            val nv21 = yuv420ToNv21(image) ?: return null
            val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
            val output = ByteArrayOutputStream(image.width * image.height / 4)
            val rect = Rect(0, 0, image.width, image.height)
            return if (yuvImage.compressToJpeg(rect, Maths.jpegQuality(quality), output)) {
                output.toByteArray()
            } else {
                null
            }
        }

        /**
         * Repack the planar YUV planes into NV21.
         *
         * Row and pixel strides must be honoured rather than assumed contiguous:
         * plenty of devices pad rows, and ignoring that produces the classic
         * sheared, green-striped image.
         */
        fun yuv420ToNv21(image: ImageProxy): ByteArray? {
            val planes = image.planes
            if (planes.size < 3) return null
            val width = image.width
            val height = image.height
            val output = ByteArray(width * height * 3 / 2)

            // Y plane, row by row.
            var offset = 0
            val yPlane = planes[0]
            val yBuffer = yPlane.buffer
            val yRowStride = yPlane.rowStride
            val yPixelStride = yPlane.pixelStride
            for (row in 0 until height) {
                if (yPixelStride == 1) {
                    yBuffer.position(row * yRowStride)
                    yBuffer.get(output, offset, width)
                    offset += width
                } else {
                    for (column in 0 until width) {
                        output[offset++] = yBuffer.get(row * yRowStride + column * yPixelStride)
                    }
                }
            }

            // NV21 wants interleaved VU, at half resolution in both axes.
            val uPlane = planes[1]
            val vPlane = planes[2]
            val uBuffer = uPlane.buffer
            val vBuffer = vPlane.buffer
            val chromaHeight = height / 2
            val chromaWidth = width / 2
            for (row in 0 until chromaHeight) {
                for (column in 0 until chromaWidth) {
                    val uIndex = row * uPlane.rowStride + column * uPlane.pixelStride
                    val vIndex = row * vPlane.rowStride + column * vPlane.pixelStride
                    if (uIndex >= uBuffer.limit() || vIndex >= vBuffer.limit()) continue
                    output[offset++] = vBuffer.get(vIndex)
                    output[offset++] = uBuffer.get(uIndex)
                }
            }
            return output
        }
    }
}
