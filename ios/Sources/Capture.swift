import AVFoundation
import CoreImage
import Foundation
import UIKit

/// Camera and microphone capture, encoding frames to JPEG for the MJPEG stream.
///
/// Uses a plain `AVCaptureSession` rather than ARKit's frames when no AR channel
/// is enabled, because a capture session is dramatically cheaper than an AR
/// session — running ARKit just to get pixels would burn battery for nothing.

protocol CaptureDelegate: AnyObject {
    func capture(didProduceJPEG jpeg: Data, timestampMillis: Int)
    func capture(didProduceAudio pcm: Data)
    func capture(didFailWith message: String)
}

final class CameraController: NSObject {
    enum Position {
        case back
        case front

        var avPosition: AVCaptureDevice.Position {
            self == .back ? .back : .front
        }
    }

    weak var delegate: CaptureDelegate?

    private let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let audioOutput = AVCaptureAudioDataOutput()
    private let videoQueue = DispatchQueue(label: "com.lostmediastudios.lostcam.video")
    private let audioQueue = DispatchQueue(label: "com.lostmediastudios.lostcam.audio")
    private let context = CIContext(options: [.useSoftwareRenderer: false])

    private(set) var position: Position = .back
    private(set) var isRunning = false

    /// JPEG quality, 0–1. Exposed because it is the main bandwidth control.
    var quality: CGFloat = 0.8
    /// Frame rate cap applied by dropping, independent of the sensor rate.
    var frameRateCap: Int = 30
    /// Set false to stop encoding entirely when nobody is watching.
    var isEncodingEnabled = true
    /// Audio is only captured when a client asked for it.
    var isAudioEnabled = false

    private var lastFrameMillis = 0
    private(set) var captureWidth = 0
    private(set) var captureHeight = 0
    private(set) var isLocked = false

    // MARK: Permissions

    static func requestCameraAccess(_ completion: @escaping (Bool) -> Void) {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            completion(true)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                DispatchQueue.main.async { completion(granted) }
            }
        default:
            completion(false)
        }
    }

    static func requestMicrophoneAccess(_ completion: @escaping (Bool) -> Void) {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            completion(true)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .audio) { granted in
                DispatchQueue.main.async { completion(granted) }
            }
        default:
            completion(false)
        }
    }

    // MARK: Configuration

    func configure(position: Position, preset: AVCaptureSession.Preset = .hd1280x720) {
        self.position = position
        videoQueue.async { [weak self] in
            guard let self else { return }
            self.session.beginConfiguration()
            defer { self.session.commitConfiguration() }

            if self.session.canSetSessionPreset(preset) {
                self.session.sessionPreset = preset
            }

            for input in self.session.inputs {
                self.session.removeInput(input)
            }

            guard let device = AVCaptureDevice.default(
                .builtInWideAngleCamera, for: .video, position: position.avPosition)
                ?? AVCaptureDevice.default(for: .video) else {
                self.report("no camera available on this device")
                return
            }

            do {
                let input = try AVCaptureDeviceInput(device: device)
                if self.session.canAddInput(input) {
                    self.session.addInput(input)
                } else {
                    self.report("could not attach the camera input")
                    return
                }
            } catch {
                self.report("camera unavailable: \(error.localizedDescription)")
                return
            }

            if !self.session.outputs.contains(self.videoOutput) {
                self.videoOutput.alwaysDiscardsLateVideoFrames = true
                self.videoOutput.videoSettings = [
                    kCVPixelBufferPixelFormatTypeKey as String:
                        kCVPixelFormatType_32BGRA
                ]
                self.videoOutput.setSampleBufferDelegate(self, queue: self.videoQueue)
                if self.session.canAddOutput(self.videoOutput) {
                    self.session.addOutput(self.videoOutput)
                }
            }

            if self.isAudioEnabled {
                self.attachAudioIfNeeded()
            }
        }
    }

    private func attachAudioIfNeeded() {
        guard let microphone = AVCaptureDevice.default(for: .audio) else { return }
        do {
            let input = try AVCaptureDeviceInput(device: microphone)
            if session.canAddInput(input) {
                session.addInput(input)
            }
            if !session.outputs.contains(audioOutput) {
                audioOutput.setSampleBufferDelegate(self, queue: audioQueue)
                if session.canAddOutput(audioOutput) {
                    session.addOutput(audioOutput)
                }
            }
        } catch {
            report("microphone unavailable: \(error.localizedDescription)")
        }
    }

    func start() {
        videoQueue.async { [weak self] in
            guard let self, !self.session.isRunning else { return }
            self.session.startRunning()
            self.isRunning = self.session.isRunning
        }
    }

    func stop() {
        videoQueue.async { [weak self] in
            guard let self, self.session.isRunning else { return }
            self.session.stopRunning()
            self.isRunning = false
        }
    }

    func switchCamera() {
        configure(position: position == .back ? .front : .back)
    }

    // MARK: Capture locks

    /// Lock exposure, white balance and focus at their current values.
    ///
    /// This exists because of what LostCam is pointed at. For a rig watching a
    /// 3D printer for hours, continuous auto-exposure and auto-white-balance are
    /// actively harmful: the same scene drifts in brightness and colour as the
    /// print grows or the room light changes, so a model trained on the frames
    /// learns the camera's reaction instead of the print. Auto-focus is worse
    /// still — it hunts, and a hunting lens produces intermittently soft frames
    /// that poison a dataset in a way that is hard to spot later.
    ///
    /// Lock once the scene is framed, and every subsequent frame is comparable.
    func lockCaptureSettings(_ locked: Bool, completion: ((String) -> Void)? = nil) {
        videoQueue.async { [weak self] in
            guard let self else { return }
            guard let device = self.currentDevice() else {
                completion?("no camera to lock")
                return
            }
            do {
                try device.lockForConfiguration()
                defer { device.unlockForConfiguration() }

                var applied: [String] = []
                var skipped: [String] = []

                if locked {
                    if device.isExposureModeSupported(.locked) {
                        device.exposureMode = .locked
                        applied.append("exposure")
                    } else {
                        skipped.append("exposure")
                    }
                    if device.isWhiteBalanceModeSupported(.locked) {
                        device.whiteBalanceMode = .locked
                        applied.append("white balance")
                    } else {
                        skipped.append("white balance")
                    }
                    if device.isFocusModeSupported(.locked) {
                        device.focusMode = .locked
                        applied.append("focus")
                    } else {
                        skipped.append("focus")
                    }
                    // Subject-area monitoring would re-trigger the very
                    // adjustments just locked out.
                    device.isSubjectAreaChangeMonitoringEnabled = false
                } else {
                    if device.isExposureModeSupported(.continuousAutoExposure) {
                        device.exposureMode = .continuousAutoExposure
                        applied.append("exposure")
                    }
                    if device.isWhiteBalanceModeSupported(
                        .continuousAutoWhiteBalance) {
                        device.whiteBalanceMode = .continuousAutoWhiteBalance
                        applied.append("white balance")
                    }
                    if device.isFocusModeSupported(.continuousAutoFocus) {
                        device.focusMode = .continuousAutoFocus
                        applied.append("focus")
                    }
                }

                self.isLocked = locked
                var message = locked
                    ? "locked: \(applied.joined(separator: ", "))"
                    : "auto: \(applied.joined(separator: ", "))"
                if !skipped.isEmpty {
                    message += " (unsupported: \(skipped.joined(separator: ", ")))"
                }
                completion?(message)
            } catch {
                completion?("could not lock the camera: \(error.localizedDescription)")
            }
        }
    }

    /// Focus and expose on a point, then hold it. Used to target the build plate
    /// rather than whatever the camera decided was interesting.
    func focusAndExpose(at point: CGPoint, thenLock: Bool = true,
                        completion: ((String) -> Void)? = nil) {
        videoQueue.async { [weak self] in
            guard let self, let device = self.currentDevice() else {
                completion?("no camera available")
                return
            }
            do {
                try device.lockForConfiguration()
                if device.isFocusPointOfInterestSupported {
                    device.focusPointOfInterest = point
                    if device.isFocusModeSupported(.autoFocus) {
                        device.focusMode = .autoFocus
                    }
                }
                if device.isExposurePointOfInterestSupported {
                    device.exposurePointOfInterest = point
                    if device.isExposureModeSupported(.autoExpose) {
                        device.exposureMode = .autoExpose
                    }
                }
                device.unlockForConfiguration()
            } catch {
                completion?("could not set the focus point: "
                            + error.localizedDescription)
                return
            }

            guard thenLock else {
                completion?("focused on the selected point")
                return
            }
            // Give the lens and metering time to settle before freezing them;
            // locking mid-adjustment freezes the wrong values.
            self.videoQueue.asyncAfter(deadline: .now() + 1.2) {
                self.lockCaptureSettings(true, completion: completion)
            }
        }
    }

    private func currentDevice() -> AVCaptureDevice? {
        for input in session.inputs {
            if let deviceInput = input as? AVCaptureDeviceInput,
               deviceInput.device.hasMediaType(.video) {
                return deviceInput.device
            }
        }
        return nil
    }

    /// Exposure/white-balance/focus settings currently in force, for /info so a
    /// dataset can record whether it was captured locked.
    func lockStateDescription() -> String {
        isLocked ? "locked" : "auto"
    }

    private func report(_ message: String) {
        DispatchQueue.main.async { [weak self] in
            self?.delegate?.capture(didFailWith: message)
        }
    }
}

// MARK: - Sample buffers

extension CameraController: AVCaptureVideoDataOutputSampleBufferDelegate,
                            AVCaptureAudioDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        if output === audioOutput {
            handleAudio(sampleBuffer)
            return
        }
        handleVideo(sampleBuffer)
    }

    private func handleVideo(_ sampleBuffer: CMSampleBuffer) {
        guard isEncodingEnabled else { return }

        // Rate cap by dropping. The sensor runs at whatever rate it likes; this
        // keeps the wire at the requested rate without a queue.
        let now = MonotonicClock.milliseconds()
        let minimumInterval = frameRateCap > 0 ? 1000 / frameRateCap : 0
        guard now - lastFrameMillis >= minimumInterval else { return }
        lastFrameMillis = now

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        captureWidth = CVPixelBufferGetWidth(pixelBuffer)
        captureHeight = CVPixelBufferGetHeight(pixelBuffer)

        guard let jpeg = Self.encodeJPEG(pixelBuffer: pixelBuffer, context: context,
                                        quality: quality) else { return }
        delegate?.capture(didProduceJPEG: jpeg, timestampMillis: now)
    }

    static func encodeJPEG(pixelBuffer: CVPixelBuffer, context: CIContext,
                           quality: CGFloat) -> Data? {
        let image = CIImage(cvPixelBuffer: pixelBuffer)
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        return context.jpegRepresentation(
            of: image, colorSpace: colorSpace,
            options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption:
                        quality])
    }

    private func handleAudio(_ sampleBuffer: CMSampleBuffer) {
        guard isAudioEnabled else { return }
        guard let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else { return }

        var length = 0
        var pointer: UnsafeMutablePointer<Int8>?
        let status = CMBlockBufferGetDataPointer(blockBuffer, atOffset: 0,
                                                lengthAtOffsetOut: nil,
                                                totalLengthOut: &length,
                                                dataPointerOut: &pointer)
        guard status == kCMBlockBufferNoErr, let pointer, length > 0 else { return }
        let pcm = Data(bytes: pointer, count: length)
        delegate?.capture(didProduceAudio: pcm)
    }
}
