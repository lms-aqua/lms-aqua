import CoreLocation
import CoreMotion
import Foundation
import UIKit

/// Non-AR measurements: Core Motion, barometer, battery/thermal, and location.
///
/// Each channel starts only when it is enabled, because these are not free —
/// Core Motion at 60 Hz and the barometer both draw power, and location draws
/// both power and scrutiny.

protocol SensorHubDelegate: AnyObject {
    func sensorHub(didProduce sample: SensorSample)
    func sensorHub(didReportProblem message: String)
}

final class SensorHub: NSObject {
    weak var delegate: SensorHubDelegate?

    private let motionManager = CMMotionManager()
    private let altimeter = CMAltimeter()
    private let locationManager = CLLocationManager()
    private var batteryTimer: Timer?
    private var startedChannels: Set<Channel> = []

    /// Sample rate for the motion channels. 60 Hz is plenty; higher rates mostly
    /// generate heat and bytes.
    var motionHz: Double = 60

    override init() {
        super.init()
        locationManager.delegate = self
    }

    // MARK: Capability

    static func isAvailable(_ channel: Channel) -> Bool {
        switch channel {
        case .attitude, .motion:
            return CMMotionManager().isDeviceMotionAvailable
        case .barometer:
            return CMAltimeter.isRelativeAltitudeAvailable()
        case .battery:
            return true
        case .location:
            return CLLocationManager.locationServicesEnabled()
        case .arWorld, .arFace, .arPlanes, .light:
            return false  // owned by ARBridge
        }
    }

    // MARK: Start / stop

    func start(channels: Set<Channel>) {
        let wanted = channels.filter { !$0.requiresARSession }
        for channel in wanted where !startedChannels.contains(channel) {
            begin(channel)
        }
        for channel in startedChannels.subtracting(wanted) {
            end(channel)
        }
    }

    func stopAll() {
        for channel in startedChannels {
            end(channel)
        }
        startedChannels.removeAll()
    }

    private func begin(_ channel: Channel) {
        switch channel {
        case .attitude, .motion:
            startDeviceMotion()
        case .barometer:
            startBarometer()
        case .battery:
            startBattery()
        case .location:
            startLocation()
        default:
            return
        }
        startedChannels.insert(channel)
    }

    private func end(_ channel: Channel) {
        switch channel {
        case .attitude, .motion:
            // Only stop when neither motion channel is still wanted.
            let other: Channel = channel == .attitude ? .motion : .attitude
            if !startedChannels.contains(other) {
                motionManager.stopDeviceMotionUpdates()
            }
        case .barometer:
            altimeter.stopRelativeAltitudeUpdates()
        case .battery:
            batteryTimer?.invalidate()
            batteryTimer = nil
            UIDevice.current.isBatteryMonitoringEnabled = false
        case .location:
            locationManager.stopUpdatingLocation()
            locationManager.stopUpdatingHeading()
        default:
            break
        }
        startedChannels.remove(channel)
    }

    // MARK: Motion

    private func startDeviceMotion() {
        guard motionManager.isDeviceMotionAvailable else {
            delegate?.sensorHub(didReportProblem: "device motion is unavailable")
            return
        }
        guard !motionManager.isDeviceMotionActive else { return }

        motionManager.deviceMotionUpdateInterval = 1.0 / max(1, motionHz)
        // Magnetic north gives a yaw that means the same thing across sessions;
        // without it, yaw is relative to wherever the app happened to start.
        let reference: CMAttitudeReferenceFrame =
            CMMotionManager.availableAttitudeReferenceFrames()
                .contains(.xMagneticNorthZVertical)
            ? .xMagneticNorthZVertical : .xArbitraryZVertical

        motionManager.startDeviceMotionUpdates(using: reference, to: .main) {
            [weak self] motion, error in
            guard let self else { return }
            if let error {
                self.delegate?.sensorHub(didReportProblem:
                    "motion error: \(error.localizedDescription)")
                return
            }
            guard let motion else { return }
            self.emit(motion: motion, reference: reference)
        }
    }

    private func emit(motion: CMDeviceMotion, reference: CMAttitudeReferenceFrame) {
        let quaternion = motion.attitude.quaternion

        if startedChannels.contains(.attitude) {
            delegate?.sensorHub(didProduce: .attitude(AttitudeSample(
                x: quaternion.x, y: quaternion.y, z: quaternion.z, w: quaternion.w,
                reference: reference == .xMagneticNorthZVertical
                    ? "magnetic" : "arbitrary",
                accuracy: Self.magneticAccuracyName(motion.magneticField.accuracy))))
        }

        guard startedChannels.contains(.motion) else { return }

        // Acceleration stays in g, matching what both platforms report natively:
        // converting here would guarantee an off-by-9.81 in one of the two apps.
        let acceleration = motion.userAcceleration
        let gravity = motion.gravity
        let rotation = motion.rotationRate
        let field = motion.magneticField.field
        let hasField = motion.magneticField.accuracy != .uncalibrated

        delegate?.sensorHub(didProduce: .motion(MotionSample(
            userAcceleration: [Float(acceleration.x), Float(acceleration.y),
                               Float(acceleration.z)],
            gravity: [Float(gravity.x), Float(gravity.y), Float(gravity.z)],
            rotationRate: [Float(rotation.x), Float(rotation.y), Float(rotation.z)],
            magneticField: hasField
                ? [Float(field.x), Float(field.y), Float(field.z)] : nil,
            magneticAccuracy: Self.magneticAccuracyName(
                motion.magneticField.accuracy))))
    }

    static func magneticAccuracyName(_ accuracy: CMMagneticFieldCalibrationAccuracy)
        -> String? {
        switch accuracy {
        case .uncalibrated: return "unreliable"
        case .low: return "low"
        case .medium: return "medium"
        case .high: return "high"
        @unknown default: return nil
        }
    }

    // MARK: Barometer

    private func startBarometer() {
        guard CMAltimeter.isRelativeAltitudeAvailable() else {
            delegate?.sensorHub(didReportProblem: "this device has no barometer")
            return
        }
        altimeter.startRelativeAltitudeUpdates(to: .main) { [weak self] data, error in
            guard let self else { return }
            if let error {
                self.delegate?.sensorHub(didReportProblem:
                    "barometer error: \(error.localizedDescription)")
                return
            }
            guard let data else { return }
            self.delegate?.sensorHub(didProduce: .barometer(BarometerSample(
                kilopascals: data.pressure.doubleValue,
                relativeAltitude: data.relativeAltitude.doubleValue)))
        }
    }

    // MARK: Battery and thermal

    private func startBattery() {
        UIDevice.current.isBatteryMonitoringEnabled = true
        emitBattery()
        // 1 Hz: this channel exists to explain a frame rate that collapsed nine
        // minutes in, and that does not need a fast sample rate.
        let timer = Timer(timeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.emitBattery()
        }
        RunLoop.main.add(timer, forMode: .common)
        batteryTimer = timer
    }

    private func emitBattery() {
        let device = UIDevice.current
        let state = device.batteryState
        delegate?.sensorHub(didProduce: .battery(BatterySample(
            level: Double(max(0, device.batteryLevel)),
            charging: state == .charging || state == .full,
            thermal: Self.thermalName(ProcessInfo.processInfo.thermalState))))
    }

    static func thermalName(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "nominal"
        }
    }

    // MARK: Location

    /// Location is the one channel with its own explicit permission step, and it
    /// is never enabled implicitly (docs/PROTOCOL.md §6.5).
    private func startLocation() {
        switch locationManager.authorizationStatus {
        case .notDetermined:
            locationManager.requestWhenInUseAuthorization()
        case .denied, .restricted:
            delegate?.sensorHub(didReportProblem:
                "location permission was denied; the location channel stays off")
            return
        default:
            break
        }
        locationManager.desiredAccuracy = kCLLocationAccuracyBest
        locationManager.startUpdatingLocation()
        if CLLocationManager.headingAvailable() {
            locationManager.startUpdatingHeading()
        }
    }

    private var lastHeading: Double?
}

// MARK: - CLLocationManagerDelegate

extension SensorHub: CLLocationManagerDelegate {
    func locationManager(_ manager: CLLocationManager,
                         didUpdateLocations locations: [CLLocation]) {
        guard startedChannels.contains(.location), let location = locations.last else {
            return
        }
        delegate?.sensorHub(didProduce: .location(LocationSample(
            latitude: location.coordinate.latitude,
            longitude: location.coordinate.longitude,
            accuracy: location.horizontalAccuracy,
            altitude: location.verticalAccuracy >= 0 ? location.altitude : nil,
            speed: location.speed,
            heading: lastHeading)))
    }

    func locationManager(_ manager: CLLocationManager,
                         didUpdateHeading newHeading: CLHeading) {
        lastHeading = newHeading.trueHeading >= 0
            ? newHeading.trueHeading : newHeading.magneticHeading
    }

    func locationManager(_ manager: CLLocationManager,
                         didFailWithError error: Error) {
        delegate?.sensorHub(didReportProblem:
            "location error: \(error.localizedDescription)")
    }

    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        guard startedChannels.contains(.location) else { return }
        switch manager.authorizationStatus {
        case .authorizedWhenInUse, .authorizedAlways:
            manager.startUpdatingLocation()
        case .denied, .restricted:
            delegate?.sensorHub(didReportProblem:
                "location permission was revoked; the channel is off")
            end(.location)
        default:
            break
        }
    }
}
