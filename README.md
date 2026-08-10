# LostCam

Use a phone as a webcam, a microphone, and a **sensor rig** — on Windows and
Linux. DroidCam-compatible video, plus an ARKit/IMU data channel and a dataset
recorder for feeding a model.

```
  ┌─────────────────────┐                          ┌──────────────────────────┐
  │  iOS app (Swift)    │ ── GET /video  MJPEG ──▶  │  lostcam (Python)        │
  │  Android app (Kotlin)│ ── GET /data   NDJSON ──▶ │                          │
  │  Any phone browser  │ ── GET /depth  u16mm ──▶  │  ├─ virtual camera       │
  └─────────────────────┘                          │  ├─ bridge: UDP/OSC/WS   │
        phone is the server                        │  └─ dataset recorder     │
        (Wi-Fi or USB)                             └──────────────────────────┘
```

| | |
| --- | --- |
| **Video** | MJPEG over HTTP on port 4747, the same shape as DroidCam — a browser, VLC or ffmpeg can open it |
| **Virtual camera** | `v4l2loopback` on Linux, OBS Virtual Camera on Windows/macOS. Shows up in Zoom, Teams, Discord, OBS |
| **Data channel** | ARKit face blendshapes, 6DoF world pose, planes, light, IMU, barometer, battery — as NDJSON |
| **Depth** | LiDAR depth frames as u16 millimetres (iPhone/iPad Pro) |
| **Bridge** | Re-emits telemetry as UDP JSON, OSC, WebSocket or CSV for Unity, TouchDesigner, Blender or a browser |
| **Dataset recorder** | Frames + depth + telemetry, time-aligned, with a manifest your training code can read |

## Quick start

**1. Desktop:**

```bash
git clone https://github.com/lms-aqua/lostcam && cd lostcam/client
pip install -e ".[all]"
lostcam doctor          # checks your virtual camera, audio and adb setup
```

**2. Phone.** Any one of:

- **iOS app** — open `ios/` in Xcode (`brew install xcodegen && xcodegen generate`
  first), set your team, run. Or grab the unsigned IPA from CI and sign it.
- **Android app** — install the APK from CI: `adb install -r app-debug.apk`.
- **No install at all** — run `lostcam serve` and open the printed `https://…`
  URL on the phone. Works on iOS and Android.

**3. Connect:**

```bash
lostcam pull 192.168.1.42          # the IP the app shows
lostcam pull 192.168.1.42 --usb    # over USB instead (Android, needs adb)
lostcam discover                   # if you don't know the IP
```

The phone's camera now appears as a webcam. Pick "LostCam"/"OBS Virtual Camera"
in Zoom, Teams, Discord or OBS.

### No phone handy?

The desktop half can be tested on its own — this is also the fastest way to find
out whether a problem is your virtual camera or your phone:

```bash
lostcam mocksender          # in one terminal: pretends to be a phone
lostcam pull 127.0.0.1      # in another
```

If the moving test pattern shows up in Zoom, the desktop side is fine.

## Prerequisites for the virtual camera

LostCam does **not** ship a camera driver. Writing and signing one is a
multi-week project per platform, so it writes into the virtual cameras you
already have.

**Linux** — the `v4l2loopback` kernel module:

```bash
sudo apt install v4l2loopback-dkms
sudo modprobe v4l2loopback devices=1 card_label=LostCam exclusive_caps=1
```

`exclusive_caps=1` is what makes Chrome and Chromium-based apps accept the
device. Check it appeared with `ls /dev/video*`.

**Windows / macOS** — install [OBS Studio](https://obsproject.com) (26.0+),
start it once, and click **Start Virtual Camera** so the device gets registered.
After that LostCam can write to it with OBS closed.

One inherited limitation: OBS exposes a *single* virtual camera instance, so you
cannot have LostCam feeding it and OBS emitting its own output on the same device
at once. Install Unity Capture and pass `--backend unitycapture` if you need both.

## Commands

| Command | What it does |
| --- | --- |
| `lostcam pull <ip>` | Phone serves, desktop connects. Feeds the virtual camera |
| `lostcam pull <ip> --usb` | Same, tunneled over USB with `adb forward` |
| `lostcam serve` | Desktop serves the browser sender page; the phone pushes frames |
| `lostcam data <ip>` | Print the sensor/AR channel as NDJSON (pipe it to `jq`) |
| `lostcam bridge <ip> --osc --udp --ws` | Fan telemetry out to other applications |
| `lostcam record <ip> --out dir` | Record telemetry to CSV + JSONL |
| `lostcam scan <ip> --plate-mm 220` | Set up: measure the **empty** build plate and save its geometry |
| `lostcam plate <ip> --plate plate.json` | Live readout of what is on the plate, in millimetres |
| `lostcam plate <ip> --plate plate.json --web` | The same readout as a page to leave open next to the printer |
| `lostcam capture <ip> --out dir` | Record an aligned **dataset**: frames + depth + telemetry |
| `lostcam discover` | Find senders on the LAN |
| `lostcam devices` | List Android devices visible to `adb` |
| `lostcam doctor` | Diagnose the local setup |
| `lostcam mocksender` | Pretend to be a phone |

Useful flags: `--rotate 90`, `--hflip`, `--fit cover`, `--width/--height`,
`--token`, `--audio`, `--no-vcam` (decode without a virtual camera).

## The data channel

Every channel is **off until you turn it on** in the app. `/info` advertises what
that phone can actually deliver, so a consumer reads capabilities rather than
assuming them.

```bash
$ lostcam data 192.168.1.42 --channels ar.face,attitude --hz 30
{"t":41231,"seq":2,"ch":"ar.face","tracked":true,"blend":{"jawOpen":0.31,...}}
{"t":41264,"seq":3,"ch":"attitude","q":[0.01,-0.02,0,0.99],"euler":[1.3,-2.3,0.1]}
```

It is plain NDJSON over HTTP, so `curl -N http://phone:4747/data | jq` is a valid
client and no library is required.

| Channel | Carries | iOS | Android |
| --- | --- | --- | --- |
| `attitude` | Orientation quaternion + euler | ✅ | ✅ |
| `motion` | Accel, gyro, gravity, magnetometer | ✅ | ✅ |
| `barometer` | Pressure, relative altitude | ✅ | ✅ |
| `battery` | Level, charging, thermal state | ✅ | ✅ |
| `location` | Lat/lon/accuracy — **opt-in, off by default** | ✅ | ✅ |
| `ar.world` | 6DoF camera pose, tracking state, intrinsics | ✅ | — |
| `ar.face` | 52 ARKit blendshapes, eye transforms, gaze | ✅ | — |
| `ar.planes` | Detected planes, extent, classification | ✅ | — |
| `light` | Ambient lumens, colour temperature | ✅ | — |
| `/depth` | LiDAR depth, u16 millimetres | Pro models | — |

**The AR channels are iOS-only, deliberately.** ARCore's Augmented Faces API
produces a mesh and three region poses but *no blendshape coefficients*, so an
`ar.face` channel on Android could never mean what it means on iOS — and
synthesising coefficients from the mesh would be worse than absence, because a
consumer would have no way to tell measured values from invented ones. The
capability list exists so Android can omit them honestly.

Field-by-field definitions, coordinate conventions and units:
**[docs/SENSORS.md](docs/SENSORS.md)**.

### Bridging it into other software

```bash
# OSC to TouchDesigner/Blender, UDP JSON to Unity, WebSocket to a browser
lostcam bridge 192.168.1.42 --osc 127.0.0.1:9000 --udp --ws 8765
```

OSC addresses come out as `/lostcam/ar/face/blend/jawOpen`, directly bindable in
most receivers without a mapping table. Every sink is independent — a UDP
listener that is not running, or a consumer that throws, cannot take the bridge
down.

## Recording a dataset

Built for a phone on a tripod watching a 3D printer, but nothing about it is
printer-specific.

```bash
lostcam capture 192.168.1.42 \
  --out runs/2026-08-09-benchy \
  --roi 420,300,880,660 --plate-mm 220 \
  --calibrate-plate --every 15
```

Produces frames, depth rasters, and a `manifest.jsonl` with one record per frame
carrying per-frame metrics (brightness, sharpness, frame-difference, depth
statistics, height above the plate in millimetres) and every sensor channel's
latest value with its age. Alignment is by the shared monotonic clock, which is
why that clock is specified the way it is.

**Two things to know before you rely on it:**

1. **Lock exposure, white balance and focus in the app** ("Focus on centre and
   lock"). Left on auto over a four-hour print, the same physical scene drifts in
   brightness and sharpness, and a model learns the camera's reaction instead of
   the print — while appearing to validate fine, because the drift correlates with
   time and time correlates with progress.
2. **LiDAR resolves centimetres, not layer heights.** It is good for "is there
   anything on the plate", gross shape, and catastrophic failure (spaghetti,
   detachment, blobs). It cannot see a 0.2 mm layer. For fine defects the colour
   frames are the signal.

Full format, reading code, and the data-quality checklist:
**[docs/DATASET.md](docs/DATASET.md)**.

## Mapping the build plate

Scan the empty plate once, and every frame after that carries measurements of
whatever is on it:

```bash
lostcam scan 192.168.1.42 --plate-mm 220 --out plate.json   # plate must be EMPTY
lostcam plate 192.168.1.42 --plate plate.json               # live readout
lostcam capture 192.168.1.42 --plate plate.json --out runs/benchy
lostcam plate 192.168.1.42 --plate plate.json --web         # …and in a browser
```

```
[2 object(s)]  tallest   41.5 mm  occupied     3400 mm²  volume     87.2 cm³  map  93%
      id         x,y mm       size mm    h mm  area mm²  vol cm³  solid
       1        -34,-12       62x58       41.5      2938     78.1   0.86
       2         46,-38       22x22       20.0       462      9.1   0.97
```

Per object, per frame: position on the bed, bounding size, max and mean height,
footprint area, volume, and **solidity** — the fraction of its bounding box the
object fills. A cube reads ~0.97; a sprawl of spaghetti reads ~0.3, which is what
makes it a usable failure signal.

Two things make the numbers trustworthy. The scan fits a **plane** rather than a
single distance, so an angled camera still reads true heights — subtracting one
distance would report half of an empty plate as below itself. And every frame is
resampled into a **top-down orthographic grid** in plate coordinates, which
removes perspective so areas and volumes are sums rather than estimates, and
produces a fixed-scale 2.5D height map that is a far better model input than a
perspective view.

The grid's cell size is derived from the sensor, not chosen for tidiness. An
iPhone samples about every 2.2 mm at working distance, so a 1 mm grid leaves gaps
between measured cells and detects **nothing at all** — the plate reports empty
with a 60 mm cube sitting on it. `lostcam scan` computes the right cell size and
warns if you override it with something too fine.

**`--web` leaves it on a screen.** Both `plate` and `capture` will serve the same
measurements as a live page — top-down heat map with millimetre rulers, the object
table, a growth trace of the tallest point, and a banner while the nozzle is in
shot. It binds to localhost only, updates once per depth frame, and never affects
the recording; `/state.json` is the same payload for scripts.

Setup, accuracy limits and reading the height maps: **[docs/PLATE.md](docs/PLATE.md)**.

## Security

Read this bit; the defaults are chosen for a trusted LAN and nothing else.

- **Video is unauthenticated by default.** Anyone who can reach the phone's port
  can watch the camera. The app binds only while streaming is on, and shows the
  URL that is live.
- **The data channel is more sensitive than the video.** Continuous face
  blendshapes are biometric-adjacent, and 6DoF pose plus planes is a partial map
  of your room. Hence: every channel off by default, and `location` behind its own
  separate opt-in that is never implied by anything else.
- **Set a token** (`--token`, and the field in the app) on any network you do not
  control. It is compared in constant time. It is a mitigation for "same Wi-Fi as
  strangers", not a substitute for not being there.
- **USB is the private option.** With `adb forward` the port is bound on the
  desktop's loopback interface and never exposed to the network at all.
- Push mode's HTTPS uses a self-signed certificate. It stops passive sniffing on
  the LAN; it does not authenticate the server against an active attacker.
- **Never port-forward any of this to the internet.**

## Repository layout

```
client/     Python desktop client — virtual camera, bridge, dataset recorder
ios/        iOS sender (Swift/SwiftUI, AVFoundation, ARKit). XcodeGen spec
android/    Android sender (Kotlin, CameraX, SensorManager). Gradle
docs/       PROTOCOL.md, SENSORS.md, DATASET.md, RESEARCH.md
```

The browser sender lives in `client/lostcam/web/index.html` and is served by
`lostcam serve`.

## Building

CI builds both apps and runs all three test suites on every push. Artifacts are
attached to each run: `LostCam-unsigned-ipa` and `LostCam-debug-apk`.

```bash
# Desktop client
cd client && pip install -e ".[dev]" && python -m pytest -q && python -m ruff check .

# iOS  (macOS + Xcode)
cd ios && brew install xcodegen && xcodegen generate && open LostCam.xcodeproj

# Android
cd android && gradle :app:testDebugUnitTest && gradle :app:assembleDebug
```

The `.xcodeproj` and Gradle wrapper jar are **not** committed — the Xcode project
is generated from `ios/project.yml` so it cannot drift or cause pbxproj merge
conflicts, and CI provides Gradle directly rather than the repository carrying a
binary.

The IPA that CI produces is **unsigned** and will not install as-is. Sign it with
your own identity, or build from Xcode with your team selected — a free Apple ID
gives you a 7-day build on your own device.

## Design notes

Four decisions worth knowing about, all explained at length in
[docs/RESEARCH.md](docs/RESEARCH.md) and [docs/PROTOCOL.md](docs/PROTOCOL.md):

- **MJPEG, not H.264.** Every frame is independently compressed, so there is no
  encoder/decoder delay budget to pay. It costs bandwidth and buys latency; on a
  LAN that is the right trade, and it is the one DroidCam makes too.
- **The phone is the server**, matching DroidCam — except in push mode, where the
  desktop serves and the phone connects out. A web page cannot accept inbound
  connections, so supporting browsers required reversing the direction. Both feed
  an identical pipeline.
- **`getUserMedia` needs a secure context.** A phone loading `http://192.168.x.y`
  gets no camera API at all — no prompt, no error. That is why push mode serves
  HTTPS with a generated self-signed cert and a one-time browser warning.
- **`seq` is per-subscriber.** Two clients asking for different channel subsets
  cannot share one counter without one seeing gaps, and the spec defines gaps as
  dropped samples. So each subscriber encodes and numbers its own stream.

## Licence

MIT. See [LICENSE](LICENSE).

Not affiliated with DroidCam or Dev47Apps. LostCam speaks a compatible MJPEG
endpoint so existing tooling works, and is an independent implementation.
