# Research: how DroidCam works, and what LostCam copies

Notes gathered before writing any code, so the design choices below are
traceable to how the real thing behaves.

## DroidCam's shape

DroidCam splits into two halves:

1. **A server on the phone.** The phone app is the *server*. It opens a
   listening TCP socket (default port **4747**, configurable in the app
   between 1024–65535) and waits. The camera is not streamed anywhere until a
   client connects.
2. **A client on the desktop.** The desktop app connects out to the phone and
   pumps the decoded frames into a *virtual camera device*, so unrelated
   programs (Zoom, Teams, Discord, OBS, browsers) see what looks like an
   ordinary webcam.

The important consequence of (1): **the phone must be reachable from the
desktop**, on the same LAN/Wi-Fi, or bridged over USB. LostCam keeps this
model and adds a reverse mode for browsers (see below).

### Transport

Video is delivered as **MJPEG** — a stream of complete JPEG images, one per
frame — over HTTP. The browser-reachable endpoint is:

```
http://<phone-ip>:4747/video
```

That URL is a plain `multipart/x-mixed-replace` MJPEG stream, which is why you
can open it in a browser or point VLC/ffmpeg at it. DroidCam also speaks a
tighter custom framing on the same port for its own client (a `VIDEO_REQ`
handshake that negotiates the stream format, defined in the Linux client's
`common.h`), but the HTTP/MJPEG path is the documented, interoperable one.

MJPEG is a deliberate trade: every frame is independently compressed, so
there is no inter-frame prediction and therefore no encoder/decoder delay
budget to pay. It costs bandwidth (roughly 5–15× H.264 for the same quality)
and buys latency and near-trivial decoding. On a LAN that is the right trade,
and it is the one LostCam makes too.

### USB mode

Wi-Fi is convenient and jittery. USB is the fix, and it is not a different
protocol — it is the *same* TCP stream tunneled over the USB cable:

- **Android:** `adb forward tcp:4747 tcp:4747` makes the phone's listening
  port appear on the desktop's `127.0.0.1:4747`. The client then connects to
  localhost and neither side knows the difference. DroidCam's client invokes
  `adb` for you; with several devices attached you disambiguate with
  `ANDROID_SERIAL`. Requires USB debugging enabled on the phone.
- **iOS:** the equivalent tunnel is `usbmuxd`, which multiplexes TCP over the
  Lightning/USB-C connection.

### The virtual camera

This is the part that cannot be faked from user space alone — an application
like Zoom enumerates *operating system* camera devices, so something has to
register one.

| OS | Mechanism |
| --- | --- |
| Linux | A V4L2 loopback kernel module creates `/dev/videoN`. One process writes frames (producer), others read them as a normal webcam (consumer). DroidCam ships its own fork, `v4l2loopback-dc`. |
| Windows | A DirectShow/Media-Foundation virtual camera filter, registered system-wide. |
| macOS | A CoreMediaIO DAL plugin (and on modern macOS, a Camera Extension). |

Writing and signing a Windows camera driver, or a macOS system extension, is
its own multi-week project with signing requirements attached. **LostCam does
not write one.** It targets the virtual cameras users already have:

- **Linux:** `v4l2loopback` (the mainline module, not DroidCam's fork).
- **Windows/macOS:** the **OBS Virtual Camera** that ships with OBS Studio
  ≥ 26.0, plus Unity Capture as an alternative on Windows.

`pyvirtualcam` is the library that abstracts these three backends behind one
`send(frame)` call, so the client stays one codebase. Its documented backends
are `v4l2loopback` (Linux), `obs` (Windows/macOS) and `unitycapture`
(Windows).

One real limitation inherited from that choice: OBS exposes a *single* virtual
camera instance, so you cannot simultaneously push frames into OBS' virtual
camera from LostCam, capture that in OBS, and re-emit it from the same virtual
camera. Use Unity Capture if you need LostCam and OBS' own output live at once.

### Audio

DroidCam also carries the phone's microphone. On Linux it routes into an
**ALSA loopback** device; on Windows it installs a virtual audio device.
Again, no driver is written here: LostCam streams PCM and plays it into an
output device you choose, which you point at a loopback/virtual cable
(VB-CABLE on Windows, an ALSA loopback or a PulseAudio/PipeWire null sink on
Linux). Documented in the README rather than pretended away.

## Findings that changed the design

Four things came out of the research that the code had to account for:

1. **`getUserMedia` needs a secure context.** A phone browser loading
   `http://192.168.1.50:8080` is *not* a secure context, so the camera API is
   simply unavailable — no prompt, no camera. Only `localhost` gets an
   exemption, and the phone is not localhost. So the browser sender is served
   over **HTTPS with a generated self-signed certificate**, and the user
   accepts a one-time warning. Missing this makes a browser-based sender look
   inexplicably broken.
2. **A browser cannot be a server.** DroidCam's phone-is-the-server model is
   impossible for a web page, which can only make outbound connections. That
   forced a second, reversed transport direction — hence LostCam's two modes,
   `pull` (phone serves, matching DroidCam) and `serve` (desktop serves, phone
   pushes). Both feed the identical frame pipeline.
3. **MJPEG boundaries cannot be trusted.** Real MJPEG producers are sloppy:
   some omit `Content-Length`, some vary the boundary's leading dashes, some
   emit preamble bytes. The parser therefore handles both the length-delimited
   and the marker-scanning (`FFD8`…`FFD9`) cases, and tolerates JPEG payloads
   split across arbitrary socket reads.
4. **iOS backgrounding will stop the camera.** iOS grants no general
   background camera access; a capture session is interrupted when the app
   leaves the foreground. The app therefore keeps the screen awake and reports
   interruptions honestly rather than claiming to survive being backgrounded.

## Sources

- [Help & FAQs | DroidCam](https://droidcam.app/help/) — port 4747, port range, `/video` endpoint
- [Linux | DroidCam](https://www.dev47apps.com/droidcam/linux/) — Wi-Fi vs ADB/USB connection modes
- [Linux Client | DroidCam](https://droidcam.app/linux/) — client/CLI behaviour, `ANDROID_SERIAL`
- [V4L2 Loopback Module | dev47apps/droidcam-linux-client (DeepWiki)](https://deepwiki.com/dev47apps/droidcam-linux-client/5.1-v4l2-loopback-module) — `v4l2loopback-dc`, producer/consumer model
- [dev47apps/droidcam-linux-client (DeepWiki)](https://deepwiki.com/dev47apps/droidcam-linux-client) — GTK + CLI clients, ALSA loopback for audio, `VIDEO_REQ` in `common.h`
- [Installation and Setup | dev47apps/droidcam-linux-client (DeepWiki)](https://deepwiki.com/dev47apps/droidcam-linux-client/2-installation-and-setup) — `adb devices`, USB setup
- [letmaik/pyvirtualcam README](https://github.com/letmaik/pyvirtualcam/blob/main/README.md) — backends, OBS single-instance limitation, `v4l2loopback` setup
- [pyvirtualcam API reference](https://letmaik.github.io/pyvirtualcam/) — `send()` / `sleep_until_next_frame()`
- [Droidcam — Gentoo wiki](https://wiki.gentoo.org/wiki/Droidcam) — phone-as-server summary
- [Kyuunex/better-droidcam-linux-client](https://github.com/Kyuunex/better-droidcam-linux-client) — adb + ffmpeg + v4l2loopback pipeline as a reference implementation
- [XcodeGen](https://github.com/yonaskolb/XcodeGen) — generating the `.xcodeproj` in CI from `project.yml`
- [Signing macOS/iOS apps in a GitHub action for test runs only](https://github.com/orgs/community/discussions/175498) — unsigned CI builds
- [xcodebuild is very slow unless you set CODE_SIGNING_ALLOWED=NO (Apple Developer Forums)](https://developer.apple.com/forums/thread/766578) — the signing-disable build settings
