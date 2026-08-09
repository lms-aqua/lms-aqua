# LostCam wire protocol v2

Two transport directions, one frame format, plus a sensor/AR data channel.
Everything is HTTP/1.1, MJPEG and NDJSON, so any half of this can be tested
with `curl`, VLC or a browser.

- **Pull mode** — the *phone* listens, the desktop connects out. This is the
  DroidCam-compatible direction, used by the iOS and Android apps.
- **Push mode** — the *desktop* listens, the phone connects out over a
  WebSocket. Required for the browser sender, because a web page cannot accept
  inbound connections.

```
 pull:  [iOS/Android app : HTTP server :4747] <─GET /video ─  [desktop client]
                                             <─GET /data  ─
                                             <─GET /depth ─
 push:  [phone browser] ──WS frames──> [desktop client : HTTPS server :8443]
                                                  │
                                        video     ▼        data
                            decode → transform → virtual camera
                                                  └─ bridge → WS / UDP / OSC / CSV
```

Video is §1–3. The sensor and AR data channel is §6, depth is §7, and the field
-by-field schema lives in [SENSORS.md](SENSORS.md).

**Version negotiation.** `/info` reports `"protocol": 2`. A v2 sender must still
serve `/video` exactly as v1 described, so a v1 consumer keeps working; a v2
consumer must treat a missing `/data` or `/depth` as "not available" rather than
an error, because that is what a v1 sender (and DroidCam) looks like.

## 1. Pull mode (phone is the server)

Default port **4747**, matching DroidCam so existing tooling works.

### `GET /video` — MJPEG video stream

Response:

```
HTTP/1.1 200 OK
Content-Type: multipart/x-mixed-replace; boundary=lostcamframe
Cache-Control: no-store
Connection: close
```

Then, per frame, forever:

```
--lostcamframe\r\n
Content-Type: image/jpeg\r\n
Content-Length: <n>\r\n
X-LostCam-Timestamp: <milliseconds, monotonic>\r\n
\r\n
<n bytes of JPEG>\r\n
```

Requirements on the producer:

- Every part is a **complete** JPEG (`FFD8` … `FFD9`). No partial frames.
- `Content-Length` **must** be sent. Consumers are still required to cope with
  its absence (see §3), but LostCam senders always emit it.
- Frames are dropped, never queued, when a consumer reads slower than capture.
  Latency is the priority; a backlog is worse than a skipped frame.

Optional query parameters, all ignorable by a minimal producer:

| Param | Meaning | Default |
| --- | --- | --- |
| `w`,`h` | requested frame size | session default |
| `fps` | requested frame rate cap | 30 |
| `q` | JPEG quality, 1–100 | 80 |
| `cam` | `back` \| `front` | `back` |

### `GET /audio` — PCM audio stream

```
HTTP/1.1 200 OK
Content-Type: audio/L16; rate=44100; channels=1
```

Body is a raw, unframed stream of **signed 16-bit little-endian** samples,
mono at 44100 Hz by default. No container, no length — it ends when the socket
closes. Chosen over a framed format so a consumer can hand the bytes straight
to an audio device.

### `GET /info` — capability probe

```json
{
  "product": "LostCam",
  "protocol": 2,
  "device": "iPhone 15 Pro",
  "platform": "ios",
  "os": "18.2",
  "cameras": ["back", "front"],
  "video": { "width": 1280, "height": 720, "fps": 30 },
  "audio": { "rate": 44100, "channels": 1, "format": "s16le" },
  "channels": ["motion", "attitude", "ar.world", "ar.face", "light", "barometer", "battery"],
  "depth": { "available": true, "width": 320, "height": 240, "format": "u16mm", "source": "lidar" }
}
```

The desktop client calls this first to size the virtual camera correctly. If it
404s, the client falls back to decoding the first video frame and using its
dimensions.

`channels` is the **capability list** for the data channel (§6). It exists
because ARKit and ARCore do not offer the same things, and because the same app
offers different things depending on the device and which camera is active — a
LiDAR iPhone has `depth`, a face-tracking session has `ar.face`, a budget
Android has neither. A consumer must drive its UI from this list rather than
assuming, and must tolerate a channel disappearing mid-session (see §6.4).

### `GET /` — status page

A human-readable page, so pointing a browser at the phone confirms it is up.

## 2. Push mode (desktop is the server)

The desktop listens on **8443 over HTTPS**. HTTPS is not optional:
`navigator.mediaDevices.getUserMedia` is gated behind a secure context, and a
LAN IP over plain HTTP is not one, so the camera would be unavailable. The
client generates a self-signed certificate; the phone shows a one-time warning
to accept.

- `GET /` — serves the sender page to the phone.
- `GET /healthz` — `200 ok`, for scripts and tests.
- `GET /ws` — WebSocket upgrade (RFC 6455).

Once upgraded:

- **Binary** messages are single complete JPEG frames — the same payload as a
  pull-mode part, without the multipart wrapper.
- **Text** messages are JSON control frames:

```json
{"type": "hello",  "width": 1280, "height": 720, "fps": 30, "device": "Pixel 8"}
{"type": "bye"}
```

`hello` should precede the first binary frame so the virtual camera can be
opened at the right size. The server replies `{"type":"ready"}`.

Server→client control:

```json
{"type": "config", "fps": 24, "quality": 70}
```

A sender may ignore `config`; it is advisory throttling, not a requirement.

## 3. Consumer requirements (MJPEG parsing)

Real-world MJPEG producers are inconsistent, and a consumer that assumes the
happy path breaks on half of them. A conforming LostCam consumer must:

1. Treat the socket as a byte stream — a JPEG payload will be split across
   reads, and multiple frames will arrive in one read.
2. Accept a boundary with or without the leading `--`, and tolerate arbitrary
   preamble before the first boundary.
3. Use `Content-Length` when present.
4. When it is absent, locate the frame by scanning for the JPEG start-of-image
   marker `FFD8` and end-of-image marker `FFD9`.
5. Bound its buffer. A producer that never emits a valid frame must not grow
   the consumer's memory without limit — exceeding the cap is an error, not an
   allocation.

## 4. Discovery

Optional, so a user does not have to read an IP off the phone screen.

The desktop broadcasts a UDP datagram to port **4748**:

```
LOSTCAM_DISCOVER_V1
```

Any listening sender replies, unicast, with its `/info` JSON plus the port its
video server is on:

```json
{"product":"LostCam","protocol":1,"port":4747,"device":"iPhone 15", ...}
```

The desktop collects replies for a short window and lists them. Discovery is
advisory: an explicit address always wins, and discovery failing never blocks
a manual connection.

## 6. Data channel — sensors and AR tracking

### `GET /data` — NDJSON sample stream

```
HTTP/1.1 200 OK
Content-Type: application/x-ndjson
Cache-Control: no-store
```

One JSON object per line, `\n`-terminated, forever:

```json
{"t":1723213456789,"seq":1,"ch":"attitude","q":[0.02,-0.71,0.01,0.70],"euler":[1.4,-89.6,0.3]}
{"t":1723213456789,"seq":2,"ch":"motion","accel":[0.01,-0.02,0.98],"rot":[0.001,0.0,-0.002],"gravity":[0.0,0.0,-1.0],"mag":[21.3,-8.7,40.1]}
{"t":1723213456801,"seq":3,"ch":"ar.world","pose":[1,0,0,0, 0,1,0,0, 0,0,1,0, 0.12,1.43,-0.88,1],"state":"normal","features":842}
{"t":1723213456801,"seq":4,"ch":"ar.face","blend":{"jawOpen":0.42,"eyeBlinkLeft":0.08},"look":[0.02,-0.01,-1.0]}
```

Every record carries three required fields and then channel-specific ones:

| Field | Type | Meaning |
| --- | --- | --- |
| `t` | int | Sender's monotonic clock in **milliseconds**. Not wall time — see §6.3. |
| `seq` | int | Monotonically increasing per connection, starting at 1. Gaps mean dropped samples. |
| `ch` | string | Channel name, from the `channels` list in `/info`. |

NDJSON rather than a binary format for a deliberate reason: the whole channel
stays inspectable with `curl -N http://phone:4747/data | jq`, which matters far
more for a telemetry stream than the bytes saved. At the sample rates involved
(a few hundred records/second at worst) the cost is not the bottleneck.

Query parameters:

| Param | Meaning | Default |
| --- | --- | --- |
| `ch` | Comma-separated channel subset, e.g. `ch=ar.face,attitude` | all available |
| `hz` | Per-channel rate cap | 60 |

A sender must honour `ch` by omitting other channels entirely, and should honour
`hz` by decimating rather than buffering.

### 6.1 Channels

Names are namespaced so a consumer can filter by prefix. Full field definitions
are in [SENSORS.md](SENSORS.md); this is the index.

| Channel | Carries | iOS source | Android source |
| --- | --- | --- | --- |
| `attitude` | Device orientation quaternion + euler | Core Motion | `TYPE_ROTATION_VECTOR` |
| `motion` | User accel, rotation rate, gravity, magnetometer | Core Motion | `TYPE_ACCELEROMETER` etc. |
| `ar.world` | 6DoF camera pose, tracking state, feature count | ARKit `ARWorldTrackingConfiguration` | ARCore `Frame.camera` |
| `ar.face` | 52 blendshapes, face transform, eye transforms, look vector | ARKit `ARFaceAnchor` | ARCore Augmented Faces (region poses only — **no blendshapes**) |
| `ar.planes` | Detected plane centre, extent, alignment, id | ARKit plane anchors | ARCore `Plane` |
| `light` | Ambient intensity (lumens) and colour temperature | ARKit `ARLightEstimate` | ARCore `LightEstimate` |
| `barometer` | Pressure (kPa) and relative altitude (m) | Core Motion altimeter | `TYPE_PRESSURE` |
| `battery` | Level, charging state, thermal state | UIDevice / ProcessInfo | `BatteryManager` |
| `location` | Latitude, longitude, accuracy, speed | Core Location | `FusedLocationProvider` |

**`ar.face` is the one channel that is not portable, and pretending otherwise
would be the worst thing this spec could do.** ARKit yields 52 named blendshape
coefficients; ARCore's Augmented Faces gives a mesh and three region poses but
no blendshape coefficients at all. So on Android `ar.face` carries `regions` and
omits `blend`, and advertises itself as `ar.face` with a
`"features":["regions"]` note in `/info`. A consumer that needs blendshapes must
check for them, not assume them.

**`location` is opt-in and off by default.** It is in the table because
"measurements" reasonably includes it, not because it should be streamed
casually — see §6.5.

### 6.2 Coordinate conventions

Getting this wrong is the single most likely source of "the numbers look
plausible but everything is mirrored", so it is pinned here.

- **Poses** (`pose`, `transform`) are 16 numbers, a 4x4 matrix in **column-major**
  order — the same layout ARKit's `simd_float4x4` and OpenGL use. The
  translation is therefore elements 12, 13, 14.
- **Quaternions** are `[x, y, z, w]`, scalar last.
- **Euler angles** are degrees, `[pitch, yaw, roll]`.
- **Axes** are right-handed, y-up, as ARKit reports: +x right, +y up, +z toward
  the viewer. **Android senders convert ARCore/Sensor values into this frame
  before sending** — the conversion belongs in the sender, once, not in every
  consumer.
- **Units** are SI unless named otherwise: metres, m/s², radians/second,
  microtesla, kPa, degrees Celsius. Blendshapes are 0–1.

### 6.3 Timestamps

`t` is the sender's **monotonic** clock in milliseconds, with an arbitrary
origin. It is deliberately not wall-clock time, because the useful operations
are "how far apart were these two samples" and "which video frame does this
sample belong to", and both survive a clock that never jumps. `X-LostCam-Timestamp`
on video frames (§1) comes from the same clock, so data and video can be
correlated.

To map to wall time, `/info` may include `"clock":{"mono":<t>,"unix":<ms>}`
sampled at request time; the offset is only as good as the request's round trip,
which is honest about what it can offer.

### 6.4 Consumer requirements

1. **Ignore unknown channels and unknown fields.** The channel list will grow.
2. **Tolerate channels appearing and disappearing.** Face tracking stops when
   the face leaves frame; ARKit tracking state degrades; a plane is removed when
   merged into another. None of these are errors.
3. **Do not assume a fixed rate.** Sample intervals vary with thermal state and
   what the OS feels like doing.
4. **Use `seq` gaps to detect loss** rather than assuming delivery.
5. **Parse line-by-line, tolerating split reads.** A JSON object will arrive
   across two TCP reads. A line that fails to parse must be skipped, not fatal.

### 6.5 Data-channel privacy

The data channel is materially more sensitive than the video stream, and it is
worth being blunt about why: continuous face blendshapes are biometric-adjacent,
6DoF pose plus planes is a partial map of the room, and `location` is location.

- Every channel is **off unless the sender's UI enables it**. There is no
  "enable everything" default.
- `location` additionally requires an explicit per-session opt-in in the app,
  separate from the other channels, and is never included in `channels` unless
  the OS permission was actually granted.
- The token gate in §5 applies to `/data` and `/depth` exactly as to `/video`.
- Recording to disk (`lostcam record`) writes only the channels asked for, and
  prints the path it is writing to. Telemetry that gets recorded silently is a
  bug.

## 7. `GET /depth` — depth frames

Available only when `/info` reports `depth.available`. LiDAR on iPhone Pro/iPad
Pro via ARKit `sceneDepth`; on Android, ARCore's Depth API where supported.

```
HTTP/1.1 200 OK
Content-Type: multipart/x-mixed-replace; boundary=lostcamdepth
```

Per frame:

```
--lostcamdepth\r\n
Content-Type: application/octet-stream\r\n
Content-Length: <w*h*2>\r\n
X-LostCam-Timestamp: <ms, same clock as §6.3>\r\n
X-LostCam-Depth: <w>x<h>; format=u16mm\r\n
X-LostCam-Intrinsics: <fx>,<fy>,<cx>,<cy>\r\n
\r\n
<w*h*2 bytes>
```

The payload is a **row-major array of unsigned 16-bit little-endian
millimetres**, `0` meaning "no measurement" (out of range, absorbed, or
low confidence). u16 millimetres rather than float32 metres halves the
bandwidth for a resolution the sensor does not have anyway: 65 metres of range
at 1 mm precision comfortably exceeds what the hardware delivers.

Depth is downsampled to a fraction of the video resolution (typically 320x240,
which is close to the native sensor resolution) and its `X-LostCam-Intrinsics`
are those of the **depth** raster, not the colour frame. Alignment between the
two is the consumer's problem and needs both sets of intrinsics.

## 8. Security posture

This protocol is designed for a trusted LAN and says so plainly.

- **The video stream is unauthenticated by default.** Anyone who can reach the
  phone's port can watch the camera. The sender apps therefore bind only while
  streaming is switched on, and show the URL that is live.
- **The data channel is more sensitive than the video.** See §6.5.
- Optional shared-secret gating: when a token is configured, requests must
  carry `?token=<t>` or `X-LostCam-Token: <t>`, compared with a constant-time
  comparison. Tokens are the mitigation for "same Wi-Fi as strangers", not a
  substitute for not doing that.
- **USB mode is the private option.** With `adb forward`, the port is bound on
  the desktop's loopback interface and never exposed to the network.
- Push mode over HTTPS uses a self-signed cert, which encrypts the stream but
  does not authenticate the server — it stops passive sniffing on the LAN, not
  an active attacker who can already redirect your traffic.
- No LostCam component should ever be port-forwarded to the public internet.
