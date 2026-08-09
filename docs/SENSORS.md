# Data channel schema

Field-by-field reference for the `/data` channel in
[PROTOCOL.md](PROTOCOL.md) §6. Conventions (column-major matrices,
`[x,y,z,w]` quaternions, right-handed y-up, SI units) are pinned in §6.2 and are
not repeated per field.

Every record has `t` (monotonic ms), `seq` (per-connection counter) and `ch`.

## `attitude`

Device orientation. The cheapest useful channel — fused, drift-corrected, and
available on every phone.

| Field | Type | Meaning |
| --- | --- | --- |
| `q` | float[4] | Orientation quaternion `[x,y,z,w]`, device→world |
| `euler` | float[3] | `[pitch, yaw, roll]` in degrees, convenience only |
| `ref` | string | Reference frame: `"magnetic"`, `"true"` (north-aligned) or `"arbitrary"` |
| `accuracy` | string? | `"high"`, `"medium"`, `"low"`, `"unreliable"` when known |

`ref` matters: without a magnetometer fix, yaw is relative to wherever the
session started and comparing it across sessions is meaningless.

## `motion`

Raw-ish inertial measurements, gravity already separated out where the platform
does so.

| Field | Type | Meaning |
| --- | --- | --- |
| `accel` | float[3] | User acceleration, gravity **removed**, in g |
| `gravity` | float[3] | Gravity vector, in g |
| `rot` | float[3] | Rotation rate, radians/second |
| `mag` | float[3]? | Magnetic field, microtesla; absent without a magnetometer |
| `magAccuracy` | string? | Calibration quality; `"unreliable"` means figure-8 the phone |

Acceleration is in **g**, not m/s², because both platforms report it that way
natively and converting in the sender would invite an off-by-9.81 in exactly one
of the two apps. Multiply by 9.80665 for m/s².

## `ar.world`

6DoF camera pose from visual-inertial odometry. Requires an AR session, which
costs battery and heat — this is why it is off by default.

| Field | Type | Meaning |
| --- | --- | --- |
| `pose` | float[16] | Camera transform, 4x4 column-major, world→camera |
| `state` | string | `"normal"`, `"limited"`, `"initializing"`, `"unavailable"` |
| `reason` | string? | When `limited`: `"excessiveMotion"`, `"insufficientFeatures"`, `"relocalizing"`, `"initializing"` |
| `features` | int? | Tracked feature-point count, a rough confidence proxy |
| `intrinsics` | float[4]? | `[fx, fy, cx, cy]` of the colour camera, pixels |
| `resolution` | int[2]? | `[width, height]` the intrinsics refer to |

**Origin is where the session started**, not anywhere absolute. Restarting
tracking silently redefines the origin, which is what `state:"initializing"`
warns about. Intrinsics are included because a pose without them cannot be used
to project anything.

## `ar.face`

The non-portable channel. Read §6.1 before relying on it.

**iOS (ARKit):**

| Field | Type | Meaning |
| --- | --- | --- |
| `blend` | object | Blendshape name → coefficient, 0–1. Up to 52 keys |
| `transform` | float[16] | Face anchor transform, 4x4 column-major |
| `leftEye` | float[16]? | Left eye transform |
| `rightEye` | float[16]? | Right eye transform |
| `look` | float[3]? | Gaze direction in face space |
| `tracked` | bool | `false` when the anchor exists but is not currently tracked |

Blendshape keys are ARKit's own names, verbatim and camel-cased —
`jawOpen`, `eyeBlinkLeft`, `mouthSmileRight`, `browInnerUp`, and so on. They are
passed through unrenamed so that existing ARKit tooling and any of the many
blendshape→rig mappings work without a translation table.

Only **non-zero** coefficients are sent, to keep the line short at 60 Hz. A key
absent from `blend` means zero, not unknown.

**Android (ARCore Augmented Faces):**

| Field | Type | Meaning |
| --- | --- | --- |
| `regions` | object | `"nose"`, `"foreheadLeft"`, `"foreheadRight"` → float[16] pose |
| `transform` | float[16] | Face centre pose |
| `tracked` | bool | Tracking state |

No `blend` key, ever. ARCore does not compute blendshape coefficients, and
synthesising fake ones from the mesh would be worse than absence: a consumer
would have no way to tell measured values from invented ones.

## `ar.planes`

Emitted on change, not on a timer — one record per added/updated/removed plane.

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | string | Stable anchor identifier within the session |
| `event` | string | `"added"`, `"updated"`, `"removed"` |
| `center` | float[3] | Plane centre in world space |
| `extent` | float[2] | Size along the plane's local x and z, metres |
| `align` | string | `"horizontal"`, `"vertical"`, `"any"` |
| `pose` | float[16]? | Full plane transform |
| `classification` | string? | `"floor"`, `"wall"`, `"table"`, `"ceiling"`, `"none"`, … |

`removed` records carry only `id` and `event`. A plane's `extent` grows as more
of it is observed, so a consumer must treat updates as replacing prior state.

## `light`

| Field | Type | Meaning |
| --- | --- | --- |
| `lumens` | float | Ambient intensity. ARKit's scale, where ~1000 is "neutral" |
| `kelvin` | float? | Colour temperature |
| `mainDirection` | float[3]? | Dominant light direction, when the platform estimates it |

## `barometer`

| Field | Type | Meaning |
| --- | --- | --- |
| `kpa` | float | Absolute pressure, kilopascals |
| `relAltitude` | float? | Metres relative to the start of the session |

Relative, not absolute, altitude: absolute altitude needs a sea-level pressure
reference the phone does not have.

## `battery`

Low-rate (about 1 Hz), and genuinely useful — it is how you find out why the
frame rate collapsed nine minutes in.

| Field | Type | Meaning |
| --- | --- | --- |
| `level` | float | 0–1 |
| `charging` | bool | |
| `thermal` | string | `"nominal"`, `"fair"`, `"serious"`, `"critical"` |

`thermal` reaching `serious` is the standard explanation for a sender that was
fine and then throttled.

## `location`

**Off by default, separately opted into, and omitted from `channels` unless the
OS permission was granted.** See §6.5.

| Field | Type | Meaning |
| --- | --- | --- |
| `lat`, `lon` | float | Degrees |
| `accuracy` | float | Horizontal accuracy, metres |
| `altitude` | float? | Metres above sea level |
| `speed` | float? | m/s, `-1` when unknown |
| `heading` | float? | Degrees from true north |

## Worked example

A face-tracking session at 30 Hz, `curl -N 'http://phone:4747/data?ch=ar.face,attitude&hz=30'`:

```json
{"t":41230,"seq":1,"ch":"attitude","q":[0.01,-0.02,0.00,0.99],"euler":[1.2,-2.3,0.1],"ref":"magnetic"}
{"t":41231,"seq":2,"ch":"ar.face","tracked":true,"transform":[1,0,0,0,0,1,0,0,0,0,1,0,0.01,-0.04,-0.38,1],"blend":{"jawOpen":0.31,"mouthSmileLeft":0.12,"mouthSmileRight":0.14,"eyeBlinkLeft":0.05,"eyeBlinkRight":0.04},"look":[0.03,-0.02,-1.0]}
{"t":41264,"seq":3,"ch":"attitude","q":[0.01,-0.02,0.00,0.99],"euler":[1.3,-2.3,0.1],"ref":"magnetic"}
{"t":41265,"seq":4,"ch":"ar.face","tracked":true,"blend":{"jawOpen":0.44,"mouthSmileLeft":0.10,"mouthSmileRight":0.13}}
```

Note what the fourth record demonstrates: `transform` and `look` are absent
because they had not changed materially, `eyeBlink*` are absent because they
returned to zero, and both omissions are legal. A consumer holds its last known
value for absent *pose* fields and treats absent *blendshape* keys as zero — the
asymmetry is deliberate and is the one thing worth reading twice.
