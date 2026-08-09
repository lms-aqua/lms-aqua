# Recording a dataset for a model

How `lostcam capture` writes data, and how to get data that is actually worth
training on. Written around the case it was built for — a phone on a tripod
watching a 3D printer's build plate for hours — but nothing here is specific to
printers.

```bash
lostcam capture 192.168.1.42 \
  --out runs/2026-08-09-benchy \
  --roi 420,300,880,660 --plate-mm 220 \
  --calibrate-plate \
  --every 15 \
  --notes "ender 3 v2, PLA, 0.2mm, 60mm/s"
```

## Read this part first: what the hardware can and cannot measure

**LiDAR depth resolves centimetres, not layer heights.** An iPhone's depth sensor
is built for room-scale AR. Its raster is roughly 256x192, its range starts around
0.25 m, and its per-pixel error is on the order of a centimetre at close range.
A 0.2 mm layer is two orders of magnitude below that.

So depth is genuinely useful for:

- is there anything on the plate at all (started / finished / cleared)
- gross shape and how tall the object is, to within a centimetre or so
- catastrophic failure — spaghetti, a detached part, a blob swallowing the nozzle
- the object having moved, which is what detachment looks like in depth

and it is useless for:

- layer height, layer adhesion, surface finish
- stringing, small under-extrusion, minor warping
- anything requiring sub-millimetre measurement

For the fine defects, the **colour frames** are the signal, and depth is a
supporting channel. Build your labels accordingly. A model told to predict layer
quality from this depth stream will learn noise.

## Getting data worth training on

Five things, in rough order of how badly each one bites.

### 1. Lock exposure, white balance and focus

In the sender app: **Focus on centre and lock**, once the shot is framed. Then
`/info` reports `"capture":{"locks":"locked"}` and `lostcam capture` stops warning
you.

This is the big one. Left on auto over a four-hour print:

- Auto-exposure rebalances as the object grows and fills more of the frame, so
  overall brightness drifts and the *same* physical scene produces different
  pixels at hour 1 and hour 4.
- Auto-white-balance shifts colour as the ambient light changes through the day.
- Auto-focus hunts, and produces occasional soft frames that are hard to spot by
  eye but very easy for a model to latch onto.

A model trained on that data learns the camera's reaction, not the print. Worse,
it will appear to work in validation — because the drift correlates with time, and
time correlates with print progress — and then fail on a print that starts at a
different time of day. The `metrics.mean` and `metrics.sharpness` columns exist
partly so you can *check* this after the fact: on a locked camera watching a slow
scene they should be nearly flat.

### 2. Set an ROI

`--roi X,Y,W,H` in pixels, around the build plate. Without it every metric is
computed over the whole frame, which is mostly room: someone walking past moves
`diff_mean` more than a failing print does.

Find the numbers by grabbing one frame and looking at it:

```bash
lostcam capture <ip> --out /tmp/oneframe --frames 1 --warmup 0
# open /tmp/oneframe/frames/000001.jpg and read the plate's pixel rectangle
```

### 3. Calibrate the scale

`--plate-mm 220` with `--roi` records millimetres per pixel, so pixel areas
become mm² and heights come out in millimetres.

The honest caveat, also recorded in `dataset.json`: this is **one scale for one
plane**. The camera is not orthographic, so something 100 mm above the plate
covers more pixels per millimetre than the plate does. For a roughly top-down or
shallow-angle view of a flat plate this is a good approximation. For a steep
oblique view it is not, and you should treat the scale as indicative.

`--calibrate-plate` additionally measures the *empty* plate's distance, which is
what turns depth readings into heights above the plate. Run it with the plate
clear. If there is not enough valid depth, it says so and omits heights rather
than inventing a reference — a wrong reference would offset every height in the
dataset by the same unknown amount, which is the kind of error you find six weeks
later.

### 4. Subsample

`--every 15` keeps one frame in fifteen. At 30 fps a four-hour print is 432,000
frames, which is both far more than you need and more disk than you want. Prints
change slowly; one frame every half second is generous, and one every few seconds
is often plenty.

Skipped frames are counted, not silently dropped — the run's summary tells you.

### 5. Tag the interesting moments

`events.jsonl` carries operator tags anchored to a frame number, so "the failure
started here" survives into training as a label boundary rather than living in
someone's memory.

## Layout

```
runs/2026-08-09-benchy/
├── dataset.json      how the recording was made, and its summary
├── manifest.jsonl    one JSON record per frame
├── events.jsonl      operator tags
├── frames/
│   ├── 000001.jpg
│   └── ...
└── depth/
    ├── 000001.u16
    └── ...
```

Deliberately boring and self-describing. No custom container, no separate index
that can drift out of sync with the files, nothing that needs this library to read
back. The JPEG bytes are exactly what the phone transmitted — re-encoding would
add a second generation of compression artefacts to the thing being measured.

### `manifest.jsonl`

One record per frame:

```json
{
  "frame": 137,
  "file": "frames/000137.jpg",
  "t": 3449864,
  "t_rel": 1217,
  "bytes": 41203,
  "label": "printing",
  "metrics": {
    "mean": 115.38, "std": 24.31, "sharpness": 182.4,
    "clipped_black": 0.0, "clipped_white": 0.002,
    "diff_mean": 0.97, "diff_fraction": 0.0,
    "depth_coverage": 0.93, "depth_min_mm": 359.0,
    "depth_max_mm": 400.0, "depth_median_mm": 400.0,
    "height_max_mm": 41.0, "height_mean_mm": 12.4
  },
  "depth": {
    "file": "depth/000137.u16", "width": 256, "height": 192,
    "format": "u16mm", "t": 3449407, "skew_ms": 457,
    "intrinsics": [180.2, 180.2, 128.0, 96.0]
  },
  "samples": {
    "attitude": {"t": 3449674, "seq": 211, "q": [0,0.96,0,-0.26], "age_ms": 190},
    "battery": {"t": 3449100, "seq": 209, "level": 0.82, "thermal": "fair", "age_ms": 764}
  }
}
```

`t` is the **sender's monotonic clock** in milliseconds (PROTOCOL.md §6.3). That
is the whole reason it is specified that way: video frames, depth frames and sensor
samples all carry it, so they can be aligned after the fact without either side
agreeing on wall time or a clock jump ruining a run.

### The two fields to filter on

**`samples.*.age_ms`** — how old that sensor value was when the frame arrived.
Values are nearest-previous, never interpolated, because interpolating a tracking
state or a plane event would invent data. Anything over ~200 ms is stale for a
moving scene; for a printer it rarely matters.

**`depth.skew_ms`** — the gap between the frame's timestamp and its depth frame's.
Depth arrives far slower than video (roughly 10 Hz against 30), so one depth raster
gets attached to several video frames and the skew grows across them. On a slow
scene that is fine. If you care, drop rows above a threshold:

```python
pairs = [r for r in records if r.get("depth", {}).get("skew_ms", 1e9) < 120]
```

Frames with no recent depth simply have no `depth` key. Never assume every frame
has one.

### `depth/NNNNNN.u16`

Row-major **unsigned 16-bit little-endian millimetres**. `0` means **no
measurement** — out of range, absorbed, or below the confidence threshold — and
*not* zero distance. Treating 0 as a distance puts a wall at the lens and will
wreck any statistic you compute.

Little-endian regardless of the machine that wrote it, so a dataset copied between
machines still reads.

## Reading it back

No LostCam import required:

```python
import json, pathlib
import numpy as np
from PIL import Image

root = pathlib.Path("runs/2026-08-09-benchy")
records = [json.loads(line) for line in
           (root / "manifest.jsonl").read_text().splitlines() if line.strip()]

for record in records:
    image = np.asarray(Image.open(root / record["file"]).convert("RGB"))

    depth = None
    if "depth" in record:
        meta = record["depth"]
        raw = np.fromfile(root / meta["file"], dtype="<u2")
        depth = raw.reshape(meta["height"], meta["width"])
        depth = np.where(depth > 0, depth, 0)  # 0 stays "no measurement"

    features = record.get("metrics", {})
    label = record.get("label")
```

Or with the reference reader, which also tolerates a truncated final line from an
interrupted run:

```python
from lostcam.dataset import read_manifest, load_depth
records = read_manifest("runs/2026-08-09-benchy")
depth = load_depth("runs/2026-08-09-benchy", records[100])
```

## Using the metrics

The per-frame numbers serve double duty.

**As features.** `height_max_mm` tracks print progress. `diff_fraction` spikes
when something moves that should not have. `depth_coverage` collapsing often means
the object fell over or a limb of spaghetti is scattering the beam.

**As data-quality checks**, which is the use people skip and regret. Before
training, plot them across the run:

| Symptom | Almost always means |
| --- | --- |
| `sharpness` drops for scattered frames | Auto-focus is on and hunting |
| `mean` drifts monotonically | Auto-exposure is on, or the room light is changing |
| `mean` steps abruptly | Someone turned a light on, or exposure re-locked |
| `clipped_white` rising | The shot is losing highlight detail — expose lower |
| `diff_fraction` spikes on a bolted-down camera | The rig was knocked; poses before and after are not comparable |
| `depth_coverage` low from the start | Camera too close, too oblique, or the surface is too dark/shiny for LiDAR |

A run that fails these is not a run to train on. Finding that out from a plot
costs an afternoon; finding it out from a model that mysteriously does not
generalise costs considerably more.

## Practical rig notes

- **Bolt the phone down.** Every pose and every ROI assumes the camera has not
  moved. A tripod that gets nudged invalidates the ROI for every frame after it.
- **Keep the light constant.** Constant artificial light beats daylight, which
  changes colour and intensity all day and correlates with print progress in a way
  that will fool your validation split.
- **Watch the thermals.** Streaming video plus an AR session is a heavy load. The
  `battery` channel is how you discover that `thermal` hit `serious` and the frame
  rate halved. Keep the phone on a charger and out of direct sun.
- **Use USB for long runs.** `adb forward` (Android) removes Wi-Fi from the
  equation entirely and charges the phone at the same time. Wi-Fi drops are
  survivable — the capture reconnects — but each drop is a gap in the data.
- **Record failures deliberately.** A dataset of successful prints teaches a model
  what success looks like and nothing about failure. Deliberately induced failures,
  tagged in `events.jsonl`, are worth more per frame than hours of a clean print.
