# Build-plate mapping

Scan the plate once, then measure whatever the printer puts on it: how many
objects, where each one is, how tall, how wide, how much material.

```bash
# 1. Setup — with the plate EMPTY
lostcam scan 192.168.1.42 --plate-mm 220 --out plate.json

# 2. Live check
lostcam plate 192.168.1.42 --plate plate.json

# 3. Record with measurements in every frame
lostcam capture 192.168.1.42 --plate plate.json --out runs/benchy
```

## The dashboard

Add `--web` to either `plate` or `capture` and the same measurements are served
as a page you can leave open on a second screen next to the printer:

```bash
lostcam plate 192.168.1.42 --plate plate.json --web        # http://127.0.0.1:8770/
lostcam capture 192.168.1.42 --plate plate.json --out runs/benchy --web 9000
```

It shows the height map as a top-down heat map with millimetre rulers, every
tracked object with its measurements, and a growth trace of the tallest point —
plus a banner whenever the nozzle is in shot or the filter has not settled yet,
so a number you are watching is never quietly untrustworthy.

Three things worth knowing:

- It binds to **127.0.0.1 only**. Nothing about the plate leaves the machine, and
  there is no authentication because there is nothing to authenticate to. If you
  want it on another screen, tunnel it (`ssh -L 8770:127.0.0.1:8770`).
- Updates are pushed at the **depth rate**, not the video rate — one per new depth
  frame, so the page shows measurements rather than a video feed.
- `/state.json` is the same payload as one poll, which makes
  `curl -s localhost:8770/state.json | jq .tallest_mm` a perfectly good way to
  wire the plate into a script. `/healthz` returns `ok`.

The dashboard never affects the recording: if the page throws, the port is busy,
or every tab is closed, `capture` carries on and says so once.

## What it actually measures

Per object, per frame, all in millimetres:

| Field | Meaning |
| --- | --- |
| `centre_mm` | Position `[x, y]` relative to the plate centre |
| `bbox_mm` | Extent along the plate's two axes |
| `height_max_mm` | Tallest point above the plate plane |
| `height_mean_mm` | Mean height over the footprint |
| `footprint_mm2` | Area actually occupied — measured cells, not the bounding box |
| `volume_mm3` | Sum of height × cell area over the footprint |
| `solidity` | Occupied fraction of the bounding box, 0–1 |
| `track` | Stable identity across frames |
| `age_frames` | How many observations this track has |

Plus whole-plate totals: `object_count`, `occupied_mm2`, `tallest_mm`,
`total_volume_mm3`, `map_coverage`.

`solidity` is the field worth knowing about. A cube reads ~0.97 and a Benchy
~0.85; a sprawl of spaghetti fills very little of its bounding box and reads
~0.3. A print whose solidity collapses while its footprint grows is a print that
has come loose and is being dragged around the bed.

## How it works

**The scan fits a plane, not a distance.** A phone on a tripod looks at the bed
at an angle, so the bed's *depth* genuinely varies across the frame. Subtracting
a single distance would report the far half of an empty plate as below the plate
and the near half as above it. The scan fits a plane by least squares, discards
outliers, and repeats — so the printer frame and the bed clips do not drag it —
then measures every later height along that plane's normal.

**Every frame becomes a top-down grid.** Depth pixels are unprojected to 3D
using the depth camera's own intrinsics, then resampled into an orthographic grid
in plate coordinates. That removes perspective: every cell is the same physical
size, so areas and volumes are sums rather than approximations, and the resulting
height map is a fixed-scale 2.5D image — a much better model input than a
perspective view whose scale depends on where in the frame something sits.

**Objects are connected components** of the cells standing more than
`--threshold-mm` above the plate. No model, no training: on a bed that starts
empty, "taller than the plate" is exactly the right definition of an object.

## The cell size matters more than you'd think

This is the one setting that can silently produce nothing at all, so the default
derives it from the sensor rather than picking a round number.

An iPhone's depth raster at 400 mm samples roughly **every 2.2 mm**. Ask for a
1 mm grid and every measured cell is surrounded by empty ones: the occupancy mask
becomes a checkerboard, connected-component labelling finds one component per
cell, every component falls below the minimum footprint, and the plate reports as
empty while a 60 mm cube sits on it.

`lostcam scan` therefore computes the sample pitch and picks a cell about 1.5×
coarser — typically 3–4 mm on an iPhone at working distance. Pass `--cell-mm`
only if you have a reason, and the scan will warn you if your value is finer than
the sensor can fill and tell you what to use instead.

As defence in depth, segmentation morphologically closes one-cell gaps before
labelling. Measurements still read only genuinely measured cells, so closing can
rejoin a shattered object but never inflates its footprint, height or volume.

## Accuracy, stated honestly

**LiDAR resolves centimetres, not layer heights.** The sensor is built for
room-scale AR: ~256×192, range from about 0.25 m, error on the order of a
centimetre at close range. A 0.2 mm layer is two orders of magnitude below that.

Reliable:

- Is anything on the plate — started, finished, cleared
- Overall shape and how tall it is, to within roughly a centimetre
- Catastrophic failure — spaghetti, detachment, a blob swallowing the nozzle
- The object having moved, which is what detachment looks like in depth
- Progress over time, because *relative* height change is far more accurate than
  absolute height

Not reliable:

- Layer height, layer adhesion, surface finish
- Stringing, mild under-extrusion, small warping
- Anything needing sub-millimetre measurement
- Overhangs and undercuts — the volume is of the *visible* shape, so material the
  sensor cannot see is not in it

For fine defects the colour frames are the signal, and the plate measurements are
the supporting channel that tells you *where* on the bed to look.

## Reading the height maps

`capture --plate` writes `height_maps/NNNNNN.u16` alongside the frames: the
orthographic grid as u16 little-endian, where **0 means no measurement** and any
other value is millimetres above the plate **plus one**. The offset exists so a
genuine 0 mm height stays distinguishable from an unmeasured cell.

```python
from lostcam.dataset import read_manifest, load_height_map

records = read_manifest("runs/benchy")
for record in records:
    grid = load_height_map("runs/benchy", record)   # float32 mm, NaN = unmeasured
    plate = record.get("plate")
    if plate:
        tallest = plate["tallest_mm"]
        objects = plate["objects"]
```

`load_height_map` returns NaN for unmeasured cells rather than 0, because an
occluded cell is not a flat one and averaging zeros into a height understates
every object.

Cells reading well *below* the plate are exported as unmeasured too. A cell
250 mm below the bed is a dropout or a bad fit, not a flat plate, and clamping it
to zero would turn "cannot measure" into a confident measurement of a surface
that is not there.

## Rig notes

- **Scan with the plate genuinely empty.** Anything left on it becomes part of the
  plate, and every later height is wrong by that object's thickness.
- **Bolt the phone down.** The calibration is a fixed relationship between camera
  and bed. Nudging the tripod invalidates it, and the fastest way to notice is a
  `diff_fraction` spike in the frame metrics.
- **Rescan after moving anything.** It takes ten seconds.
- **Aim for a shallow angle.** Steeper than about 60° off head-on and the far side
  of the bed is poorly sampled and gets occluded by whatever prints in front of
  it. The scan reports the tilt and warns past 60°.
- **Watch `map_coverage`.** Low from the start means the camera is too close, too
  oblique, or the bed is too dark or shiny for the beam — glass and mirror-finish
  PEI are the usual culprits. Coverage dropping mid-print is normal: the object
  occludes the bed behind it.
- **Check the plate-coordinate origin.** It is the centre of the visible flat
  surface. If the bed sits on a larger flat table that is also in view, the origin
  drifts toward the middle of *everything* flat — heights stay correct, but
  positions are offset. The scan warns when the visible plane is much larger than
  the configured plate; crop the view to the bed for positions you can trust.

## The nozzle and gantry

The hotend and gantry pass through the camera's view constantly, and
geometrically they are indistinguishable from a print: a typical hotend assembly
stands 40–60 mm above the bed. Unfiltered, it *is* the tallest object in almost
every frame, and worse, whatever it stands in front of is occluded and vanishes
from the measurements.

Measured against the mock printer, with a 55 mm hotend sweeping over a print
growing slowly from 5 mm:

| | Tallest reported | Objects | Machinery reported |
| --- | --- | --- | --- |
| `--no-machinery-filter` | **54.4 mm** — the nozzle | 1–2 | never |
| default | **7.4 mm** — the print | always 1 | every frame |

Three mechanisms do the work, and they are on by default:

**A temporal median.** The nozzle occupies any given cell for a fraction of a
second as it traverses; the print occupies it for the rest of the run. A per-cell
median over `--median-frames` (default 7) outvotes the nozzle entirely. The
window is short deliberately: a print grows millimetres per *minute* while the
nozzle moves millimetres per *frame*, so about a second of frames separates them
with room to spare and without lagging real growth.

**A motion mask.** Cells whose height varies by more than `--motion-mm`
(default 6) across the window are machinery, not print. This catches what the
median cannot — a gantry that lingers, and the boundary cells where a fast part
half-covers a cell. Those cells are excluded and reported as `moving_mm2`, which
doubles as "is the machine in shot right now".

**A held last-good map.** While a cell is occluded, its last stable value is held
rather than treated as absent, so an object's footprint and volume do not dip
every time the gantry sweeps past. Held cells are counted in `held_cells`, so the
substitution is never invisible.

For a gantry rail parked above the bed, add `--max-height-mm` a little above your
tallest plausible print — anything above the ceiling is rejected outright and
counted in `ceiling_mm2`.

Two fields exist because of all this and are worth filtering on:

- **`settled: false`** on a frame means the filter has not seen enough depth
  frames yet. Those are the least trustworthy measurements in the run.
- **`confirmed: false`** on an object means it has been seen for fewer than
  `--min-age-frames` (default 3) observations. A single frame of machinery that
  slipped through looks exactly like a new object; persistence is what separates
  them. For training, prefer `confirmed` objects on `settled` frames.
