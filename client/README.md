# LostCam desktop client

The desktop half of [LostCam](../README.md). Receives the phone's camera and
pushes it into a virtual camera the OS already provides.

```bash
pip install -e ".[all]"     # from this directory
lostcam doctor              # check the virtual camera / audio / adb setup
```

| Command | What it does |
| --- | --- |
| `lostcam pull <phone-ip>` | Connect to a phone serving MJPEG (the iOS app, or DroidCam) |
| `lostcam pull <ip> --usb` | Same, tunneled over USB with `adb` |
| `lostcam serve` | Serve the browser sender page; the phone pushes frames |
| `lostcam mocksender` | Pretend to be a phone, to test this machine alone |
| `lostcam discover` | Find senders on the LAN |
| `lostcam devices` | List Android devices visible to `adb` |
| `lostcam doctor` | Diagnose the local setup |

Full documentation, including the virtual camera prerequisites per OS, is in the
[top-level README](../README.md). The wire format is in
[docs/PROTOCOL.md](../docs/PROTOCOL.md).
