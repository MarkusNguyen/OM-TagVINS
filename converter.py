#!/usr/bin/env python3
# Usage:
#   python video_processor.py sample.insv
#
# Output:
#   data/base/<input>.rrd
import math
import os
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
import rerun as rr

MAGIC = b"8db42d694ccc418790edff439fe026bf"
JPEG = cv2.IMWRITE_JPEG_QUALITY
QUALITY = 90
OUTPUT_DIR = Path(__file__).resolve().parent / "logs" / "base"


def varint(buf, pos):
    value = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7


def protobuf(buf):
    fields = {}
    pos = 0
    while pos < len(buf):
        key, pos = varint(buf, pos)
        number, wire = key >> 3, key & 7
        if not number:
            break
        if wire == 0:
            value, pos = varint(buf, pos)
        elif wire == 1:
            value = struct.unpack_from("<d", buf, pos)[0]
            pos += 8
        elif wire == 2:
            length, pos = varint(buf, pos)
            value = buf[pos : pos + length]
            pos += length
        elif wire == 5:
            value = struct.unpack_from("<f", buf, pos)[0]
            pos += 4
        else:
            break
        fields[number] = value
    return fields


def records(path):
    size = path.stat().st_size
    with path.open("rb") as handle:
        handle.seek(-32, 2)
        if handle.read(32) != MAGIC:
            raise ValueError(f"{path} has no Insta360 trailer")
        handle.seek(-40, 2)
        start = size - struct.unpack("<I", handle.read(4))[0]
        pos = size - 78
        found = {}
        while pos > start:
            handle.seek(pos)
            _, ident, length = struct.unpack("<BBL", handle.read(6))
            offset = pos - length
            if offset < start:
                break
            found[ident] = (offset, length)
            pos = offset - 6
    return found


def read(path, offset, length):
    with path.open("rb") as handle:
        handle.seek(offset)
        return handle.read(length)


def load_rolling_shutter_readout_s(source: Path) -> float:
    """Read Insta360 frame_readout_time (ms) and return seconds.

    Mirrors specialist-ai/specialist: the X3 metadata stores the full-frame
    rolling-shutter readout in the first telemetry record under
    Lens.Data.frame_readout_time, in milliseconds.
    """
    try:
        import telemetry_parser
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "telemetry-parser is required to read Insta360 rolling-shutter "
            "metadata; install config/requirements.txt"
        ) from exc

    parser = telemetry_parser.Parser(str(source))
    telemetry = parser.telemetry()
    if not isinstance(telemetry, (list, tuple)) or not telemetry:
        raise RuntimeError(f"{source} has no telemetry metadata")

    meta = telemetry[0]
    lens = meta.get("Lens", {}).get("Data", {}) if isinstance(meta, dict) else {}
    raw = lens.get("frame_readout_time")
    if raw is None:
        raise RuntimeError(f"{source} has no Lens.Data.frame_readout_time")

    readout_s = float(raw) * 1e-3
    if not np.isfinite(readout_s) or readout_s < 0.0:
        raise RuntimeError(f"Invalid rolling-shutter readout in {source}: {raw}")
    return readout_s


def log_rolling_shutter_readout_s(readout_s: float) -> None:
    """Store the readout on the base Rerun recording's time timeline."""
    rr.send_columns(
        "calibration/rolling_shutter_readout_s",
        indexes=[rr.TimeColumn("time", duration=[0.0])],
        columns=rr.Scalars.columns(scalars=[float(readout_s)]),
    )


def imu_samples(data, info):
    raw = info.get(62)
    if raw is None:
        raw = len(data) % 20 == 0 and len(data) % 56 != 0
    samples = []
    if raw:
        config = protobuf(info[65]) if isinstance(info.get(65), bytes) else {}
        accel = float(config.get(1, 32)) / 32768
        gyro = float(config.get(2, 2000)) * math.pi / (32768 * 180)
        for pos in range(0, len(data), 20):
            t, ax, ay, az, gx, gy, gz = struct.unpack_from("<Q6H", data, pos)
            samples.append(
                (t, (ax - 32768) * accel, (ay - 32768) * accel,
                 (az - 32768) * accel, (gx - 32768) * gyro,
                 (gy - 32768) * gyro, (gz - 32768) * gyro)
            )
    else:
        for pos in range(0, len(data), 56):
            t, ax, ay, az, gx, gy, gz = struct.unpack_from("<Q3d3d", data, pos)
            samples.append((t * 1000, ax, ay, az, gx, gy, gz))
    return sorted(samples)


def load_imu(source):
    found = records(source)
    if 1 not in found or 3 not in found:
        raise RuntimeError(f"{source} has no IMU data")
    info = protobuf(read(source, *found[1]))
    samples = imu_samples(read(source, *found[3]), info)
    if not samples:
        raise RuntimeError(f"{source} has no IMU samples")
    return samples, info.get(24, samples[0][0])


def log_image(entity, image):
    ok, encoded = cv2.imencode(".jpg", image, [JPEG, QUALITY])
    if not ok:
        raise RuntimeError(f"Could not encode {entity}")
    rr.log(
        entity,
        rr.EncodedImage(contents=encoded.tobytes(), media_type=rr.MediaType.JPEG),
    )


def log_imu(samples, origin):
    times = [(t - origin) / 1e6 for t, *_ in samples]
    values = list(zip(*(sample[1:] for sample in samples)))
    for group, start in (("accel", 0), ("gyro", 3)):
        for axis, series in zip("xyz", values[start : start + 3]):
            rr.send_columns(
                f"imu/{group}/{axis}",
                indexes=[rr.TimeColumn("time", duration=times)],
                columns=rr.Scalars.columns(scalars=series),
            )


def log_frames(cap, fps, reference, origin):
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        if width != 2 * height:
            raise ValueError(
                f"Frame is {width}x{height}; expected a 2:1 side-by-side layout"
            )
        rear, front = frame[:, : width // 2], frame[:, width // 2 :]
        rr.set_time("time", duration=(reference - origin) / 1e6 + index / fps)
        rr.set_time("frame", sequence=index)
        log_image("camera/front", front)
        log_image("camera/rear", rear)
        index += 1
    return index


def export(source):
    source = Path(source).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"INSV file not found: {source}")

    samples, reference = load_imu(source)
    readout_s = load_rolling_shutter_readout_s(source)
    origin = min(samples[0][0], reference)
    target = OUTPUT_DIR / f"{source.stem}.rrd"
    target.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {source}")
    try:
        rr.init(source.stem)
        rr.save(target)
        log_rolling_shutter_readout_s(readout_s)
        log_imu(samples, origin)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = log_frames(cap, fps, reference, origin)
    finally:
        cap.release()
        rr.disconnect()

    if not frames:
        raise RuntimeError(f"{source} has no frames")
    print(f"Rolling-shutter readout: {readout_s * 1e3:.3f} ms")
    return target


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "INSV_INPUT", "sample.insv"
    )
    try:
        target = export(source)
    except (FileNotFoundError, ValueError, RuntimeError, OSError, cv2.error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
