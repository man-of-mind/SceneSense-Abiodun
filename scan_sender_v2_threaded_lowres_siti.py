#!/usr/bin/env python
"""
Threaded sender with low-resolution SI/TI computation.

Built on top of scan_sender_v2_threaded.py. The structural change is the
same -- camera callback stays lean, SI/TI runs on a worker thread, world.tick
never waits for it. The new change is in compute_si_ti(): we DOWNSAMPLE the
frame to a small resolution (default 320x180, configurable) before running
the Sobel + std-deviation math. That makes the SI/TI work roughly 16x cheaper,
which frees CPU cores that x265enc needs to keep up at 20 fps.

WHY THIS HELPS
--------------
On the previous run with full-resolution SI/TI, the worker was processing
~11 fps and the encoder was emitting ~10.5 fps -- both starved by CPU
contention. Camera still produced 20 fps, so 794 SI/TI frames got dropped
out of the queue, and x265enc fell behind, which slowly choked appsrc and
caused mild sluggishness. Cutting SI/TI compute cost ~16x should let both
the worker AND the encoder run at full 20 fps.

CAVEAT FROM ITU-T P.910 / SCAN-AI METHODOLOGY
---------------------------------------------
SI and TI are sensitive to resolution -- a downsampled image has less spatial
detail, so SI in particular will be lower in absolute terms than at 1280x720.
Time-series PATTERNS and CORRELATIONS with telemetry (which is what SCAN-AI's
analysis uses) should be preserved, but you cannot directly compare absolute
SI/TI values from this script with those produced by sender_v2.

If this matters for the research, confirm with the supervisor before
publishing any numbers from this script. The SI_TI_DOWNSAMPLE_WIDTH/HEIGHT
constants below make the resolution one knob -- set them to (1280, 720) to
exactly match sender_v2's math.

CSV outputs go to ./phase3_logs_lowres/ to keep them from clobbering the
full-resolution baseline data. plot_phase3.py works on either folder if you
pass the log-dir as an argument.

Usage:
    python3 scan_sender_v2_threaded_lowres_siti.py            # 2 Mbps target
    python3 scan_sender_v2_threaded_lowres_siti.py 4000       # 4 Mbps target
"""

import csv
import glob
import os
import queue
import random
import sys
import threading
import time

# --- Locate the CARLA Python egg ---------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHONAPI_DIR = os.path.dirname(os.path.dirname(THIS_DIR))
try:
    egg_pattern = os.path.join(
        PYTHONAPI_DIR, 'carla', 'dist',
        f'carla-*{sys.version_info.major}.{sys.version_info.minor}-'
        f'{"win-amd64" if os.name == "nt" else "linux-x86_64"}.egg',
    )
    sys.path.append(glob.glob(egg_pattern)[0])
except IndexError:
    pass

import carla
import numpy as np
import cv2

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst


# --- Configuration ----------------------------------------------------------
CARLA_HOST = 'localhost'
CARLA_PORT = 2000

IM_WIDTH = 1280
IM_HEIGHT = 720
FRAMERATE = 20
FIXED_DT = 1.0 / FRAMERATE

UDP_HOST = '127.0.0.1'
UDP_PORT = 65000

NUM_NPC_VEHICLES = 10
RANDOM_SEED = 0

BITRATE_KBPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2048

# --- THE NEW KNOBS ----------------------------------------------------------
# Compute SI/TI at this resolution instead of IM_WIDTH x IM_HEIGHT.
# Setting them equal to IM_WIDTH/IM_HEIGHT disables downsampling and you get
# byte-identical behavior to scan_sender_v2_threaded.py.
# 320x180 keeps the original 16:9 aspect ratio (1/4 in each axis = 1/16 area).
SI_TI_DOWNSAMPLE_WIDTH = 320
SI_TI_DOWNSAMPLE_HEIGHT = 180

SI_TI_QUEUE_MAX = 8
DROP_REPORT_EVERY = 30

LOG_DIR = os.path.join(THIS_DIR, 'phase3_logs_lowres')
os.makedirs(LOG_DIR, exist_ok=True)
FRAMES_CSV = os.path.join(LOG_DIR, 'frames.csv')
ENCODED_CSV = os.path.join(LOG_DIR, 'encoded.csv')


# --- GStreamer pipeline (identical to sender_v2) ----------------------------
GST_PIPELINE = (
    f"appsrc name=src ! videoconvert ! "
    f"x265enc name=enc tune=zerolatency key-int-max={FRAMERATE} "
    f"bitrate={BITRATE_KBPS} ! "
    f"rtph265pay config-interval=1 ! "
    f"udpsink host={UDP_HOST} port={UDP_PORT}"
)

Gst.init(None)
pipeline = Gst.parse_launch(GST_PIPELINE)
appsrc = pipeline.get_by_name('src')
encoder = pipeline.get_by_name('enc')
if appsrc is None or encoder is None:
    raise RuntimeError("Could not locate appsrc/encoder by name in pipeline.")

appsrc.set_property('caps', Gst.Caps.from_string(
    f"video/x-raw,format=BGRA,width={IM_WIDTH},height={IM_HEIGHT},"
    f"framerate={FRAMERATE}/1"
))


# --- Shared state -----------------------------------------------------------
state_lock = threading.Lock()
writer_lock = threading.Lock()

imu_state = {
    'speed_ms': 0.0,
    'ax_ms2': 0.0,
    'ay_ms2': 0.0,
    'yaw_rate_dps': 0.0,
}
frame_counter = {'n': 0}
encoded_counter = {'n': 0}
queue_drops = {'n': 0, 'last_reported': 0}

si_ti_queue: "queue.Queue[tuple | None]" = queue.Queue(maxsize=SI_TI_QUEUE_MAX)
worker_stop = threading.Event()


# --- CSV writers (same column layout as sender_v2) --------------------------
frames_file = open(FRAMES_CSV, 'w', newline='', buffering=1)
frames_writer = csv.writer(frames_file)
frames_writer.writerow([
    't_sim', 'frame_idx', 'si', 'ti',
    'speed_ms', 'ax_ms2', 'ay_ms2', 'yaw_rate_dps',
])

encoded_file = open(ENCODED_CSV, 'w', newline='', buffering=1)
encoded_writer = csv.writer(encoded_file)
encoded_writer.writerow(['t_wall', 'enc_idx', 'encoded_bytes'])


# --- SI / TI per ITU-T P.910, computed on a DOWNSAMPLED luma ---------------
def compute_si_ti(bgra_bytes, prev_luma_arr):
    """Same math as sender_v2's compute_si_ti, but on a small luma image.
    Returns (si, ti, current_luma_low_res)."""
    arr = np.frombuffer(bgra_bytes, dtype=np.uint8).reshape(
        IM_HEIGHT, IM_WIDTH, 4)
    bgr = arr[:, :, :3]
    luma_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Downsample BEFORE the Sobel/std math. INTER_AREA is the recommended
    # interpolation for downscaling (gives clean, low-noise results).
    if (SI_TI_DOWNSAMPLE_WIDTH != IM_WIDTH
            or SI_TI_DOWNSAMPLE_HEIGHT != IM_HEIGHT):
        luma = cv2.resize(
            luma_full,
            (SI_TI_DOWNSAMPLE_WIDTH, SI_TI_DOWNSAMPLE_HEIGHT),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)
    else:
        luma = luma_full.astype(np.float32)

    sx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sx * sx + sy * sy)
    si = float(np.std(grad_mag))

    if prev_luma_arr is None:
        ti = 0.0
    else:
        ti = float(np.std(luma - prev_luma_arr))

    return si, ti, luma


# --- SI/TI worker thread ----------------------------------------------------
def si_ti_worker():
    prev_luma = None
    while True:
        try:
            item = si_ti_queue.get(timeout=0.5)
        except queue.Empty:
            if worker_stop.is_set():
                return
            continue
        if item is None:
            return
        t_sim, frame_idx, raw_bgra, imu = item
        try:
            si, ti, luma = compute_si_ti(raw_bgra, prev_luma)
            prev_luma = luma
            with writer_lock:
                frames_writer.writerow([
                    f"{t_sim:.6f}",
                    frame_idx,
                    f"{si:.3f}", f"{ti:.3f}",
                    f"{imu['speed_ms']:.3f}",
                    f"{imu['ax_ms2']:.3f}",
                    f"{imu['ay_ms2']:.3f}",
                    f"{imu['yaw_rate_dps']:.3f}",
                ])
        except Exception as e:
            print(f"[worker] SI/TI compute error: {e}")


# --- Camera callback --------------------------------------------------------
def on_camera_image(image):
    try:
        buf = Gst.Buffer.new_wrapped(image.raw_data)
        appsrc.emit('push-buffer', buf)

        with state_lock:
            imu = dict(imu_state)

        raw_copy = bytes(image.raw_data)
        item = (image.timestamp, frame_counter['n'], raw_copy, imu)
        try:
            si_ti_queue.put_nowait(item)
        except queue.Full:
            try:
                si_ti_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                si_ti_queue.put_nowait(item)
            except queue.Full:
                pass
            queue_drops['n'] += 1
            if queue_drops['n'] - queue_drops['last_reported'] >= DROP_REPORT_EVERY:
                print(f"[sender] WARN: SI/TI queue full, "
                      f"dropped {queue_drops['n']} frames so far")
                queue_drops['last_reported'] = queue_drops['n']

        frame_counter['n'] += 1
    except Exception as e:
        print(f"[sender] camera callback error: {e}")


# --- IMU callback -----------------------------------------------------------
def on_imu(imu_data, ego_ref):
    accel = imu_data.accelerometer
    gyro = imu_data.gyroscope
    try:
        v = ego_ref[0].get_velocity()
        speed = (v.x * v.x + v.y * v.y + v.z * v.z) ** 0.5
    except Exception:
        speed = 0.0
    with state_lock:
        imu_state['speed_ms'] = speed
        imu_state['ax_ms2'] = accel.x
        imu_state['ay_ms2'] = accel.y
        imu_state['yaw_rate_dps'] = gyro.z * (180.0 / np.pi)


# --- Encoder pad probe ------------------------------------------------------
def on_encoded_buffer(_pad, info):
    buf = info.get_buffer()
    if buf is None:
        return Gst.PadProbeReturn.OK
    size = buf.get_size()
    with writer_lock:
        encoded_writer.writerow([
            f"{time.time():.6f}",
            encoded_counter['n'],
            size,
        ])
    encoded_counter['n'] += 1
    return Gst.PadProbeReturn.OK


# --- Main -------------------------------------------------------------------
def main():
    client = None
    world = None
    original_settings = None
    spawned_vehicles = []
    camera = None
    imu_sensor = None
    ego_ref = [None]

    worker_thread = threading.Thread(
        target=si_ti_worker, name='si_ti_worker', daemon=True)
    worker_thread.start()
    print(f"[sender] SI/TI worker thread started "
          f"(queue max={SI_TI_QUEUE_MAX}, drop-oldest policy).")
    print(f"[sender] SI/TI compute resolution: "
          f"{SI_TI_DOWNSAMPLE_WIDTH}x{SI_TI_DOWNSAMPLE_HEIGHT} "
          f"(from camera {IM_WIDTH}x{IM_HEIGHT}).")

    try:
        client = carla.Client(CARLA_HOST, CARLA_PORT)
        client.set_timeout(10.0)
        world = client.get_world()

        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DT
        world.apply_settings(settings)

        tm = client.get_trafficmanager()
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points in this map.")

        bp_lib = world.get_blueprint_library()
        models = ['dodge', 'audi', 'model3', 'mini', 'mustang',
                  'lincoln', 'prius', 'nissan', 'crown', 'impala']
        vehicle_bps = [bp for bp in bp_lib.filter('*vehicle*')
                       if any(m in bp.id for m in models)]
        if not vehicle_bps:
            vehicle_bps = list(bp_lib.filter('*vehicle*'))

        n = min(NUM_NPC_VEHICLES, len(spawn_points))
        for sp in random.sample(spawn_points, n):
            actor = world.try_spawn_actor(random.choice(vehicle_bps), sp)
            if actor is not None:
                spawned_vehicles.append(actor)
        if not spawned_vehicles:
            raise RuntimeError("No vehicles spawned.")

        ego_vehicle = spawned_vehicles[0]
        ego_ref[0] = ego_vehicle
        for v in spawned_vehicles:
            v.set_autopilot(True)
        print(f"[sender] Spawned {len(spawned_vehicles)} vehicles.")
        print(f"[sender] Ego: {ego_vehicle.type_id} (id={ego_vehicle.id})")

        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(IM_WIDTH))
        camera_bp.set_attribute('image_size_y', str(IM_HEIGHT))
        camera_bp.set_attribute('fov', '120')
        camera = world.spawn_actor(
            camera_bp,
            carla.Transform(carla.Location(x=0.95, z=1.3)),
            attach_to=ego_vehicle,
        )
        camera.listen(on_camera_image)

        imu_bp = bp_lib.find('sensor.other.imu')
        imu_sensor = world.spawn_actor(
            imu_bp, carla.Transform(), attach_to=ego_vehicle)
        imu_sensor.listen(lambda data: on_imu(data, ego_ref))

        enc_src_pad = encoder.get_static_pad('src')
        enc_src_pad.add_probe(Gst.PadProbeType.BUFFER, on_encoded_buffer)

        pipeline.set_state(Gst.State.PLAYING)
        print(f"[sender] Streaming H.265/RTP -> {UDP_HOST}:{UDP_PORT}")
        print(f"[sender] Encoder target: {BITRATE_KBPS} kbps "
              f"({BITRATE_KBPS/1000:.1f} Mbps) | "
              f"{IM_WIDTH}x{IM_HEIGHT} @ {FRAMERATE} fps")
        print(f"[sender] Logging frames  -> {FRAMES_CSV}")
        print(f"[sender] Logging encoded -> {ENCODED_CSV}")
        print("[sender] Press Ctrl+C to stop.")

        while True:
            world.tick()

    except KeyboardInterrupt:
        print("\n[sender] Ctrl+C received, shutting down.")

    finally:
        print("[sender] Cleaning up...")
        if camera is not None and camera.is_alive:
            try:
                camera.stop()
                camera.destroy()
            except Exception as e:
                print(f"[sender] camera cleanup: {e}")
        if imu_sensor is not None and imu_sensor.is_alive:
            try:
                imu_sensor.stop()
                imu_sensor.destroy()
            except Exception as e:
                print(f"[sender] imu cleanup: {e}")
        if client is not None and spawned_vehicles:
            try:
                client.apply_batch_sync(
                    [carla.command.DestroyActor(v.id)
                     for v in spawned_vehicles if v.is_alive], True)
                print(f"[sender] Destroyed {len(spawned_vehicles)} vehicles.")
            except Exception as e:
                print(f"[sender] vehicles cleanup: {e}")
        if world is not None and original_settings is not None:
            try:
                world.apply_settings(original_settings)
                print("[sender] Restored async world settings.")
            except Exception as e:
                print(f"[sender] settings restore: {e}")
        try:
            pipeline.send_event(Gst.Event.new_eos())
            pipeline.set_state(Gst.State.NULL)
            print("[sender] GStreamer pipeline stopped.")
        except Exception as e:
            print(f"[sender] pipeline cleanup: {e}")

        worker_stop.set()
        try:
            si_ti_queue.put_nowait(None)
        except queue.Full:
            try:
                si_ti_queue.get_nowait()
                si_ti_queue.put_nowait(None)
            except (queue.Empty, queue.Full):
                pass
        worker_thread.join(timeout=2.0)
        if worker_thread.is_alive():
            print("[sender] WARN: SI/TI worker did not exit cleanly.")

        try:
            frames_file.close()
            encoded_file.close()
        except Exception:
            pass
        print(f"[sender] Wrote {frame_counter['n']} rows to frames.csv")
        print(f"[sender] Wrote {encoded_counter['n']} rows to encoded.csv")
        if queue_drops['n']:
            print(f"[sender] SI/TI queue dropped {queue_drops['n']} frames total.")
        print("[sender] Done.")


if __name__ == '__main__':
    main()
