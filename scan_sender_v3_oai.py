#!/usr/bin/env python
"""
OAI variant of the SCAN-AI sender. GStreamer pipeline identical in shape
to scan_sender_v2_threaded.py (appsrc -> videoconvert -> x265enc ->
rtph265pay -> udpsink), trimmed of SI/TI logging so we focus on the
single question: do frames traverse the 5G data plane and arrive at the
receiver container?

Pairs with /abiodun/scan_receiver_v3_oai.py running inside the
oai-perception-rx container at 192.168.70.140.

The only OAI-specific bit is `bind-address` on udpsink: it forces the
sending socket to source from the UE tunnel IP (10.0.0.2 on oaitun_ue1),
which is what makes the kernel route packets through the 5G stack
(host -> oaitun_ue1 -> nr-uesoftmodem -> rfsim -> gNB -> UPF -> bridge ->
container). Without it the kernel would shortcut via the docker bridge
and skip 5G entirely.

Env-var overrides (all optional):
    UDP_HOST   destination IP        default 192.168.70.140
    UDP_PORT   destination port      default 65000
    UE_IP      local bind (source)   default 10.0.0.2

Run:
    python3 scan_sender_v3_oai.py            # 2 Mbps default
    python3 scan_sender_v3_oai.py 4000       # 4 Mbps
"""

import glob
import os
import random
import sys

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

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst


CARLA_HOST = 'localhost'
CARLA_PORT = 2000

IM_WIDTH = 1280
IM_HEIGHT = 720
FRAMERATE = 20
FIXED_DT = 1.0 / FRAMERATE

UDP_HOST = os.environ.get('UDP_HOST', '192.168.70.140')
UDP_PORT = int(os.environ.get('UDP_PORT', '65000'))
UE_IP = os.environ.get('UE_IP', '10.0.0.2')

NUM_NPC_VEHICLES = 10
RANDOM_SEED = 0

BITRATE_KBPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2048


# --- GStreamer pipeline -----------------------------------------------------
# Same shape the team uses elsewhere. The OAI-specific change is the
# `bind-address` on udpsink: it pins the source socket to the UE tunnel IP
# so packets are routed through oaitun_ue1 instead of the docker bridge.
GST_PIPELINE = (
    f"appsrc name=src ! videoconvert ! "
    f"x265enc name=enc tune=zerolatency key-int-max={FRAMERATE} "
    f"bitrate={BITRATE_KBPS} ! "
    f"rtph265pay config-interval=1 ! "
    f"udpsink bind-address={UE_IP} host={UDP_HOST} port={UDP_PORT}"
)

Gst.init(None)
pipeline = Gst.parse_launch(GST_PIPELINE)
appsrc = pipeline.get_by_name('src')
if appsrc is None:
    raise RuntimeError("Could not locate appsrc in pipeline.")

appsrc.set_property('caps', Gst.Caps.from_string(
    f"video/x-raw,format=BGRA,width={IM_WIDTH},height={IM_HEIGHT},"
    f"framerate={FRAMERATE}/1"
))


# --- CARLA camera callback --------------------------------------------------
def on_camera_image(image):
    buf = Gst.Buffer.new_wrapped(bytes(image.raw_data))
    appsrc.emit('push-buffer', buf)


# --- Main -------------------------------------------------------------------
def main():
    print(f"[sender-oai] source bind (UE) = {UE_IP}")
    print(f"[sender-oai] dest             = {UDP_HOST}:{UDP_PORT}")
    print(f"[sender-oai] pipeline: {GST_PIPELINE}")

    pipeline.set_state(Gst.State.PLAYING)

    client = None
    world = None
    original_settings = None
    spawned_vehicles = []
    camera = None

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
        print(f"[sender-oai] Spawned {len(spawned_vehicles)} vehicles.")

        ego_vehicle = spawned_vehicles[0]
        for v in spawned_vehicles:
            v.set_autopilot(True)

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

        print(f"[sender-oai] Streaming H.265/RTP {UE_IP} -> {UDP_HOST}:{UDP_PORT} "
              f"@ {BITRATE_KBPS} kbps")
        print("[sender-oai] Press Ctrl+C to stop.")

        while True:
            world.tick()

    except KeyboardInterrupt:
        print("\n[sender-oai] Ctrl+C received, shutting down.")
    except Exception as e:
        print(f"[sender-oai] Unhandled exception: {e}")

    finally:
        if camera is not None and camera.is_alive:
            try:
                camera.stop()
                camera.destroy()
            except Exception as e:
                print(f"[sender-oai] camera cleanup: {e}")

        if client is not None and spawned_vehicles:
            try:
                client.apply_batch_sync(
                    [carla.command.DestroyActor(v.id)
                     for v in spawned_vehicles if v.is_alive], True)
            except Exception as e:
                print(f"[sender-oai] vehicles cleanup: {e}")

        if world is not None and original_settings is not None:
            try:
                world.apply_settings(original_settings)
            except Exception as e:
                print(f"[sender-oai] settings restore: {e}")

        pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    main()
