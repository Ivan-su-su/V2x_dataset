import math
import threading
import time

import carla
import cv2
import numpy as np
from flask import Flask, Response


CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000
WEB_HOST = "127.0.0.1"
WEB_PORT = 8080

IMAGE_WIDTH = 960
IMAGE_HEIGHT = 540
JPEG_QUALITY = 80


app = Flask(__name__)

latest_frames = {
    "vehicle": None,
    "uav": None,
    "rsu": None,
}

frame_lock = threading.Lock()
stop_event = threading.Event()
owned_actors = []


HTML = """
<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>Active AirV2X Viewer</title>
    <style>
        body {
            margin: 0;
            background: #111827;
            color: white;
            font-family: Arial, sans-serif;
        }
        h1 {
            margin: 16px 20px 4px;
            font-size: 24px;
        }
        p {
            margin: 4px 20px 16px;
            color: #9ca3af;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            padding: 0 20px 20px;
        }
        .panel {
            background: #1f2937;
            border-radius: 10px;
            padding: 10px;
        }
        .panel h2 {
            margin: 0 0 8px;
            font-size: 17px;
        }
        .panel img {
            display: block;
            width: 100%;
            background: black;
            border-radius: 6px;
        }
        .wide {
            grid-column: 1 / 3;
        }
    </style>
</head>
<body>
    <h1>Active AirV2X Live Viewer</h1>
    <p>CARLA 0.9.16 · Vehicle / UAV / RSU</p>

    <div class="grid">
        <div class="panel">
            <h2>Vehicle Front Camera</h2>
            <img src="/stream/vehicle">
        </div>

        <div class="panel">
            <h2>RSU Camera</h2>
            <img src="/stream/rsu">
        </div>

        <div class="panel wide">
            <h2>UAV Top-down Camera</h2>
            <img src="/stream/uav">
        </div>
    </div>
</body>
</html>
"""


def image_callback(name, image):
    """Convert CARLA BGRA image into browser-friendly JPEG."""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))

    bgr = array[:, :, :3]

    success, encoded = cv2.imencode(
        ".jpg",
        bgr,
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
    )

    if success:
        with frame_lock:
            latest_frames[name] = encoded.tobytes()


def stream_frames(name):
    while not stop_event.is_set():
        with frame_lock:
            jpeg = latest_frames.get(name)

        if jpeg is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg
                + b"\r\n"
            )

        time.sleep(0.04)


@app.route("/")
def index():
    return HTML


@app.route("/stream/<name>")
def stream(name):
    if name not in latest_frames:
        return "Unknown camera", 404

    return Response(
        stream_frames(name),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def look_at(source, target):
    """Create a CARLA rotation looking from source toward target."""
    dx = target.x - source.x
    dy = target.y - source.y
    dz = target.z - source.z

    horizontal_distance = math.sqrt(dx * dx + dy * dy)

    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(dz, horizontal_distance))

    return carla.Rotation(
        pitch=pitch,
        yaw=yaw,
        roll=0.0,
    )


def create_camera(world, name, transform, attach_to=None):
    blueprint = world.get_blueprint_library().find("sensor.camera.rgb")

    blueprint.set_attribute("image_size_x", str(IMAGE_WIDTH))
    blueprint.set_attribute("image_size_y", str(IMAGE_HEIGHT))
    blueprint.set_attribute("fov", "90")
    blueprint.set_attribute("sensor_tick", "0.05")

    camera = world.spawn_actor(
        blueprint,
        transform,
        attach_to=attach_to,
    )

    camera.listen(
        lambda image, camera_name=name:
        image_callback(camera_name, image)
    )

    owned_actors.append(camera)
    return camera


def spawn_vehicle(world):
    blueprints = world.get_blueprint_library().filter("vehicle.*")
    vehicle_blueprint = blueprints[0]

    if vehicle_blueprint.has_attribute("role_name"):
        vehicle_blueprint.set_attribute(
            "role_name",
            "activeair_viewer_ego",
        )

    for spawn_point in world.get_map().get_spawn_points():
        vehicle = world.try_spawn_actor(
            vehicle_blueprint,
            spawn_point,
        )

        if vehicle is not None:
            owned_actors.append(vehicle)
            vehicle.set_autopilot(True, 8000)
            return vehicle

    raise RuntimeError("无法找到可用的车辆出生点")


def follow_vehicle_with_uav(vehicle, uav_camera):
    """Debug only: keep the aerial camera above the vehicle."""
    while not stop_event.is_set():
        if not vehicle.is_alive or not uav_camera.is_alive:
            break

        location = vehicle.get_location()

        uav_camera.set_transform(
            carla.Transform(
                carla.Location(
                    x=location.x,
                    y=location.y,
                    z=location.z + 40.0,
                ),
                carla.Rotation(
                    pitch=-90.0,
                    yaw=0.0,
                    roll=0.0,
                ),
            )
        )

        time.sleep(0.1)


def cleanup():
    stop_event.set()

    for actor in reversed(owned_actors):
        try:
            if actor.is_alive:
                if "sensor." in actor.type_id:
                    actor.stop()
                actor.destroy()
        except RuntimeError:
            pass

    print("Viewer actors cleaned up")


def main():
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(20.0)

    world = client.get_world()

    print("Connected:", client.get_server_version())
    print("Map:", world.get_map().name)

    vehicle = spawn_vehicle(world)
    vehicle_location = vehicle.get_location()

    create_camera(
        world,
        "vehicle",
        carla.Transform(
            carla.Location(x=1.5, z=2.4),
            carla.Rotation(pitch=-5.0),
        ),
        attach_to=vehicle,
    )

    uav_camera = create_camera(
        world,
        "uav",
        carla.Transform(
            carla.Location(
                x=vehicle_location.x,
                y=vehicle_location.y,
                z=vehicle_location.z + 40.0,
            ),
            carla.Rotation(pitch=-90.0),
        ),
    )

    rsu_location = carla.Location(
        x=vehicle_location.x + 15.0,
        y=vehicle_location.y + 15.0,
        z=vehicle_location.z + 8.0,
    )

    create_camera(
        world,
        "rsu",
        carla.Transform(
            rsu_location,
            look_at(rsu_location, vehicle_location),
        ),
    )

    uav_thread = threading.Thread(
        target=follow_vehicle_with_uav,
        args=(vehicle, uav_camera),
        daemon=True,
    )
    uav_thread.start()

    print(f"Viewer running at http://{WEB_HOST}:{WEB_PORT}")

    try:
        app.run(
            host=WEB_HOST,
            port=WEB_PORT,
            threaded=True,
            use_reloader=False,
        )
    finally:
        cleanup()


if __name__ == "__main__":
    main()