import time
import cv2
import vmbpy
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("captures")
INTERVAL_SECS = 10 #you can change it to whatever interval you want


def save_frame(frame: vmbpy.Frame, output_dir: Path, suffix: str = "") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"capture{suffix}_{timestamp}.png"
    bgr_frame = frame.convert_pixel_format(vmbpy.PixelFormat.Bgr8)
    cv2.imwrite(str(path), bgr_frame.as_numpy_ndarray())
    return path


def _open_camera(vmb: vmbpy.VmbSystem, camera_id: str | None):
    cameras = vmb.get_all_cameras()
    if not cameras:
        raise RuntimeError("No cameras found.")
    return next((c for c in cameras if c.get_id() == camera_id), cameras[0])


def run(camera_id: str | None = None):
    output_dir = Path("/home/kpatel2/ws_moveit/src/hello_moveit/vimba_images")
    output_dir.mkdir(parents=True, exist_ok=True)

    with vmbpy.VmbSystem.get_instance() as vmb:
        cam = _open_camera(vmb, camera_id)
        print(f"Using camera: {cam.get_id()}")

        with cam:
            try:
                while True:
                    frame = cam.get_frame(timeout_ms=2000)
                    path = save_frame(frame, output_dir)
                    print(f"Saved: {path}")
                    time.sleep(INTERVAL_SECS)
            except KeyboardInterrupt:
                print("Stopped.")


def capture_once(cycle: int | None = None, camera_id: str | None = None) -> Path:
    output_dir = Path("/home/kpatel2/ws_moveit/src/hello_moveit/vimba_images")
    output_dir.mkdir(parents=True, exist_ok=True)

    with vmbpy.VmbSystem.get_instance() as vmb:
        cam = _open_camera(vmb, camera_id)
        print(f"Using camera: {cam.get_id()}")

        with cam:
            frame = cam.get_frame(timeout_ms=2000)
            suffix = f"_cycle{cycle}" if cycle is not None else ""
            path = save_frame(frame, output_dir, suffix=suffix)
            print(f"Saved: {path}")
            return path


if __name__ == "__main__":
    run()


##### ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e use_mock_hardware:=false headless_mode:=true robot_ip:=192.168.2.100 

######## ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5e launch_rviz:=false
