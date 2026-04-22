"""
real_camera_node.py
────────────────────────────────────────────────────────────────────
Real Camera Object Detection Node — YOLO Primary
Eye-on-Hand Robot System (ROS2 / Standalone)

Detects a specific object class using YOLOv8 and publishes the
bounding-box result to /simulated_yolo_bbox so ibvs_controller
picks up the signal and drives the robot.

Detection pipeline:
  OpenCV webcam → YOLOv8 inference → class filter → EMA smooth → publish

Published topics:
  /simulated_yolo_bbox  (geometry_msgs/Point)
    .x  normalized bbox centre-u  [0..1],  -1 = no detection
    .y  normalized bbox centre-v  [0..1],  -1 = no detection
    .z  normalized bbox area      [0..1],   0 = no detection

Subscribed topics:
  /target_ik_point  (geometry_msgs/Point)
    Sentinel (-9,-9,-9) resets tracking after delivery

────────────────
Standalone (no ROS2):
  python real_camera_node.py                        # track person (default)
  python real_camera_node.py --class bottle         # by name
  python real_camera_node.py --class 39             # by COCO index
  python real_camera_node.py --list-classes         # show all class indices

ROS2:
  ros2 run <pkg> real_camera_node \\
    --ros-args \\
    -p model_path:=yolov8n.pt \\
    -p target_class:=person \\
    -p camera_id:=0 \\
    -p conf_threshold:=0.45 \\
    -p show_preview:=true
────────────────
"""

import sys
import argparse
import time

import cv2
import numpy as np

# ── ROS2 optional ────────────────────────────────────────────────
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Point
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    class Node:
        def __init__(self, *a, **kw): pass
        def declare_parameter(self, *a, **kw): pass
        def get_parameter(self, n):   return _Param()
        def create_publisher(self, *a, **kw): return _NullPub()
        def create_subscription(self, *a, **kw): return None
        def create_timer(self, *a, **kw): return None
        def get_logger(self):          return _Logger()
        def destroy_node(self):        pass
    class _Param:
        def get_parameter_value(self): return self
        string_value  = ''
        integer_value = 0
        bool_value    = True
        double_value  = 0.45
    class _NullPub:
        def publish(self, _): pass
    class _Logger:
        def info(self, m):  print(f"[INFO]  {m}")
        def warn(self, m):  print(f"[WARN]  {m}")
        def error(self, m): print(f"[ERROR] {m}")
    class Point:
        def __init__(self): self.x = self.y = self.z = 0.0

# ── YOLO ─────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ── Defaults ─────────────────────────────────────────────────────
DEFAULT_MODEL = 'yolov8n.pt'
DEFAULT_CLASS = 'car'       # COCO index 0
DEFAULT_CONF  = 0.15
DEFAULT_CAM   = 0

# ════════════════════════════════════════════════════════════════

def load_model(model_path: str) -> "YOLO":
    """Load a YOLO model; raises RuntimeError if ultralytics missing."""
    if not YOLO_AVAILABLE:
        raise RuntimeError(
            "ultralytics is not installed.\n"
            "Run:  pip install ultralytics")
    print(f"[INFO] Loading YOLO model: {model_path} …")
    return YOLO(model_path)


def resolve_class(model: "YOLO", cls_input: str) -> str:
    """
    Convert a class specifier (name OR COCO index) to a lowercase class name.
    Raises SystemExit on invalid input.
    """
    cls_input = cls_input.strip()
    if cls_input.lstrip('-').isdigit():
        idx = int(cls_input)
        if idx not in model.names:
            print(f"[ERROR] Index {idx} not in model. Use --list-classes.")
            sys.exit(1)
        name = model.names[idx].lower()
        print(f"[INFO] Index {idx} → '{name}'")
        return name
    # By name
    name_map = {v.lower(): k for k, v in model.names.items()}
    if cls_input.lower() not in name_map:
        print(f"[ERROR] Class '{cls_input}' not found. Use --list-classes.")
        sys.exit(1)
    return cls_input.lower()


def print_classes(model: "YOLO", model_path: str):
    """Print full COCO class index table and exit."""
    print(f"\nClass index table for '{model_path}':\n")
    for idx, name in sorted(model.names.items()):
        print(f"  {idx:>3} : {name}")
    print()


# ════════════════════════════════════════════════════════════════
#  YOLO DETECTOR  (framework-agnostic)
# ════════════════════════════════════════════════════════════════

class YOLODetector:
    """
    Wraps a pre-loaded YOLOv8 model with EMA smoothing and overlay drawing.

    Parameters
    ----------
    model        : already-loaded YOLO instance
    target_class : class name to track (must be lowercase, validated beforehand)
    conf_threshold : minimum detection confidence
    camera_id    : OpenCV VideoCapture index
    ema_alpha    : EMA smoothing factor [0..1]
    """

    BOX_COLOR   = (0, 220, 80)   # BGR
    LABEL_COLOR = (30, 30, 30)

    def __init__(self, model: "YOLO", target_class: str,
                 conf_threshold: float = 0.45,
                 camera_id: int = 0,
                 ema_alpha: float = 0.35):

        self.model          = model
        self.target_class   = target_class.lower()
        self.conf_threshold = conf_threshold
        self.ema_alpha      = ema_alpha

        # Camera
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {camera_id}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] Camera {camera_id} opened  ({self.W}×{self.H})")
        print(f"[INFO] Tracking class: '{self.target_class}'  conf≥{conf_threshold}")

        # EMA state
        self._nx   = 0.5
        self._ny   = 0.5
        self._area = 0.0
        self._has  = False

        # FPS
        self._t_prev = time.time()
        self.fps     = 0.0

    # ── Inference ─────────────────────────────────────────────────
    def detect_frame(self, frame):
        """
        Run YOLO on `frame`, draw overlay in-place.
        Returns (norm_x, norm_y, norm_area) EMA-smoothed, or None.
        """
        results  = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        best_box = None
        best_conf = 0.0

        for box in results.boxes:
            if results.names[int(box.cls)].lower() != self.target_class:
                continue
            conf = float(box.conf)
            if conf > best_conf:
                best_conf = conf
                best_box  = box

        # FPS
        t_now = time.time()
        self.fps = 0.9 * self.fps + 0.1 / max(t_now - self._t_prev, 1e-6)
        self._t_prev = t_now

        if best_box is None:
            self._has  = False
            self._area = 0.0
            self._draw_status(frame, detected=False)
            return None

        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(self.W-1, x2); y2 = min(self.H-1, y2)

        raw_cx   = ((x1 + x2) / 2) / self.W
        raw_cy   = ((y1 + y2) / 2) / self.H
        raw_area = ((x2-x1) * (y2-y1)) / (self.W * self.H)

        a = self.ema_alpha
        if not self._has:
            self._nx, self._ny, self._area = raw_cx, raw_cy, raw_area
            self._has = True
        else:
            self._nx   = a * raw_cx   + (1-a) * self._nx
            self._ny   = a * raw_cy   + (1-a) * self._ny
            self._area = a * raw_area + (1-a) * self._area

        self._draw_box(frame, x1, y1, x2, y2, best_conf)
        sx, sy = int(self._nx * self.W), int(self._ny * self.H)
        cv2.circle(frame, (sx, sy), 6, (255, 255, 0), -1)
        cv2.circle(frame, (sx, sy), 6, (0, 0, 0), 1)
        self._draw_status(frame, detected=True, conf=best_conf)

        return self._nx, self._ny, self._area

    def reset(self):
        self._has  = False
        self._area = 0.0

    def read_frame(self):
        return self.cap.read()

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

    # ── Drawing ───────────────────────────────────────────────────
    def _draw_box(self, frame, x1, y1, x2, y2, conf):
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.BOX_COLOR, 2)
        label = f"{self.target_class}  {conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1-lh-8), (x1+lw+4, y1), self.BOX_COLOR, -1)
        cv2.putText(frame, label, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, self.LABEL_COLOR, 1)

    def _draw_status(self, frame, detected, conf=0.0):
        h, w = frame.shape[:2]

        # ── Crosshair (principal point) ───────────────────────────
        cx, cy = w // 2, h // 2
        cv2.line(frame, (cx, 0),  (cx, h),  (50, 50, 50), 1)
        cv2.line(frame, (0, cy),  (w, cy),  (50, 50, 50), 1)
        # Label principal-point coordinates in normalised space
        cv2.putText(frame, "u=0.5", (cx + 4, h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)
        cv2.putText(frame, "v=0.5", (6, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)

        # ── Axis widget (bottom-right corner) ────────────────────
        ox, oy = w - 60, h - 60        # widget origin
        AL = 40                         # arrow length (px)

        cv2.circle(frame, (ox, oy), AL + 8, (20, 20, 20), -1)
        cv2.circle(frame, (ox, oy), AL + 8, (60, 60, 60),  1)

        # X axis → image RIGHT  (red)
        cv2.arrowedLine(frame, (ox, oy), (ox + AL, oy),
                        (50, 50, 200), 2, tipLength=0.28)
        cv2.putText(frame, "X", (ox + AL + 3, oy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (50, 50, 200), 1)

        # Y axis → image DOWN   (green)
        cv2.arrowedLine(frame, (ox, oy), (ox, oy + AL),
                        (50, 200, 50), 2, tipLength=0.28)
        cv2.putText(frame, "Y", (ox - 12, oy + AL + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (50, 200, 50), 1)

        # Origin dot
        cv2.circle(frame, (ox, oy), 4, (200, 200, 50), -1)

        # ── HUD: status + live bbox coords ────────────────────────
        clr = (0, 220, 80) if detected else (0, 60, 220)
        txt = "TRACKING" if detected else "SEARCHING"
        cv2.putText(frame, f"FPS:{self.fps:4.1f}  [{txt}]",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, clr, 2)
        cv2.putText(frame, f"target: {self.target_class}",
                    (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)
        if detected and self._has:
            cv2.putText(frame,
                        f"bx={self._nx:.3f}  by={self._ny:.3f}  area={self._area:.4f}",
                        (8, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1)




# ════════════════════════════════════════════════════════════════
#  ROS2 NODE
# ════════════════════════════════════════════════════════════════

class RealCameraNode(Node):
    def __init__(self):
        super().__init__('real_camera_node')

        # Parameters (all with consistent defaults matching CLI)
        self.declare_parameter('model_path',     DEFAULT_MODEL)
        self.declare_parameter('target_class',   DEFAULT_CLASS)  # name OR COCO index
        self.declare_parameter('camera_id',       DEFAULT_CAM)
        self.declare_parameter('conf_threshold',  DEFAULT_CONF)
        self.declare_parameter('show_preview',    True)

        model_path  = self.get_parameter('model_path').get_parameter_value().string_value
        target_raw  = self.get_parameter('target_class').get_parameter_value().string_value
        camera_id   = self.get_parameter('camera_id').get_parameter_value().integer_value
        conf_thr    = self.get_parameter('conf_threshold').get_parameter_value().double_value
        self.show   = self.get_parameter('show_preview').get_parameter_value().bool_value

        # Load model and resolve class (supports index or name)
        _model     = load_model(model_path)
        target_cls = resolve_class(_model, target_raw)

        self.detector = YOLODetector(
            model          = _model,
            target_class   = target_cls,
            conf_threshold = conf_thr,
            camera_id      = camera_id,
        )

        self.bbox_pub   = self.create_publisher(Point, '/simulated_yolo_bbox', 10)
        self.target_sub = self.create_subscription(
            Point, '/target_ik_point', self._sentinel_cb, 10)
        self.timer = self.create_timer(1.0/30.0, self._loop)

        self.get_logger().info(
            f"Real Camera Node ready | model={model_path} | "
            f"class='{target_cls}' | conf≥{conf_thr}")

    def _sentinel_cb(self, msg: Point):
        if msg.x == -9.0 and msg.y == -9.0 and msg.z == -9.0:
            self.detector.reset()
            self.get_logger().info("Sentinel → tracking reset.")
            self._publish(-1.0, -1.0, 0.0)

    def _loop(self):
        ok, frame = self.detector.read_frame()
        if not ok:
            self._publish(-1.0, -1.0, 0.0)
            return

        result = self.detector.detect_frame(frame)
        if result is not None:
            self._publish(*result)
        else:
            self._publish(-1.0, -1.0, 0.0)

        if self.show:
            cv2.imshow("Real Camera — YOLO Detection", frame)
            cv2.waitKey(1)

    def _publish(self, x, y, z):
        msg = Point()
        msg.x, msg.y, msg.z = float(x), float(y), float(z)
        self.bbox_pub.publish(msg)

    def destroy_node(self):
        self.detector.release()
        super().destroy_node()


# ════════════════════════════════════════════════════════════════
#  STANDALONE (no ROS2)
# ════════════════════════════════════════════════════════════════

def run_standalone(model: "YOLO", target_class: str, conf: float,
                   cam: int):
    """Camera loop — model already loaded and class already resolved."""
    detector = YOLODetector(
        model          = model,
        target_class   = target_class,
        conf_threshold = conf,
        camera_id      = cam,
    )
    print(f"[INFO] Press 'q' to quit | 'r' to reset EMA")

    while True:
        ok, frame = detector.read_frame()
        if not ok:
            print("[ERROR] Cannot read frame."); break

        result = detector.detect_frame(frame)
        if result is not None:
            nx, ny, area = result
            print(f"\r  bbox_x={nx:.3f}  bbox_y={ny:.3f}  area={area:.4f}"
                  f"  [TRACKING: {detector.target_class}]  ", end='')
        else:
            print(f"\r  Searching for '{detector.target_class}' …"
                  f"                        ", end='')

        cv2.imshow("Real Camera — YOLO Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            detector.reset()
            print(f"\n[INFO] EMA reset.")

    detector.release()
    print()


# ════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Real Camera YOLO Detection Node",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "examples:\n"
            "  python real_camera_node.py                        # track person (default)\n"
            "  python real_camera_node.py --class bottle         # by name\n"
            "  python real_camera_node.py --class 39             # by COCO index (bottle)\n"
            "  python real_camera_node.py --list-classes         # show all index→name table\n"
            "  python real_camera_node.py --model yolov8s.pt --class cup --conf 0.5"
        )
    )
    parser.add_argument('--model',  default=DEFAULT_MODEL,
                        help=f"YOLO model weights path (default: {DEFAULT_MODEL})")
    parser.add_argument('--class',  default=DEFAULT_CLASS, dest='cls',
                        help=f"Class name OR COCO index to track (default: {DEFAULT_CLASS} / 0)")
    parser.add_argument('--cam',    default=DEFAULT_CAM, type=int,
                        help=f"OpenCV camera index (default: {DEFAULT_CAM})")
    parser.add_argument('--conf',   default=DEFAULT_CONF, type=float,
                        help=f"Detection confidence threshold (default: {DEFAULT_CONF})")
    parser.add_argument('--list-classes', action='store_true',
                        help="Print class index table and exit")
    parser.add_argument('--standalone', action='store_true',
                        help="Force standalone mode (no ROS2)")

    cli_args = [a for a in sys.argv[1:]
                if not (a.startswith('--ros-args') or a in
                        ('-r', '-p', '__node:=', '__ns:='))]
    args, _ = parser.parse_known_args(cli_args)

    if not YOLO_AVAILABLE:
        print("[ERROR] ultralytics not installed.  Run: pip install ultralytics")
        sys.exit(1)

    # Load model ONCE — shared by both --list-classes and run_standalone
    model = load_model(args.model)

    if args.list_classes:
        print_classes(model, args.model)
        sys.exit(0)

    # Resolve class (name or index) — validated here, passed downstream
    target_class = resolve_class(model, args.cls)
    print(f"[INFO] Target class: '{target_class}'")

    if args.standalone or not ROS2_AVAILABLE:
        run_standalone(model, target_class, args.conf, args.cam)
    else:
        rclpy.init()
        node = RealCameraNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
