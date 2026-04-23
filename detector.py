"""
detector.py
-----------
Lightweight YOLO-based pattern/object detector used by the Basler GUI.

Wraps `ultralytics.YOLO` (COCO-pretrained) and exposes a simple API:

    d = Detector()                   # loads yolov8n.pt on first use (auto-download)
    d.set_classes(["person", "tv", "laptop"])
    d.set_conf(0.35)
    annotated_bgr, detections = d.infer(bgr_image)

No GPU required; runs on CPU ~15-25 FPS on modern laptops at 640px imgsz.
"""

from __future__ import annotations

import threading
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

# The full COCO class list the default model was trained on (80 classes).
# Useful ones for the user: person, tv, laptop, mouse, remote, keyboard,
# cell phone, book, clock, chair, couch, potted plant, bed, dining table,
# toilet, microwave, oven, toaster, sink, refrigerator, bottle, cup, etc.
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# Sensible default subset matching the user's ask ("human, tv, monitor, etc.")
DEFAULT_CLASSES = ["person", "tv", "laptop", "cell phone", "book", "keyboard",
                   "mouse", "remote", "chair", "bottle", "cup"]


class Detector:
    """Thread-safe YOLOv8n detector with class filtering and conf threshold."""

    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self._model = None
        self._lock = threading.Lock()
        self._load_error: Optional[str] = None

        # Filtering state
        self.enabled_classes: List[str] = list(DEFAULT_CLASSES)
        self.conf_thresh: float = 0.35
        self.imgsz: int = 640

    # -----------------------------------------------------------------
    def ensure_loaded(self) -> None:
        """Lazy-load the model (auto-downloads yolov8n.pt on first call)."""
        if self._model is not None:
            return
        try:
            # Imported lazily so the GUI can start even without ultralytics.
            from ultralytics import YOLO  # type: ignore
            self._model = YOLO(self.model_path)
            self._load_error = None
        except Exception as e:
            self._load_error = f"{type(e).__name__}: {e}"
            self._model = None

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    # -----------------------------------------------------------------
    def set_classes(self, names: Iterable[str]) -> None:
        self.enabled_classes = [n for n in names if n in COCO_CLASSES]

    def set_conf(self, conf: float) -> None:
        self.conf_thresh = max(0.01, min(0.99, float(conf)))

    # -----------------------------------------------------------------
    def _class_indices(self) -> Optional[List[int]]:
        if not self.enabled_classes:
            return None  # empty -> detect all
        return [COCO_CLASSES.index(n) for n in self.enabled_classes
                if n in COCO_CLASSES]

    # -----------------------------------------------------------------
    def infer(self, bgr: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """Run inference on a BGR image. Returns (annotated_bgr, detections).

        detections = [{"cls": "person", "conf": 0.87,
                       "xyxy": (x1,y1,x2,y2)}, ...]
        Safe to call even if the model failed to load -- returns the input
        image unchanged.
        """
        if self._model is None:
            return bgr, []

        cls_idx = self._class_indices()

        with self._lock:
            try:
                results = self._model.predict(
                    source=bgr,
                    conf=self.conf_thresh,
                    imgsz=self.imgsz,
                    classes=cls_idx,
                    device=self.device,
                    verbose=False,
                )
            except Exception:
                return bgr, []

        if not results:
            return bgr, []

        r = results[0]
        detections: List[dict] = []
        annotated = bgr.copy()

        try:
            names = r.names  # {idx: "class_name"}
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                return annotated, []

            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, cls):
                label = names.get(int(k), str(int(k)))
                color = _color_for_class(int(k))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                txt = f"{label} {c:.2f}"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX,
                                              0.5, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 6),
                              (x1 + tw + 4, y1), color, -1)
                cv2.putText(annotated, txt, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
                detections.append({
                    "cls": label,
                    "conf": float(c),
                    "xyxy": (int(x1), int(y1), int(x2), int(y2)),
                })
        except Exception:
            return bgr, []

        return annotated, detections


# ---------------------------------------------------------------------------
def _color_for_class(idx: int) -> Tuple[int, int, int]:
    """Deterministic BGR color per class index."""
    np.random.seed(idx * 97 + 13)
    c = tuple(int(x) for x in np.random.randint(64, 256, size=3))
    return c  # type: ignore[return-value]
