from ultralytics import YOLO
import cv2
import time
import random



class Camera:
    def __init__(self, source = 0, width = 480, height = 640, mirror = True):
        self.source = source
        self.width = width
        self.height = height
        self.mirror = mirror
        self.cap = None

    def start(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera: {self.source}")
        if self.width: self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height: self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return self

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return False, None
        if self.mirror:
            frame = cv2.flip(frame, 1)
        return True, frame

    def stop(self):
        if self.cap:
            self.cap.release()
            self.cap = None

class Yolo_detector:
    def __init__(self, weights = "yolov8n.pt", conf = 0.25, imgsz = 640, target_names = ("person",)):
        self.weights = weights
        self.conf = conf
        self.imgsz = imgsz
        self.target_names = set(target_names)
        self._model = None
        self._colors = {}

    def _ensure_model(self):
        if self._model is None:
            self._model = YOLO(self.weights)

    def _color_for(self, cls_id):
        if cls_id not in self._colors:
            self._colors[cls_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0,255))
        return self._colors[cls_id]

    def detect_and_draw(self, frame_bgr):
        self._ensure_model()
        res = self._model(frame_bgr, conf = self.conf, imgsz = self.imgsz, verbose = False)[0]
        names = res.names

        detections = []
        if res.boxes is not None and len(res.boxes):
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                cls_id = int(box.cls[0])
                name = names.get(cls_id, str(cls_id))
                score = float(box.conf[0])

                if self.target_names and name not in self.target_names:
                    continue

                detections.append({"name": name, "conf": score, "box": (x1, y1, x2, y2), "cls": cls_id})
        drawn = frame_bgr.copy()

        for d in detections:
            x1, y1, x2, y2 = d["box"]
            color = self._color_for(d["cls"])
            cv2.rectangle(drawn, (x1, y1), (x2, y2), color, 2)
            cv2.putText(drawn, f'{d["name"]} {d["conf"]:.2f}', (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            return detections, drawn

class Alert:
    def __init__(self, cooldown_sec = 8, beep = True):
        self.cooldown_sec = cooldown_sec
        self.beep = beep
        self._last = 0

    def _do_beep(self):
        try:
            import winsound
            winsound.Beep(1000, 250)
        except Exception:
            print("\a", end = "")

    def maybe_alert(self, detections):
        if not detections:
            return
        now = time.time()
        if now - self._last < self.cooldown_sec:
            return
        self._last = now

        summary = ",".join(f"{d['name']} ({d['conf']:.2f})" for d in detections)
        print(f"[ALERT] {time.strftime('%H: %M: %S')} -> {summary}")
        if self.beep:
            self._do_beep()


def main():
    cam = Camera(source = 0, width = 960, height = 540, mirror = False).start()
    yolo = Yolo_detector(weights = "yolov8n.pt", conf = 0.35, imgsz = 640, target_names = ("person", ))
    alert = Alert(cooldown_sec = 0, beep = True)

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                print("stream ended")
                break

            detecttions, drawn = yolo.detect_and_draw(frame)
            alert.maybe_alert(detecttions)

            cv2.imshow("Minimal YOLOv8 (q = quit)", drawn)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

















