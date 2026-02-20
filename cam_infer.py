import time
import threading
import numpy as np
import cv2
import onnxruntime as ort
from picamera2 import Picamera2
from flask import Flask, Response

MODEL_PATH = "best.onnx"
IMG_SIZE = 640
CONF_THRES = 0.25
CLASS_NAMES = ["UnMarkedBump", "UnPavedRoad", "Others"]
MAX_BOXES = 3
INFER_EVERY_N = 4
JPEG_QUALITY = 60

cv2.setNumThreads(1)
cv2.setUseOptimized(True)

latest_frame = None
output_frame = None
frame_lock = threading.Lock()
running = True

inference_fps = 0.0
output_fps = 0.0

def decode_yolo(output, img_w, img_h):
    output = output.squeeze().T
    class_scores = output[:, 4:]
    cls_ids = np.argmax(class_scores, axis=1)
    confs = class_scores[np.arange(len(cls_ids)), cls_ids]

    mask = confs > CONF_THRES
    output = output[mask]
    confs = confs[mask]
    cls_ids = cls_ids[mask]

    if len(output) == 0:
        return [], [], []

    scale = min(IMG_SIZE / img_w, IMG_SIZE / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    pad_x = (IMG_SIZE - new_w) // 2
    pad_y = (IMG_SIZE - new_h) // 2

    boxes, scores, class_ids = [], [], []

    for det, score, cls_id in zip(output, confs, cls_ids):
        x, y, w, h = det[:4]
        cx = (x - pad_x) / scale
        cy = (y - pad_y) / scale
        bw = w / scale
        bh = h / scale

        boxes.append([
            int(cx - bw / 2),
            int(cy - bh / 2),
            int(bw),
            int(bh)
        ])
        scores.append(float(score))
        class_ids.append(int(cls_id))

    idxs = cv2.dnn.NMSBoxes(boxes, scores, 0.0, 0.45)
    if len(idxs) == 0:
        return [], [], []

    idxs = idxs.flatten()[:MAX_BOXES]
    return (
        [boxes[i] for i in idxs],
        [scores[i] for i in idxs],
        [class_ids[i] for i in idxs],
    )

def camera_loop():
    global latest_frame
    picam2 = Picamera2()
    picam2.configure(
        picam2.create_video_configuration(
            main={"size": (960, 540), "format": "RGB888"},
            buffer_count=2
        )
    )
    picam2.start()
    while running:
        frame = picam2.capture_array()
        with frame_lock:
            latest_frame = frame
    picam2.stop()

def infer_loop():
    global latest_frame, output_frame, inference_fps, output_fps

    frame_id = 0
    last_boxes, last_scores, last_class_ids = [], [], []

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    sess = ort.InferenceSession(
        MODEL_PATH,
        sess_options=so,
        providers=["CPUExecutionProvider"]
    )

    infer_count = 0
    infer_t0 = time.time()
    out_count = 0
    out_t0 = time.time()

    while running:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        frame_id += 1
        img_h, img_w = frame.shape[:2]

        if frame_id % INFER_EVERY_N == 0:
            inp = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            inp = inp.astype(np.float32) / 255.0
            inp = np.transpose(inp, (2, 0, 1))
            inp = np.expand_dims(inp, axis=0)

            outputs = sess.run(None, {"images": inp})
            boxes, scores, class_ids = decode_yolo(outputs[0], img_w, img_h)

            last_boxes, last_scores, last_class_ids = boxes, scores, class_ids

            infer_count += 1
            infer_elapsed = time.time() - infer_t0
            if infer_elapsed >= 1.0:
                inference_fps = infer_count / infer_elapsed
                infer_count = 0
                infer_t0 = time.time()
        else:
            boxes, scores, class_ids = last_boxes, last_scores, last_class_ids

        for box, score, cls_id in zip(boxes, scores, class_ids):
            x, y, bw, bh = box
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{CLASS_NAMES[cls_id]} {score:.2f}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        out_count += 1
        out_elapsed = time.time() - out_t0
        if out_elapsed >= 1.0:
            output_fps = out_count / out_elapsed
            out_count = 0
            out_t0 = time.time()

        cv2.putText(
            frame,
            f"Inference FPS: {inference_fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )
        cv2.putText(
            frame,
            f"Output FPS: {output_fps:.2f}",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        with frame_lock:
            output_frame = frame

app = Flask(__name__)

def mjpeg_generator():
    global output_frame
    while True:
        with frame_lock:
            frame = output_frame
        if frame is None:
            time.sleep(0.01)
            continue
        ret, jpeg = cv2.imencode(
            ".jpg", frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )
        if not ret:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            jpeg.tobytes() +
            b"\r\n"
        )
        time.sleep(0.03)

@app.route("/")
def video_feed():
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

def run_server():
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=False,
        use_reloader=False,
        threaded=True
    )

t_cam = threading.Thread(target=camera_loop, daemon=True)
t_inf = threading.Thread(target=infer_loop, daemon=True)
t_srv = threading.Thread(target=run_server, daemon=True)

t_cam.start()
t_inf.start()
t_srv.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    running = False
    print("Stopping...")