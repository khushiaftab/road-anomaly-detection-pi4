import cv2
import numpy as np
from pathlib import Path
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType
)

MODEL_FP32 = "runs/detect/road_project/yolov8n_pi4/weights/best.onnx"
MODEL_INT8 = "runs/detect/road_project/yolov8n_pi4/weights/best_int8.onnx"
CALIB_DIR = Path("calib_images")

IMG_SIZE = 640


class ImageCalibrationReader(CalibrationDataReader):
    def __init__(self, image_dir):
        self.image_paths = list(image_dir.iterdir())
        self.index = 0

    def get_next(self):
        if self.index >= len(self.image_paths):
            return None

        img = cv2.imread(str(self.image_paths[self.index]))
        self.index += 1

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0

        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        img = np.expand_dims(img, axis=0)   # BCHW

        return {"images": img}


reader = ImageCalibrationReader(CALIB_DIR)

quantize_static(
    model_input=MODEL_FP32,
    model_output=MODEL_INT8,
    calibration_data_reader=reader,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8,
    per_channel=False,
    reduce_range=True
)

print("INT8 quantization complete.")