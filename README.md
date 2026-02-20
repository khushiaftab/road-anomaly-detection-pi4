Road Anomaly Detection on Raspberry Pi 4

About the Project
This project focuses on detecting road anomalies such as unpaved roads and unmarked
speed bumps using a camera-based system. The main goal was to run an object detection
model directly on a Raspberry Pi 4 without using cloud services or a GPU.

The model was trained on a custom dataset and later optimized so that it can run on
a low-power edge device with acceptable speed.


 What This System Does
- Detects road surface problems from camera input
- Runs completely on Raspberry Pi 4
- Does not require internet connection
- Uses CPU-only inference

 Detected Classes
 - Unpaved Road
 - Unmarked Speed Bump
 - Road Surface Irregularities

 Hardware Used
- Raspberry Pi 4 Model B (4GB RAM)
- USB Camera / Dashcam
- SD Card (32GB or higher)
- HEATSINK
 Model Details
- Model Used: YOLOv8 Nano
 - Input Size: 640 × 640
- Framework: Ultralytics YOLO
 - Optimization:
- Model exported to ONNX format
- INT8 quantization applied for faster inference
- ONNX Runtime used for deployment

 Performance

On Raspberry Pi 4, the system achieves around 4-5 FPS, which is sufficient for
real-time road monitoring applications.

 Project Structure

road-anomaly-detection-pi4/
├── training scripts
├── quantization scripts
├── inference scripts
├── ONNX models
├── project report
├──README.md
└──RUN cam_infer.py on rasbery pie


 Running Inference
After setting up the environment on Raspberry Pi:

 ```bash
python fps_test.py

## Note:
Some experimental files and large datasets are not uploaded to keep the repository
clean and lightweight.
