from ultralytics import YOLO

def train_full():
    model = YOLO("yolov8n.pt")

    model.train(
        data="road_data.yaml",
        epochs=100,           # full convergence
        imgsz=640,
        batch=16,             # reduce to 8 if RAM constrained
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        mosaic=1.0,           # robustness
        mixup=0.1,
        close_mosaic=10,      # turn OFF mosaic near end
        patience=20,
        device="cpu",         # GPU if available, CPU is fine
        workers=4,
        project="road_project",
        name="yolov8n_pi4"
    )

if __name__ == "__main__":
    train_full()