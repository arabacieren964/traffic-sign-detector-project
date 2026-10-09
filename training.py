from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")

    model.train(
        data="C:/Users/ASUS/Desktop/YOLOv8 Traffic Sign Detector Model/data/data.yaml",
        epochs=10,
        imgsz=640,
        batch=16,
        name="traffic-sign model",
        lr0=0.01,
        optimizer="SGD",
        weight_decay=0.0005,
        momentum=0.935,
        patience=5,
        workers=0,  # RTX 4060'ı hızlı beslemek için 4 idealdir
        device="cuda",  # 1. GPU (RTX 4060)
        save=True,
        save_period=1,
        val=True,
        verbose=True,
    )