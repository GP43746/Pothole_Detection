from ultralytics import YOLO
import torch
import os
from multiprocessing import freeze_support


def main():
   
    assert torch.cuda.is_available(), "CUDA not available. Check GPU setup."
    print("Using GPU:", torch.cuda.get_device_name(0))

   
    DATA_YAML = "dataset/yolo_v8/pothole.yaml"
    MODEL_WEIGHTS = "yolov8n.pt"

    
    model = YOLO(MODEL_WEIGHTS)

   
    results = model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=640,
    batch=8,              
    optimizer="AdamW",
    lr0=0.001,
    weight_decay=0.0005,
    patience=15,
    device=0,             
    workers=0,            
    verbose=True,
    project="runs/detect",
    name="baseline_rdd2020",
)


    print("\nTraining complete.")
    print("Best model saved at:")
    print(os.path.join(results.save_dir, "weights", "best.pt"))


if __name__ == "__main__":
    freeze_support()   
    main()

