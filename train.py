from ultralytics import YOLO
import torch

if __name__ == '__main__':

    model = YOLO("yolov8n-obb.pt")  


    results = model.train(
        data="car_bev_dataset/bev_car.yaml", # dataset
        epochs=30,          # amount of training
        imgsz=640,         # image size
        batch=16,            
        device=0,           # gpu   
        augment=True,
        mosaic=0.5,
        workers=4,
        rect=False,       
        cache='ram',
        flipud=0.5,        
        fliplr=0.5,
        amp=True,
        patience=10,
        degrees=180.0,
        project="bev_car_train",
        name="yolov8n_bev_car",
        exist_ok=True
)

    model.val()