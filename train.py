from ultralytics import YOLO

model = YOLO("yolo11s.pt")

model.train(
    data    = "data.yaml",
    epochs  = 50,         
    imgsz   = 640,        
    batch   = 16,          
    name    = "hail_yolo_v1",
    project = "runs"     
)