from ultralytics import YOLO
import cv2  
model = YOLO('../yolo_weights/yolov8l.pt')
results = model("chapter5_running_YOLO/Images/3.png", show=True)
cv2.waitKey(0)