from ultralytics import YOLO
import cv2
import cvzone
import math 


cap = cv2.VideoCapture("Videos/cars.mp4")


model = YOLO("../yolo_weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
tracking_class_names = ["motorbike", "car", "bus", "truck"]

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:

            # bounding box 
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 255), thickness=3)
            # print(x1, y1, x2, y2)

            w, h = x2 - x1, y2 - y1 
            # bbox = int(x1), int(y1), int(w), int(h)

            
            # confidence 
            conf = math.ceil((box.conf[0]*100))/100
            
            # Class Name
            cls = box.cls[0] # class id type tensor
            current_class = classNames[int(cls)]
            if current_class in tracking_class_names and conf >= 0.3:
                
                cvzone.cornerRect(img, 
                              (x1, y1, w, h), 
                              l= 10,
                              t=2, 
                              colorC=(255, 242, 32), 
                              colorR=(32, 243, 123))
            
                cvzone.putTextRect(img,
                                f'{current_class} {conf}',
                                pos=(max(0, x1 + 10), max(35, y1 - 10)),
                                scale=0.6, 
                                thickness=1,
                                colorR=(255, 242, 32),
                                font=cv2.FONT_HERSHEY_COMPLEX,
                                offset=3)


    cv2.imshow("Image", img)
    cv2.waitKey(0) 
