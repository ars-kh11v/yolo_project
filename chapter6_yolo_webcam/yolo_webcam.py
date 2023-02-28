from ultralytics import YOLO
import cv2
import cvzone
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("../yolo_weights/yolov8n.pt")


while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # x1, y1, x2, y2 = box.xyxy[0]
            # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 255), thickness=3)
            # print(x1, y1, x2, y2)

            x1, y1, w, h = box.xywh[0]
            bbox = int(x1), int(y1), int(w), int(h)
            cvzone.cornerRect(img, bbox)
    cv2.imshow("Image", img)
    cv2.waitKey(1) 
