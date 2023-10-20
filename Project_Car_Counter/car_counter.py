from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("../Videos/cars.mp4")

model = YOLO("../weights/yolov8n.pt")

mask = cv2.imread("mask.jpeg")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

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
limits = [400, 297, 673, 297]
total_count = []

if not cap.isOpened():
    print("Error: Couldn't read video stream from file.")
else:
    while True:
        success, img = cap.read()
        print(img.shape)

        print(mask.shape)

        img_region = cv2.bitwise_and(img, mask)

        img_graphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
        # print(img_graphics.shape)
        # print('------------------------------')
        # print(img.shape)
        # print('------------------------------')

        # img_graphics_resized= cv2.resize(img_graphics, (256,128), interpolation=cv2.INTER_AREA, cv2.IMREAD_UNCHANGED)
        # print(img_graphics_resized.shape)
        img = cvzone.overlayPNG(img, img_graphics, [0, 0])

        results = model(img_region, stream=True)

        detections = np.empty((0, 5))

        for r in results:
            boxes = r.boxes
            for box in boxes:

                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                # confidence
                conf = math.ceil((box.conf[0] * 100)) / 100

                # Class Name
                cls = box.cls[0]  # class id type tensor
                current_class = classNames[int(cls)]

                if current_class in tracking_class_names and conf >= 0.3:
                    current_array = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, current_array))

        results_tracker = tracker.update(detections)

        # painting the line
        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

        for result in results_tracker:

            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            print(result)

            w, h = x2 - x1, y2 - y1

            cvzone.cornerRect(img,
                              (x1, y1, w, h),
                              l=10,
                              t=2,
                              colorC=(255, 242, 32),
                              colorR=(255, 0, 0),
                              rt=2)

            cvzone.putTextRect(img,
                               f'{int(id)}',
                               pos=(max(0, x1 + 10), max(35, y1 - 10)),
                               scale=2,
                               thickness=3,
                               colorR=(255, 242, 32),
                               font=cv2.FONT_HERSHEY_COMPLEX,
                               offset=1)

            # painting the center of bbox
            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), thickness=cv2.FILLED)

            if (limits[0] < cx < limits[2]) and (limits[1] - 15 < cy < limits[3] + 15):
                if total_count.count(id) == 0:
                    total_count.append(id)
                    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

        cvzone.putTextRect(img,
                           f'{len(total_count)}',
                           pos=(200, 105),
                           colorR=(255, 242, 32),
                           font=cv2.FONT_HERSHEY_COMPLEX,
                           offset=1,
                           scale=4)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
