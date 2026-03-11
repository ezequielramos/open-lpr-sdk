from ultralytics import YOLO
import os
import cv2

model = YOLO("whoami.pt")

for image_path in os.listdir('dataset'):
    image = cv2.imread(f"dataset/{image_path}")
    print(f"dataset/{image_path}")

    results = model(image)

    result = results[0]

    boxes = result.boxes.xyxy
    confs = result.boxes.conf

    for box, conf in zip(boxes, confs):

        print("rect confidence:", float(conf))

        if float(conf) < 0.75:
            print('not enough confidence')
            break

        x1, y1, x2, y2 = map(int, box)

        plate = image[y1:y2, x1:x2]

        h, w = plate.shape[:2]

        target_width = 320
        scale = target_width / w

        new_w = int(w * scale)
        new_h = int(h * scale)

        plate = cv2.resize(plate, (new_w, new_h))

        break
