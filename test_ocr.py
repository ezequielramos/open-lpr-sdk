from fast_plate_ocr import LicensePlateRecognizer
import os
import cv2
import numpy as np

m = LicensePlateRecognizer('cct-xs-v1-global-model')
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK = len(CHARS)

def ocr_preprocess(img):
    h, w = img.shape[:2]

    target_w = 128
    target_h = 64

    scale = min(target_w / w, target_h / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h))

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2

    canvas[y:y+new_h, x:x+new_w] = resized

    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    canvas = canvas.astype(np.uint8)

    canvas = canvas[np.newaxis]   # [1,64,128,3]

    return canvas

def decode(output):
    plate = ""
    prev = BLANK
    confs = []

    for timestep in output:

        idx = int(np.argmax(timestep))
        conf = float(np.max(timestep))

        # ignore blanks
        if idx != BLANK:

            # só colapsa se o anterior for igual
            if idx != prev:
                plate += CHARS[idx]
                confs.append(conf)

        prev = idx

    confidence = np.mean(confs) if confs else 0

    return plate, confidence

# input_name = m.model.get_inputs()[0].name
# print(input_name)
# plate = cv2.imread('plates/FXL7E66.jpg')
# res = m.model.run(None, {input_name: ocr_preprocess(plate)})
# print(decode(res))

for image_path in os.listdir('plates'):
    print(image_path,m.run(f'plates/{image_path}'))