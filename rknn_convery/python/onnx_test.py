import cv2
import numpy as np
import onnxruntime as ort
import time

IMG_SIZE = 640
OBJ_THRESH = 0.25
NMS_THRESH = 0.45

# ONNX 模型路径
ONNX_MODEL_PATH = '../exp65/weights/best.onnx'

# 类别
CLASSES = ['JDB_can', 'baishi_black', 'baishi_blue', 'cp_ml', 'cp_nm', 'cp_peach',
           'cp_yz', 'dp_bottle', 'dp_can', 'kaishui', 'redbull_bottle', 'redbull_can',
           'rio_gp', 'rio_lizhi', 'rio_orange', 'rio_peach', 'rio_sb', 'rusuanjun',
           'wanglaoji', 'wangzai', 'yangledu', 'yingyang_apple', 'yingyang_purple',
           'yingyang_white', 'yingyang_zao']

# 预处理（letterbox缩放）
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, dw, dh

# 非极大值抑制
def non_max_suppression(prediction, conf_thresh=0.25, iou_thresh=0.45):
    boxes = []
    confidences = []
    class_ids = []

    for det in prediction:
        conf = det[4]
        if conf < conf_thresh:
            continue
        class_scores = det[5:]
        class_id = np.argmax(class_scores)
        score = class_scores[class_id] * conf
        if score < conf_thresh:
            continue
        x, y, w, h = det[0:4]
        x1 = x - w / 2
        y1 = y - h / 2
        boxes.append([x1, y1, w, h])
        confidences.append(float(score))
        class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, iou_thresh)
    results = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        x1, y1, w, h = box
        results.append((int(x1), int(y1), int(w), int(h), confidences[i], class_ids[i]))
    return results

# 加载ONNX模型
session = ort.InferenceSession(ONNX_MODEL_PATH)
input_name = session.get_inputs()[0].name

# 打开摄像头
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized, r, dw, dh = letterbox(img, (IMG_SIZE, IMG_SIZE))
    img_input = img_resized.astype(np.float32) / 255.0
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0)

    outputs = session.run(None, {input_name: img_input})[0]
    outputs = np.squeeze(outputs)

    results = non_max_suppression(outputs, OBJ_THRESH, NMS_THRESH)

    for x, y, w, h, conf, cls_id in results:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f'{CLASSES[cls_id]} {conf:.2f}'
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("ONNX Camera Inference", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
