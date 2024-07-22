import cv2
import numpy as np
from blazeface import *
from cvs import *
import aidlite_gpu
import time

aidlite = aidlite_gpu.aidlite(1)
print('aidlite init successed')

def plot_detections(img, detections, with_keypoints=True):
    output_img = img
    print(img.shape)

    print("Found %d faces" % len(detections))
    for i in range(len(detections)):
        ymin = detections[i][0] * img.shape[0]
        xmin = detections[i][1] * img.shape[1]
        ymax = detections[i][2] * img.shape[0]
        xmax = detections[i][3] * img.shape[1]
        w = int(xmax - xmin)
        h = int(ymax - ymin)
        if w < h:
            xmin = xmin - (h - w) / 3.0
            xmax = xmax + (h - w) / 3.0
        else:
            ymin = ymin - (w - h) / 3.0
            ymax = ymax + (w - h) / 3.0

        p1 = (int(xmin), int(ymin))
        p2 = (int(xmax), int(ymax))
        print(p1, p2)
        cv2.rectangle(output_img, p1, p2, (0, 255, 255), 2, 1)
        cv2.putText(output_img, "Face found!", (p1[0] + 10, p2[1] - 10), cv2.FONT_ITALIC, 1, (0, 255, 129), 2)

        if with_keypoints:
            keypoints = []
            for k in range(6):
                kp_x = int(detections[i, 4 + k * 2] * img.shape[1])
                kp_y = int(detections[i, 4 + k * 2 + 1] * img.shape[0])
                cv2.circle(output_img, (kp_x, kp_y), 4, (0, 255, 255), 4)
                keypoints.append((kp_x, kp_y))

            # 进行疲劳检测
            fatigue_detected = detect_fatigue(keypoints)
            if fatigue_detected:
                cv2.putText(output_img, "Fatigue detected!", (p1[0] + 10, p2[1] - 40), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

    return output_img

def detect_fatigue(keypoints):
    left_eye = keypoints[0]
    right_eye = keypoints[1]
    mouth_left = keypoints[2]
    mouth_right = keypoints[3]
    nose = keypoints[4]
    mouth_center = keypoints[5]

    # 简单的疲劳检测逻辑，例如通过眼睛和嘴巴的状态进行判断
    eye_aspect_ratio = (right_eye[1] - left_eye[1]) / (right_eye[0] - left_eye[0])
    
    mouth_width = mouth_right[0] - mouth_left[0]
    if mouth_width == 0:
        return False

    mouth_aspect_ratio = (mouth_center[1] - nose[1]) / mouth_width

    eye_threshold = 0.05  # 眼睛闭合的阈值
    mouth_threshold = 0.2  # 嘴巴张开的阈值

    if eye_aspect_ratio < eye_threshold or mouth_aspect_ratio > mouth_threshold:
        return True
    return False

input_shape = [128, 128]
inShape = [1 * 128 * 128 * 3 * 4,]
outShape = [1 * 896 * 16 * 4, 1 * 896 * 1 * 4]
model_path = "models/face_detection_front.tflite"
print('gpu:', aidlite.FAST_ANNModel(model_path, inShape, outShape, 4, 0))

anchors = np.load('models/anchors.npy').astype(np.float32)

camid = 1
cap = cvs.VideoCapture(camid)
while True:
    frame = cap.read()
    if frame is None:
        continue
    if camid == 1:
        frame = cv2.flip(frame, 0)  # 镜像翻转图像

    img = cv2.resize(frame, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = img / 127.5 - 1.0

    aidlite.setTensor_Fp32(img, input_shape[1], input_shape[1])
    start_time = time.time()
    aidlite.invoke()

    t = (time.time() - start_time)
    lbs = 'Fps: ' + str(int(1 / t)) + " ~~ Time:" + str(t * 1000) + "ms"
    cvs.setLbs(lbs)

    raw_boxes = aidlite.getTensor_Fp32(0)
    classificators = aidlite.getTensor_Fp32(1)
    print(raw_boxes.shape, classificators.shape)

    detections = blazeface(raw_boxes, classificators, anchors)
    out = plot_detections(frame, detections[0])

    cvs.imshow(out)