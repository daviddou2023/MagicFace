import cv2
import numpy as np
from blazeface import *
from cvs import *
import aidlite_gpu
import time
import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

aidlite = aidlite_gpu.aidlite(1)
print('aidlite init successed')

# 定义一个虚拟的BlazeFace模型类
class BlazeFaceModel(nn.Module):
    def __init__(self):
        super(BlazeFaceModel, self).__init__()

    def forward(self, x):
        # 这个虚拟模型不做任何操作。如果需要，可以替换为实际的BlazeFace逻辑。
        return x

# 实例化模型并将其转换为TorchScript
model = BlazeFaceModel()
example_input = torch.randn(1, 128, 128, 3)  # 示例输入张量
traced_model = torch.jit.trace(model, example_input)

# 优化模型以供移动设备使用
optimized_traced_model = optimize_for_mobile(traced_model)

# 将优化后的模型保存到文件
optimized_traced_model._save_for_lite_interpreter("models/testface_model.pt")

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
            for k in range(6):
                kp_x = int(detections[i, 4 + k * 2] * img.shape[1])
                kp_y = int(detections[i, 4 + k * 2 + 1] * img.shape[0])
                cv2.circle(output_img, (kp_x, kp_y), 4, (0, 255, 255), 4)

    return output_img

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