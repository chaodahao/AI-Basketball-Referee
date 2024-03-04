"""
Author: Chen Weilin
Date: March 4, 2024
Reference: https://github.com/ultralytics/yolov5/blob/master/detect.py
Description: This file implements part of the YOLOv8-compatible interfaces for YOLOv5
"""

import os
import sys
import copy
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn

# FILE = Path(__file__).resolve()

# ROOT = FILE.parents[0]  # YOLOv5 root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from yolov5_utils.general import (
    check_img_size,
    non_max_suppression,
    scale_boxes,
    cv2
)
from yolov5_utils.torch_utils import select_device
from yolov5_utils.augmentations import letterbox

class YOLOV5Box:
    def __init__(self,pred) -> None:
        self.pred = pred[0] # only for single image
        self.xyxy = self.pred[:,:4]
        self.cls = self.pred[:,4]
        self.conf = self.pred[:,5]
        
class YOLOV5Result:
    def __init__(self, img, pred, names = ["basketball"]) -> None:
        self.img = img
        self.pred = pred
        self.boxes = YOLOV5Box(pred)
        self.names = names
        # self.annotator = Annotator(img, line_width=3, example=str(names))

    def plot(self):
        hide_conf = False
        hide_labels = False
        names = self.names
        annotator = Annotator(self.img, line_width=3, example=str(names))
        for *xyxy, conf, cls in reversed(self.pred[0]):
            c = int(cls)  # integer class
            label = names[c] if hide_conf else f"{names[c]}"
            confidence = float(conf)
            confidence_str = f"{confidence:.2f}"
            
            c = int(cls)  # integer class
            label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
            annotator.box_label(xyxy, label, color=colors(c, True))

        return annotator.result()

class YOLOV5(nn.Module):
    def __init__(self, 
                 weights = "./checkpoint/yolov5s_basketball.pt",
                 device = "",
                 conf_thres = 0.55, 
                 iou_thres = 0.45,
                 max_det = 1000,
                 ):
        super().__init__()
        self.device = select_device(device)
        self.half = False
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det

        # Load model
        self.model = DetectMultiBackend(weights, device=self.device, dnn=False, fp16=self.half).eval()
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size([640,640], s=self.stride)  # check image size
        
    
    def forward(self, img, verbose=False, conf=0.65):
        with torch.no_grad():
            img0 = copy.deepcopy(img)
            img = letterbox(img, stride=self.stride, auto=False)[0]
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0

            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = self.model(img, augment=False, visualize=False)[0]
            
            # Apply NMS
            # list[tensor([], device='cuda:0', size=(0, 6))]
            pred = non_max_suppression(pred, 
                                    self.conf_thres, 
                                    self.iou_thres, 
                                    classes=None, 
                                    agnostic=False, 
                                    max_det=self.max_det
                                    )
            
            for det in pred:
                if(len(det)):
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()

        return [YOLOV5Result(img0, pred)]


if __name__ == "__main__":
    x = YOLOV5(conf_thres=0.65)
    img = cv2.imread("../../kun_images/100.jpg")
    y = x(img)
    print(y[0].boxes.xyxy)

    img = y[0].plot()
    cv2.imshow("x", img)
    cv2.waitKey(0)
    