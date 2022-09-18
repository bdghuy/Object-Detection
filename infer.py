import tensorflow as tf
import numpy as np
import argparse
import cv2
import os

from detector_arch import infer_detector
from preprocessor import preprocess_data
from utils import draw_bboxes


parser = argparse.ArgumentParser()

# detector arch
parser.add_argument('--num_classes', default=80, type=int)
parser.add_argument('--R_input', default=512, type=int)
parser.add_argument('--W_bifpn', default=64, type=int)
parser.add_argument('--D_bifpn', default=1, type=int)
parser.add_argument('--D_head', default=3, type=int)

# weights
parser.add_argument('--weights', default='checkpoint/weights.h5', type=str)

# nms
parser.add_argument('--conf_threshold', default=0.5, type=float)
parser.add_argument('--iou_threshold', default=0.5, type=float)

# --------
parser.add_argument('--img_path', type=str, required=True)
parser.add_argument('--result', default='result', type=str)

args = parser.parse_args()


model = infer_detector(R_input=args.R_input,
                       num_classes=args.num_classes,
                       W_bifpn=args.W_bifpn,
                       D_bifpn=args.D_bifpn,
                       D_head=args.D_head,
                       conf_threshold=args.conf_threshold,
                       iou_threshold=args.iou_threshold,
                       weights_path=args.weights)


test_img = cv2.imread(args.img_path)
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
test_img = preprocess_data(args.R_input, training_mode=False)(test_img, [], [])[0]
test_img = np.expand_dims(test_img, axis=0)

detections = model.predict(test_img)
num_detections = detections.valid_detections[0]

test_img = draw_bboxes(test_img[0],
                       detections.nmsed_boxes[0][:num_detections],
                       detections.nmsed_classes[0][:num_detections],
                       detections.nmsed_scores[0][:num_detections])

os.makedirs(args.result, exist_ok=True)
cv2.imwrite(os.path.join(args.result, os.path.basename(args.img_path)), test_img)
