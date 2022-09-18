import numpy as np
import argparse
import cv2
import os
from pycocotools.coco import COCO
from mapcalc import calculate_map, calculate_map_range

from detector_arch import infer_detector
from preprocessor import preprocess_data


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
parser.add_argument('--val_dir', type=str, required=True)

args = parser.parse_args()

# Build model with trained weights
model = infer_detector(R_input=args.R_input,
                       num_classes=args.num_classes,
                       W_bifpn=args.W_bifpn,
                       D_bifpn=args.D_bifpn,
                       D_head=args.D_head,
                       conf_threshold=args.conf_threshold,
                       iou_threshold=args.iou_threshold,
                       weights_path=args.weights)


annotation = os.path.join(args.val_dir, '_annotations.coco.json')
coco = COCO(annotation)
ids = list(sorted(coco.imgs.keys()))

mAP, AP50, AP75 = [], [], []

for idx in ids:
  # List: get annotation id from coco
  ann_ids = coco.getAnnIds(imgIds=idx)
  # Dictionary: target coco_annotation file for an image
  coco_annotation = coco.loadAnns(ann_ids)
  # path for input image
  path = coco.loadImgs(idx)[0]['file_name']
  # open the input image
  image = cv2.imread(os.path.join(args.val_dir, path))
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  # number of objects in the image
  num_objs = len(coco_annotation)

  # Bounding boxes for objects
  # In coco format, bbox = [xmin, ymin, width, height]
  # The input should be [xmin, ymin, xmax, ymax]
  boxes = []
  cls_ids=[]
  for j in range(num_objs):
    xmin = coco_annotation[j]['bbox'][0]
    ymin = coco_annotation[j]['bbox'][1]
    xmax = xmin + coco_annotation[j]['bbox'][2]
    ymax = ymin + coco_annotation[j]['bbox'][3]
    if xmin>=xmax or ymin>=ymax:
      continue
    boxes.append([xmin, ymin, xmax, ymax])
    cls_ids.append(coco_annotation[j]['category_id'])

  image, boxes, cls_ids = preprocess_data(args.R_input, training_mode=False)(image, boxes, cls_ids)

  ground_truth = {
    'boxes': boxes,
    'labels': cls_ids}

  detections = model.predict(np.expand_dims(image, axis=0))
  num_detections = detections.valid_detections[0]
  result_dict = {'boxes': detections.nmsed_boxes[0][:num_detections],
                'labels': detections.nmsed_classes[0][:num_detections],
                'scores':detections.nmsed_scores[0][:num_detections]}

  mAP.append(calculate_map_range(ground_truth, result_dict, 0.5, 0.95, 0.05))
  AP50.append(calculate_map(ground_truth, result_dict, 0.5))
  AP75.append(calculate_map(ground_truth, result_dict, 0.75))

print('_' * 34)
print('|' + 'mAP'.center(10) + '|' + 'AP@50'.center(10) + '|' + 'AP@75'.center(10) + '|')
print('-' * 34)
print('|' + '{:.2f}'.format(np.mean(mAP)*100).center(10) + '|' + '{:.2f}'.format(np.mean(AP50)*100).center(10) + '|' + '{:.2f}'.format(np.mean(AP75)*100).center(10) + '|')