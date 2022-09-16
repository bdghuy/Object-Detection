import tensorflow as tf
import keras
import random
import os
import cv2
import numpy as np
from pycocotools.coco import COCO 

from preprocessor import preprocess_data
from utils import convert_to_xywh, compute_iou
from anchors import AnchorBox


class LabelEncoder:
    """Transforms the raw labels into targets for training.

    This class has operations to generate targets for a batch of samples which
    is made up of the input images, bounding boxes for the objects present and
    their class ids.

    Attributes:
      anchor_box: Anchor box generator to encode the bounding boxes.
      box_variance: The scaling factors used to scale the bounding box targets.
    """

    def __init__(self):
        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )

    def _match_anchor_boxes(
        self, anchor_boxes, gt_boxes, match_iou=0.5, ignore_iou=0.4
    ):
        """Matches ground truth boxes to anchor boxes based on IOU.

        1. Calculates the pairwise IOU for the M `anchor_boxes` and N `gt_boxes`
          to get a `(M, N)` shaped matrix.
        2. The ground truth box with the maximum IOU in each row is assigned to
          the anchor box provided the IOU is greater than `match_iou`.
        3. If the maximum IOU in a row is less than `ignore_iou`, the anchor
          box is assigned with the background class.
        4. The remaining anchor boxes that do not have any class assigned are
          ignored during training.

        Arguments:
          anchor_boxes: A float tensor with the shape `(total_anchors, 4)`
            representing all the anchor boxes for a given input image shape,
            where each anchor box is of the format `[x, y, width, height]`.
          gt_boxes: A float tensor with shape `(num_objects, 4)` representing
            the ground truth boxes, where each box is of the format
            `[x, y, width, height]`.
          match_iou: A float value representing the minimum IOU threshold for
            determining if a ground truth box can be assigned to an anchor box.
          ignore_iou: A float value representing the IOU threshold under which
            an anchor box is assigned to the background class.

        Returns:
          matched_gt_idx: Index of the matched object
          positive_mask: A mask for anchor boxes that have been assigned ground
            truth boxes.
          ignore_mask: A mask for anchor boxes that need to by ignored during
            training
        """
        iou_matrix = compute_iou(anchor_boxes, gt_boxes)
        max_iou = tf.reduce_max(iou_matrix, axis=1)
        matched_gt_idx = tf.argmax(iou_matrix, axis=1)
        positive_mask = tf.greater_equal(max_iou, match_iou)
        negative_mask = tf.less(max_iou, ignore_iou)
        ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask))
        return (
            matched_gt_idx,
            tf.cast(positive_mask, dtype=tf.float32),
            tf.cast(ignore_mask, dtype=tf.float32),
        )

    def _compute_box_target(self, anchor_boxes, matched_gt_boxes):
        """Transforms the ground truth boxes into targets for training"""
        box_target = tf.concat(
            [
                (matched_gt_boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:],
                tf.math.log(matched_gt_boxes[:, 2:] / anchor_boxes[:, 2:]),
            ],
            axis=-1,
        )
        box_target = box_target / self._box_variance
        return box_target

    def _encode_sample(self, image_shape, gt_boxes, cls_ids):
        """Creates box and classification targets for a single sample"""
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        cls_ids = tf.cast(cls_ids, dtype=tf.float32)
        matched_gt_idx, positive_mask, ignore_mask = self._match_anchor_boxes(
            anchor_boxes, gt_boxes
        )
        matched_gt_boxes = tf.gather(gt_boxes, matched_gt_idx)
        box_target = self._compute_box_target(anchor_boxes, matched_gt_boxes)
        matched_gt_cls_ids = tf.gather(cls_ids, matched_gt_idx)
        cls_target = tf.where(
            tf.not_equal(positive_mask, 1.0), -1.0, matched_gt_cls_ids
        )
        cls_target = tf.where(tf.equal(ignore_mask, 1.0), -2.0, cls_target)
        cls_target = tf.expand_dims(cls_target, axis=-1)
        label = tf.concat([box_target, cls_target], axis=-1)
        return label

    def encode_batch(self, batch_images, gt_boxes, cls_ids):
        """Creates box and classification targets for a batch"""
        images_shape = tf.shape(batch_images)
        batch_size = images_shape[0]

        labels = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=True)
        for i in range(batch_size):
            label = self._encode_sample(images_shape, gt_boxes[i], cls_ids[i])
            labels = labels.write(i, label)

        return batch_images, labels.stack()


class DataGenerator(keras.utils.all_utils.Sequence):
    'Generates data for Keras'
    def __init__(self, root, annotation, training_mode=True, dim=512, batch_size=32, shuffle=True):
        'Initialization'
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.categories = {value['id']:value['name'] for value in self.coco.cats.values()}

        self.dim = dim
        self.training_mode = training_mode
        self.transforms = preprocess_data
        
        self.batch_size = batch_size
        self.LabelEncoder=LabelEncoder()

        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of ids
        ids_temp = [self.ids[k] for k in indexes]
        # Generate data
        batch_images, labels = self.__data_generation(ids_temp)

        return batch_images, labels

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, ids_temp):
        'Generates data containing batch_size samples' 
        # Initialization
        batch_images = [] 
        batch_gt_boxes = []
        batch_cls_ids = []

        # Own coco file
        coco = self.coco
        # Generate data
        for i, idx in enumerate(ids_temp):
          # Image ID
          img_id = self.ids[idx]
          # List: get annotation id from coco
          ann_ids = coco.getAnnIds(imgIds=img_id)
          # Dictionary: target coco_annotation file for an image
          coco_annotation = coco.loadAnns(ann_ids)
          # path for input image
          path = coco.loadImgs(img_id)[0]['file_name']
          # open the input image
          image = cv2.imread(os.path.join(self.root, path))
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          # number of objects in the image
          num_objs = len(coco_annotation)

          # Bounding boxes for objects
          # In coco format, bbox = [xmin, ymin, width, height]
          # The input should be [xmin, ymin, xmax, ymax]
          boxes = []
          cls_ids=[]
          for j in range(num_objs):
            cls_ids.append(coco_annotation[j]['category_id'])

            xmin = coco_annotation[j]['bbox'][0]
            ymin = coco_annotation[j]['bbox'][1]
            xmax = xmin + coco_annotation[j]['bbox'][2]
            ymax = ymin + coco_annotation[j]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
          
          # Image augmentation
          try:
            image, boxes, cls_ids = self.transforms(target_size=self.dim, training_mode=self.training_mode)(image, boxes, cls_ids)
          except:
            continue
            
          if len(boxes) == 0:
            continue
          boxes = convert_to_xywh(np.array(boxes))
    
          # Add to batch
          batch_images.append(image) 
          batch_gt_boxes.append(boxes)
          batch_cls_ids.append(cls_ids)


        #Encoding labels
        batch_images = np.array(batch_images)
        batch_images, labels = self.LabelEncoder.encode_batch(batch_images, batch_gt_boxes, batch_cls_ids)

        return batch_images, labels
