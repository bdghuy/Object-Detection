import albumentations as A
import tensorflow as tf 
import random
import numpy as np 

def preprocess_data(target_size=512, training_mode=True):
  if training_mode:
    s = random.uniform(0.1, 2.0)
    p = 0.5
  else:
    s = 1
    p = 0
  
  def input(image, bboxes, class_id):
    #scale and flip
    scale_and_flip = A.Compose([
                                A.LongestMaxSize (max_size=int(s*target_size)),
                                A.HorizontalFlip(p=p)
                               ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_id']))
    
    transformed = scale_and_flip(image=image, bboxes=bboxes, class_id=class_id)
    image = transformed['image']
    bboxes = transformed['bboxes']
    class_id = transformed['class_id']
    #crop
    if s > 1:
      crop_dims=np.minimum(list(image.shape[:2]),[target_size, target_size])
      crop = A.Compose([
                        A.RandomCrop (crop_dims[0], crop_dims[1])
                       ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.2, label_fields=['class_id']))

      transformed = crop(image=image, bboxes=bboxes, class_id=class_id)
      image = transformed['image']
      bboxes = transformed['bboxes']
      class_id = transformed['class_id']
    #padding
    image=tf.image.pad_to_bounding_box(image, 0, 0, target_size, target_size)

    return image.numpy(), np.array(bboxes, dtype=np.float32), class_id
  
  return input