## Object-Detection

[<img src='https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667' width=100 class="left">]

Implementation of custom detector with:
- Backbone: [EfficientNetV2 B0](https://arxiv.org/abs/2104.00298)
- Feature Pyramid Network: [biFPN](https://arxiv.org/abs/1911.09070)
- Weight Standardization: [paper](https://arxiv.org/abs/1903.10520)
- PolyLoss: [paper](https://arxiv.org/abs/2204.12511)

### Augmentation
- Large Scale Jittering only: An image will be resized and cropped with a resize range of 0.1 to 2.0 of the original image size.

<img src="https://github.com/bdghuy/Object-Detection/blob/main/images/LSJ.PNG" width="319" height="212">


### Set up your dataset
```
train
 |-img_1.jpg
 |-img_2.jpg
 | ...
 |-_annotations.coco.json
val
 |-img_1.jpg
 |-img_2.jpg
 | ...
 |-_annotations.coco.json
```
### Training

```
python main.py --train_dir ${train_folder}  --val_dir ${val_folder}
```

### Inference

```
python infer.py --num_classes ${num_classes} --img_path ${img_path}
```

### Experiment

<img src="https://github.com/bdghuy/Object-Detection/blob/main/images/img.PNG" width="359" height="113">

