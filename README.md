## Object-Detection

Implementation of custom detector with:
- Backbone: [EfficientNetV2 B0](https://arxiv.org/abs/2104.00298)
- Feature Pyramid Network: [biFPN](https://arxiv.org/abs/1911.09070)
- Weight Standardization: [paper](https://arxiv.org/abs/1903.10520)
- PolyLoss: [paper](https://arxiv.org/abs/2204.12511)

### Training

```
python main.py --train_dir ${train_folder}  --val_dir ${val_folder}
```

### Inference

```
python infer.py --num_classes ${num_classes} --img_path ${img_path}
```
