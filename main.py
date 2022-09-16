import argparse
import os
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow_addons.optimizers import SGDW

from detector_arch import detector
from dataloader import DataGenerator
from loss import Loss
from utils import WarmUpCosineDecayScheduler

parser = argparse.ArgumentParser()

# data directory
parser.add_argument('--train_dir', required=True)
parser.add_argument('--val_dir', required=True)

# training
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--R_input', default=512, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--lr_base', default=0.016, type=float)
parser.add_argument('--warmup_epochs', default=2., type=float)

# detector arch
parser.add_argument('--W_bifpn', default=64, type=int)
parser.add_argument('--D_bifpn', default=1, type=int)
parser.add_argument('--D_head', default=3, type=int)

# checkpoint
parser.add_argument('--checkpoint', default='checkpoint', type=str)

args = parser.parse_args()
os.makedirs(args.checkpoint, exist_ok=True)


# Data generator
train_datagen = DataGenerator(root=args.train_dir, 
                           annotation=os.path.join(args.train_dir, '_annotations.coco.json'), 
                           batch_size=args.batch_size)

val_datagen = DataGenerator(root=args.val_dir, 
                           annotation=os.path.join(args.val_dir, '_annotations.coco.json'),
                           training_mode=False, 
                           batch_size=args.batch_size,
                           shuffle=False)

categories = train_datagen.categories
num_classes = len(categories)


warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=args.lr_base,
                                        total_steps=args.epochs*len(train_datagen),
                                        warmup_learning_rate=0.0,
                                        warmup_steps=args.warmup_epochs*len(train_datagen),
                                        hold_base_rate_steps=0)

model_checkpoint_callback = ModelCheckpoint(filepath=os.path.join(args.checkpoint, 'weights.h5'),
                                            save_weights_only=True,
                                            monitor='val_loss',
                                            mode='min',
                                            save_best_only=True,
                                            verbose=1)

model = detector(R_input=args.R_input,
                 num_classes=num_classes,
                 W_bifpn=args.W_bifpn,
                 D_bifpn=args.D_bifpn,
                 D_head=args.D_head)
optimizer = SGDW(momentum = 0.9,
                 weight_decay = 1e-4,
                 nesterov = True)

model.compile(loss=Loss(num_classes),
              optimizer=optimizer)

model.fit(train_datagen,
          epochs=args.epochs,
          validation_data=val_datagen,
          verbose=1,
          callbacks=[warm_up_lr,
                     model_checkpoint_callback])

