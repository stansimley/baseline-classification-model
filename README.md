# Baseline Image Classification Tensorflow/Keras Model
*Implementation of fastest/smallest available keras model with pretrained weights and standard data augmentation*

## Abstract
Image Classification problems are complex where there is no one size fits all solution. However, a simple, general and fast solution may be adopted for developers to have a "baseline" comparison. This script provides that, using the smallest and arguably most efficient deep learning neural network on keras available with pretrained weights.

## Settings
* MobileNetV2
* alpha = 0.35 (scaling of width to original layers; 0.35 is the smallest available setting with pretrained weights from Imagenet)
* data augmentations: width and height shift = 0.05, rotation range = 45 (degrees), horizontal and vertical flips = True
* train/val split of 70/30
