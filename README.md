# VGG_net

Implementation of VGG_net using TensorFlow. 
Testing with CIFAR10 dataset resized up to 224x224 pixels. Imagenet download started but expected to take substantial time (2 weeks or so)
VGG structure built in vgg.py, including optimizer.
Training graph set up in vgg_train.py
Evaluation is only implemented on the test set, in vgg_eval.py.
Load_data.py loads the CIFAR10 data into numpy arrays for testing.

