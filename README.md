# Coding task

In this coding task, you'll have to build the inference pipeline for an image classification task on a well-known dataset. The dataset images are provided within this bundle, in the `/dataset` directory. The corresponding labels are serialized in `labels.json`.

### EfficientNet

The model whose performances we want to assess is EfficientNet (https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html). We recommend to load the trained model using this package: https://pypi.org/project/efficientnet-pytorch/ and to use the `b0` version of the model. This specific model is already trained on the `ImageNet` dataset, no model training is involved in this exercise.

### Inference, metrics and analyzis

Once the inference pipeline is working, we ask you to provide a critical and creative analysis of the inference results obtained (e.g. metrics, imbalance, ...).


# Solution

I've created a solution in two parts, first a notebook with naive implementation of the inference pipeline alongside with some metrics and results. Then, since I have limited ressources and the inference on the whole dataset is quite long, I have tried to implement another solution using dataloaders to save some computation time (this time directly in a python file).
Creating a virtual env and running the command `pip3 install -r requierments.txt` should be enough to run the code, i am using python 3.10.2