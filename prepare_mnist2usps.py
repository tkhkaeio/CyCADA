import os
from PIL import Image
import numpy as np
import subprocess
import torchvision
os.makedirs("../data", exist_ok=True)
os.makedirs("../data/mnist_USPS", exist_ok=True)
root = "./data"
istrain = False
datasetA = torchvision.datasets.MNIST(root, train=istrain, transform=None, target_transform=None, download=True)
datasetB = torchvision.datasets.USPS(root, train=istrain, transform=None, target_transform=None, download=True)


for idx, (img, label) in enumerate(datasetA):
    os.makedirs(os.path.join(root, "mnist_USPS","testA", str(label)), exist_ok=True)
    img.save('{}/mnist_USPS/testA/{}/{:05d}.jpg'.format(root, label, idx))

for idx, (img, label) in enumerate(datasetB):
    os.makedirs(os.path.join(root, "mnist_USPS","testB", str(label)), exist_ok=True)
    img.save('{}/mnist_USPS/testB/{}/{:05d}.jpg'.format(root, label, idx))


istrain = True
datasetA = torchvision.datasets.MNIST(root, train=istrain, transform=None, target_transform=None, download=True)
dataset = torchvision.datasets.USPS(root, train=istrain, transform=None, target_transform=None, download=True)

for idx, (img, label) in enumerate(datasetA):
    os.makedirs(os.path.join(root, "mnist_USPS","trainA", str(label)), exist_ok=True)
    img.save('{}/mnist_USPS/trainA/{}/{:05d}.jpg'.format(root, label, idx))

for idx, (img, label) in enumerate(datasetB):
    os.makedirs(os.path.join(root, "mnist_USPS","trainB", str(label)), exist_ok=True)
    img.save('{}/mnist_USPS/trainB/{}/{:05d}.jpg'.format(root, label, idx))