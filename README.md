# CyCADA
This is unofficial implementation of CyCADA: Cycle-Consistent Adversarial Domain Adaptation (ICML2018).

## Requirements
```
python >= 3.6
pytorch>= 1.0
torchvision
```

## Setup dataset
I prepare a download code of MNIST->USPS dataset and run below.
```
python prepare_mnist2usps.py
```

If you conduct experiments on your dataset, please put data on the path: `../data/[your dataset]` and specify dataroot option in `scripts/train_cycada.sh` (default: dataroot=`../data/mnist_USPS`)

Dataset structure must be
```
- [your dataset]
    - trainA
    - trainB
    - testA
    - testB
```

Domain A is source and domain B is target, \
but if specifying `direction="BtoA"` in `scripts/train_cycada.sh`, switch source and target.

## Directory structure

- `data`: preprocess data and set loaders
- `options`: set options for train and test phase
- `results`: contain test results
- `util`: pack useful functions
- `checkpoints`: save training processes
- `models`: model implementation

## Pretraining

Pretrained models contain in `pretrain` \
If you pretrain a source classifier before adaptation, please specify `pretrain=1`.


## Train
If you conduct domain adaptation, please run below. All hyperparameters are packed.
```
./scripts/train_cycada.sh
```

## Test
This test code automatically searches unevaluated models in checkpoints.
```
./scripts/test.sh
```

## Model architecture
Generator: resnet-based networks with two residual blocks \
Discriminator: 4-layers \
Classifier: Revised LeNet for 32x32 images

## Result
The result can be reproduced by using pretrained mnist, usps classifiers I set in `pretrain`.
```
models/pretrain/lenet_mnist_acc_97.5000.pt
models/pretrain/lenet_usps_acc_97.1599.pt
```

|Model| Direction   | M-U  |
|-----|-----|-------|
|Source-only        | -> |91.68|
|Source-only        | <- |68.55|
|Cycada             | -> |96.0 (95.6)  |
|Cycada             | <- |95.0 (96.5)  |
|Target-only        | -> |97.15|
|Target-only        | <- |97.50|

() denotes reference values in the cycada paper


## Reference
- Paper \
CYCADA: CYCLE-CONSISTENT ADVERSARIAL DOMAIN ADAPTATION \
In ICML, 2018 \
https://arxiv.org/pdf/1711.03213.pdf

- Implementation \
Code is mainly borrowed from 
junyanz/pytorch-CycleGAN-and-pix2pix \
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix




