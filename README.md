Deep Residual Learning for Image Recognition
============================================

This is a Torch implementation of ["Deep Residual Learning for Image Recognition",Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun](http://arxiv.org/abs/1512.03385) the winners of the 2015 ILSVRC and COCO challenges.

**What's working:** CIFAR converges, as per the paper.

**What's not working yet:** Imagenet. I also have only implemented Option
(A) for the residual network bottleneck strategy.

How to use
----------
- You need at least CUDA 7.0 and CuDNN v4.
- Install Torch.
- Install the Torch CUDNN V4 library: `git clone https://github.com/soumith/cudnn.torch; cd cudnn; git co R4; luarocks make` This will give you `cudnn.SpatialBatchNormalization`, which helps save quite a lot of memory.
- Download
  [CIFAR 10](http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz).
  Use `--dataRoot <cifar>` to specify the location of the extracted CIFAR 10 folder.
- Run `train-cifar.lua`.

CIFAR: Effect of model size
---------------------------

For this test, our goal is to reproduce Figure 6 from the original paper:

![figure 6 from original paper](https://i.imgur.com/q3lcHic.png)

We train our model for 200 epochs (this is about 7.8e4 of their
iterations on the above graph). Like their paper, we start at a
learning rate of 0.1 and reduce it to 0.01 at 80 epochs and then to
0.01 at 160 epochs.

###Training loss
![Training loss curve](http://i.imgur.com/7ztVYwS.png)

###Testing error (rolling average)
![Test error curve](http://i.imgur.com/0NXOxLD.png)

| Model                                 | My Test Error | Reference Test Error from Tab. 6 |
|:-------------------------------------:|:--------:|:--------------------:|
| Nsize=3, 20 layers                    | 0.084473 | 0.875 |
| Nsize=5, 32 layers                    | 0.079102 | 0.751 |
| Nsize=7, 44 layers                    | 0.075684 | 0.717 |
| Nsize=9, 56 layers                    | 0.063477 | 0.697 |
| Nsize=18, 110 layers, fancy policy¹   | 0.067871 | 0.661² |

All of these results are very unstable, hence the rolling average. I really think I only have about one and a half significant figures. **The standard deviation of the test error between epochs 195 and 200 was frequently more than half of a percent,** so it probably isn't appropriate to rank-order the results. The right thing to do is to re-run the above experiments several times and take the mean result.

¹: For this run, we started from a learning rate of 0.001 until the
first 400 iterations. We then raised the learning rate to 0.1 and
trained as usual. This is consistent with the actual paper's results.

²: Note that the paper reports the best run from five runs, as well as
the mean. I consider the mean to be a valid test protocol, but I don't
like reporting the 'best' score because this is effectively training
on the test set. (This method of reporting effectively introduces an
extra parameter into the model--which model from an ensemble to
use--and this parameter is fitted to the test set)

CIFAR: Effect of model architecture
-----------------------------------

This experiment explores the effect of different NN architectures that
alter the "Building Block" model inside the residual network.

The original paper used a "Building Block" similar to the "Reference"
model on the left part of the figure below, with the standard
convolution layer, batch normalization, and ReLU, followed by another
convolution layer and batch normalization. The only interesting piece
of this architecture is that they move the ReLU after the addition.

We investigated two alternate strategies.

![Three different alternate CIFAR architectures](https://i.imgur.com/uRMBOaS.png)

- **Alternate 1: Move batch normalization after the addition.**
  (Middle) The reasoning behind this choice is to test whether
  normalizing the first term of the addition is desirable. It grew out
  of the mistaken belief that batch normalization always normalizes to
  have zero mean and unit variance. If this were true, building an
  identity building block would be impossible because the input to the
  addition always has unit variance. However, this is not true. BN
  layers have additional learnable scale and bias parameters, so the
  input to the batch normalization layer is not forced to have unit
  variance.

- **Alternate 2: Remove the second ReLU.** The idea behind this was
  noticing that in the reference architecture, the input cannot
  proceed to the output without being modified by a ReLU. This makes
  identity connections *technically* impossible because negative
  numbers would always be clipped as they passed through the skip
  layers of the network. To avoid this, we could either move the ReLU
  before the addition or remove it completely. However, it is not
  correct to move the ReLU before the addition: such an architecture
  would ensure that the output would never decrease because the first
  addition term could never be negative. The other option is to simply
  remove the ReLU completely, sacrificing the nonlinear property of
  this layer. It is unclear which approach is better.

To test these strategies, we repeat the above protocol using the
smallest (20-layer) residual network model.

(Note: The other experiments all use the leftmost "Reference" model.)

![Training loss](http://i.imgur.com/DyTIfIx.png)

![Testing error](http://i.imgur.com/79Vzxe8.png)

| Architecture  | Test error |
|:-------------:|:----------:|
| ReLU, BN after add                  | 0.083496 |
| No ReLU, BN before add              | 0.089355 |
| No ReLU, BN after add               | 0.079102 |
| ReLU, BN before add (ORIG PAPER)    | 0.084473 |

All methods achieve accuracies within about 2% of each other. Removing
ReLU and moving the batch normalization after the addition seems to
make a small improvement, but it's clear that the test error curve is
too noisy to rank order these methods.

Zooming in to the beginning of the loss curve reveals that the "ReLU,
BN after add" architecture begins converging slower than the other
ones, but it eventually achieves comparable performance.

![Training loss](http://i.imgur.com/XYiD6A8.png)


TODO: Alternate training strategies (RMSPROP, Adawhatever)
----------------------------------------------------------

TODO: Effect of preprocessing
-----------------------------
