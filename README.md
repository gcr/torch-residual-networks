Deep Residual Learning for Image Recognition
============================================

This is a Torch implementation of ["Deep Residual Learning for Image Recognition",Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun](http://arxiv.org/abs/1512.03385) the winners of the 2015 ILSVRC and COCO challenges.

**What's working:** CIFAR converges, as per the paper.

**What's not working yet:** Imagenet. I also have only implemented Option
(A) for the residual network bottleneck strategy.

Changes
-------
- 2016-01-15:
  - **New CIFAR results**: I re-ran all the CIFAR experiments and
  updated the results. There were a few bugs: we were only testing on
  the first 2,000 images in the training set, and they were sampled
  with replacement. These new results are much more stable over time.
- 2016-01-12: Release results of CIFAR experiments.

How to use
----------
- You need at least CUDA 7.0 and CuDNN v4.
- Install Torch.
- Install the Torch CUDNN V4 library: `git clone https://github.com/soumith/cudnn.torch; cd cudnn; git co R4; luarocks make` This will give you `cudnn.SpatialBatchNormalization`, which helps save quite a lot of memory.
- Download
  [CIFAR 10](http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz).
  The code expects the files to be located in
  `/mnt/cifar/data_batch_*.t7`
- Comment out all the `workbook` calls. You should replace them with
  your own reporting and model saving code.
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
![Training loss curve](http://i.imgur.com/XqKnNX1.png)

###Testing error
![Test error curve](http://i.imgur.com/lt2D5cA.png)

| Model                                 | My Test Error | Reference Test Error from Tab. 6 |
|:-------------------------------------:|:--------:|:--------------------:|
| Nsize=3, 20 layers                    | 0.0829 | 0.0875 |
| Nsize=5, 32 layers                    | 0.0763 | 0.0751 |
| Nsize=7, 44 layers                    | 0.0714 | 0.0717 |
| Nsize=9, 56 layers                    | 0.0694 | 0.0697 |
| Nsize=18, 110 layers, fancy policy¹   | 0.0673 | 0.0661² |

We can reproduce the results from the paper to typically within 0.5%.
In all cases except for the 32-layer network, we achieve very slightly
improved performance, though this may just be noise.

¹: For this run, we started from a learning rate of 0.001 until the
first 400 iterations. We then raised the learning rate to 0.1 and
trained as usual. This is consistent with the actual paper's results.

²: Note that the paper reports the best run from five runs, as well as
the mean. I consider the mean to be a valid test protocol, but I don't
like reporting the 'best' score because this is effectively training
on the test set. (This method of reporting effectively introduces an
extra parameter into the model--which model to use from the
ensemble--and this parameter is fitted to the test set)

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

![Training loss](http://i.imgur.com/qDDLZLQ.png)

![Testing error](http://i.imgur.com/fTY6TL5.png)

| Architecture                        | Test error |
|:-----------------------------------:|:----------:|
| ReLU, BN before add (ORIG PAPER)    | 0.0829 |
| No ReLU, BN before add              | 0.0862 |
| ReLU, BN after add                  | 0.0834 |
| No ReLU, BN after add               | 0.0823 |

All methods achieve accuracies within about 0.5% of each other.
Removing ReLU and moving the batch normalization after the addition
seems to make a small improvement on CIFAR, but there is too much
noise in the test error curve to reliably tell a difference.

TODO: Alternate training strategies (RMSPROP, Adawhatever)
----------------------------------------------------------

TODO: Effect of preprocessing
-----------------------------

TODO: Effect of batch norm momentum
-----------------------------------
