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

![Training loss curve](http://i.imgur.com/7ztVYwS.png)

![Test error curve](http://i.imgur.com/0NXOxLD.png)

| Paper                                 | Test Error | Reference (Tab. 6) |
|:-------------------------------------:|:--------:|:--------------------:|
| Nsize=3, 20 layers                    | 0.084473 | 0.875 |
| Nsize=5, 32 layers                    | 0.079102 | 0.751 |
| Nsize=7, 44 layers                    | 0.075684 | 0.717 |
| Nsize=9, 56 layers                    | 0.063477 | 0.697 |
| Nsize=18, 110 layers, fancy policy¹   | 0.067871 | 0.661² |

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
