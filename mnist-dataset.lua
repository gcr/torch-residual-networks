require 'dp'
require 'hdf5'
path = require 'pl.path'
gm = require 'graphicsmagick'

Dataset = {}
local MNIST, parent = torch.class("Dataset.MNIST")

function MNIST:__init(hdf5path, mode)
   self.f = hdf5.open(hdf5path)
   local all = self.f:all()
   self.label = all['label_'..mode]:float():add(1)
   self.data = all['data_'..mode]:view(-1, 1,28,28):float()
end

function MNIST:preprocess(mean, std)
   mean = mean or self.data:mean(1)
   std = std or self.data:std() -- Complete std!
   self.data:add(-mean:expandAs(self.data)):mul(1/std)
   return mean,std
end

function MNIST:sampleIndices(batch, indices)
   if not indices then
      indices = batch
      batch = nil
   end
   local n = indices:size()[1]
   batch = batch or {}
   batch.inputs = self.data:index(1, indices)
   batch.outputs = self.label:index(1, indices)
   if self.use_cuda then
      batch.inputs:cuda()
      batch.outputs:cuda()
   end
   return batch
end

function MNIST:sample(batch, batch_size)
   if not batch_size then
      batch_size = batch
      batch = nil
   end
   return self:sampleIndices(
      batch,
      (torch.rand(batch_size) * self:size()):long():add(1)
   )
end

function MNIST:size()
   return self.data:size(1)
end

function MNIST:cuda()
   self.use_cuda = true
end
