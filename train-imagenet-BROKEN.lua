--[[
Copyright (c) 2016 Michael Wilber

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgement in the product documentation would be
   appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
--]]

require 'residual-layers'
require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'train-helpers'
display = require 'display'
workbook = (require'lab-workbook-for-trello'):newExperiment{}

opt = lapp[[
      --batchSize       (default 64)      Sub-batch size
      --iterSize        (default 1)       How many sub-batches in each batch
      --nThreads        (default 4)       Data loader threads
      --dataTrainRoot   (default /mnt/imagenet/train)   Data root folder
      --dataValRoot     (default /mnt/imagenet/val)   Data root folder
      --verbose         (default true)
      --loadSize        (default 256)     Size of image when loading
      --fineSize        (default 224)     Size of image crop
      --loadFrom        (default "")      Model to load
]]
print(opt)

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
dataTrain = DataLoader.new(opt.nThreads, 'folder', {dataRoot = opt.dataTrainRoot,
                                                    fineSize = opt.fineSize,
                                                    loadSize = opt.loadSize,
                                                    batchSize = opt.batchSize,
                                                })
dataVal = DataLoader.new(opt.nThreads, 'folder', {dataRoot = opt.dataValRoot,
                                                  fineSize = opt.fineSize,
                                                  loadSize = opt.loadSize,
                                                  batchSize = opt.batchSize,
                                              })
print("Dataset size: ", dataTrain:size())


-- dataset_train = Dataset.MNIST("mnist.hdf5", 'train')
-- dataset_test = Dataset.MNIST("mnist.hdf5", 'test')

-- mean,std = dataset_train:preprocess()
-- dataset_test:preprocess(mean,std)

-- -- LENET here
-- -- model = nn.Sequential()
-- -- -- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
-- -- model:add(nn.SpatialConvolutionMM(1, 32, 5, 5, 1,1))
-- -- model:add(nn.ReLU())
-- -- model:add(nn.SpatialMaxPooling(2,2, 2,2))
-- -- -- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
-- -- model:add(nn.SpatialConvolutionMM(32, 64, 5, 5, 1,1))
-- -- model:add(nn.ReLU())
-- -- model:add(nn.SpatialMaxPooling(2, 2, 2,2))
-- -- -- stage 3 : standard 2-layer MLP:
-- -- model:add(nn.Reshape(64*4*4))
-- -- model:add(nn.Linear(64*4*4, 10))
-- -- -- model:add(nn.ReLU())
-- -- -- model:add(nn.Linear(200, 10))
-- -- model:add(nn.LogSoftMax())

-- Residual network.
-- Input: 3x224x224
if opt.loadFrom == "" then
    input = nn.Identity()()
    ------> 64, 112,112
    model = cudnn.SpatialConvolution(3, 64, 7,7, 2,2, 3,3)(input)
    --model = nn.SpatialBatchNormalization(64)(model)
    model = cudnn.ReLU(true)(model)
    model = cudnn.SpatialMaxPooling(3,3,  2,2,  1,1)(model)
    ------> 64, 56,56
    model = addResidualLayer2(model, 64)
    model = addResidualLayer2(model, 64)
    model = addResidualLayer2(model, 64)
    ------> 128, 28,28
    model = addResidualLayer2(model, 64, 128, 2)
    model = addResidualLayer2(model, 128)
    model = addResidualLayer2(model, 128)
    model = addResidualLayer2(model, 128)
    ------> 256, 14,14
    model = addResidualLayer2(model, 128, 256, 2)
    model = addResidualLayer2(model, 256)
    model = addResidualLayer2(model, 256)
    model = addResidualLayer2(model, 256)
    model = addResidualLayer2(model, 256)
    model = addResidualLayer2(model, 256)
    ------> 512, 7,7
    model = addResidualLayer2(model, 256, 512, 2)
    model = addResidualLayer2(model, 512)
    model = addResidualLayer2(model, 512)
    ------> 1000, 1,1
    model = cudnn.ReLU(true)(cudnn.SpatialConvolution(512, 1000, 7,7)(model))
    ------> 1000
    model = nn.Reshape(1000)(model)
    model = nn.LogSoftMax()(model)

    model = nn.gModule({input}, {model})
    model:cuda()
else
    print("Loading model from "..opt.loadFrom)
    cutorch.setDevice(1)
    model = torch.load(opt.loadFrom)
    print "Done"
end

loss = nn.ClassNLLCriterion()
loss:cuda()

-- Convert model to multi-GPU model :)
----[[
dmodel = nn.DataParallelTable(1)
dmodel:add(model, 1)
for i=2, 4 do
    cutorch.setDevice(i)
    dmodel:add(model:clone(), i)
end
cutorch.setDevice(1)
model = dmodel
--]]
-- required to fix call to SyncParameters
model:forward(torch.randn(8, 3, 224,224):cuda())

-- Dirty trick: make the first conv layer weights easier to modify
-- model.modules[2].weight:mul(0.5)

--[[
model:apply(function(m)
    -- Initialize weights
    local name = torch.type(m)
    if name:find('Convolution') then
        m.weight:normal(0.0, math.sqrt(2/(m.nInputPlane*m.kW*m.kH)))
        print(m.weight:std())
        m.bias:fill(0)
    elseif name:find('BatchNormalization') then
        if m.weight then m.weight:normal(1.0, 0.002) end
        if m.bias then m.bias:fill(0) end
    end
end)
--]]



sgdState = {
   --- For SGD with momentum ---
   ----[[
   -- My semi-working settings
   learningRate = 0.001,
   weightDecay    = 1e-4,
   -- Settings from their paper
   --learningRate = 0.1,
   --weightDecay    = 1e-4,

   momentum     = 0.9,
   dampening    = 0,
   nesterov     = true,
   --]]
   --- For rmsprop, which is very fiddly and I don't trust it at all ---
   --[[
   learningRate = 1e-5,
   alpha = 0.9,
   whichOptimMethod = 'rmsprop',
   --]]
   --- For adadelta, which sucks ---
   --[[
   rho              = 0.3,
   whichOptimMethod = 'adadelta',
   --]]
   --- For adagrad, which also sucks ---
   --[[
   learningRate = 3e-4,
   whichOptimMethod = 'adagrad',
   --]]
   --- For adam, which also sucks ---
   --[[
   learningRate = 0.005,
   whichOptimMethod = 'adam',
   --]]
   --- For the alternate implementation of NAG ---
   --[[
   learningRate = 0.01,
   weightDecay = 1e-6,
   momentum = 0.9,
   whichOptimMethod = 'nag',
   --]]
}


if opt.loadFrom ~= "" then
    print("Trying to load sgdState from "..string.gsub(opt.loadFrom, "model", "sgdState"))
    collectgarbage(); collectgarbage(); collectgarbage()
    sgdState = torch.load(""..string.gsub(opt.loadFrom, "model", "sgdState"))
    collectgarbage(); collectgarbage(); collectgarbage()
    print("Got", sgdState.nSampledImages,"images")
end

-- Actual Training! -----------------------------
weights, gradients = model:getParameters()
function forwardBackwardBatch(batch)
    -- After every batch, the different GPUs all have different gradients
    -- (because they saw different data), and only the first GPU's weights were
    -- actually updated.
    -- We have to do two changes:
    --   - Copy the new parameters from GPU #1 to the rest of them;
    --   - Zero the gradient parameters so we can accumulate them again.
   model:training()
   model:syncParameters() -- This copies parameters from first GPU to
   model:zeroGradParameters()

   --[[
   if sgdState.nSampledImages < 10000 then
       sgdState.learningRate = 0.001
   else
       sgdState.learningRate = 0.01
   end
   --]]

   local loss_val = 0
   local N = opt.iterSize
   local inputs, labels
   for i=1,N do
       inputs, labels = dataTrain:getBatch()
       inputs = inputs:cuda()
       labels = labels:cuda()
       collectgarbage(); collectgarbage();
       local y = model:forward(inputs)
       loss_val = loss_val + loss:forward(y, labels)
       local df_dw = loss:backward(y, labels)
       model:backward(inputs, df_dw)
       -- The above call will accumulate all GPUs' parameters onto GPU #1
   end
   loss_val = loss_val / N
   gradients:mul( 1.0 / N )

   if sgdState.nEvalCounter % 20 == 0 then
      display.image(model.modules[1].modules[2].weight, {win=24, title="First layer weights"})
   end

   return loss_val, gradients, inputs:size(1) * N
end


function evalModel()
   print("No evaluation...")
   -- if sgdState.epochCounter > 10 then os.exit(1) end
   local results = evaluateModel(model, dataVal)
   print(results)
   --table.insert(sgdState.accuracies, acc)
end

--[[
local results = evaluateModel(model, dataVal)
print(results)
--]]

--[[
require 'graph'
graph.dot(model.fg, 'MLP', '/tmp/MLP')
os.execute('convert /tmp/MLP.svg /tmp/MLP.png')
display.image(image.load('/tmp/MLP.png'), {title="Network Structure", win=23})
--]]

--[[
require 'ncdu-model-explore'
local y = model:forward(torch.randn(opt.batchSize, 3, 224,224):cuda())
local df_dw = loss:backward(y, torch.zeros(opt.batchSize):cuda())
model:backward(torch.randn(opt.batchSize,3,224,224):cuda(), df_dw)
exploreNcdu(model)
--]]


-- --[[
TrainingHelpers.trainForever(
   model.modules[1],
   forwardBackwardBatch,
   weights,
   sgdState,
   dataTrain:size(),
   evalModel,
   "snapshots/imagenet-residual-experiment2"
)
--]]
