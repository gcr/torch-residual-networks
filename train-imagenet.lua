require 'residual-layers'
require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'train-helpers'

opt = lapp[[
      --batchSize       (default 32)      Batch size
      --nThreads        (default 4)       Data loader threads
      --dataRoot        (default /mnt/imagenet/train)   Data root folder
      --dataset_name    (default 'folder')
      --verbose         (default true)
      --loadSize        (default 256)     Size of image when loading
      --fineSize        (default 224)     Size of image crop
]]
print(opt)

-- create data loader
local DataLoader = paths.dofile('data.lua')
data = DataLoader.new(opt.nThreads, 'folder', opt)
print("Dataset size: ", data:size())


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
input = nn.Identity()()
model = nn.SpatialConvolution(3, 64, 7,7, 2,2, 3,3)(input)
------> 64, 112,112
model = nn.ReLU(true)(model)
model = nn.SpatialMaxPooling(3,3,  2,2,  1,1)(model)
------> 64, 56,56
model = addResidualLayer2(model, 64)
model = nn.SpatialBatchNormalization(64)(model)
model = addResidualLayer2(model, 64)
model = nn.SpatialBatchNormalization(64)(model)
model = nn.ReLU(true)(nn.SpatialConvolution(64, 128, 3,3, 2,2, 1,1)(model))
------> 128, 28,28
model = addResidualLayer2(model, 128)
model = nn.SpatialBatchNormalization(128)(model)
model = addResidualLayer2(model, 128)
model = nn.SpatialBatchNormalization(128)(model)
model = nn.ReLU(true)(nn.SpatialConvolution(128, 256, 3,3, 2,2, 1,1)(model))
------> 256, 14,14
model = addResidualLayer2(model, 256)
model = nn.SpatialBatchNormalization(256)(model)
model = addResidualLayer2(model, 256)
model = nn.SpatialBatchNormalization(256)(model)
model = nn.ReLU(true)(nn.SpatialConvolution(256, 512, 3,3, 2,2, 1,1)(model))
------> 512, 7,7
model = addResidualLayer2(model, 512)
model = nn.SpatialBatchNormalization(512)(model)
model = addResidualLayer2(model, 512)
model = nn.SpatialBatchNormalization(512)(model)
model = nn.ReLU(true)(nn.SpatialConvolution(512, 1000, 7,7)(model))
------> 1000, 1,1
model = nn.Reshape(1000)(model)
------> 1000
model = nn.LogSoftMax()(model)

model = nn.gModule({input}, {model})

loss = nn.ClassNLLCriterion()
model:cuda()
loss:cuda()


print(#model:forward(torch.randn(10, 3, 224,224)))
inspectMemory(model)
mem_usage = {}
for i,module in ipairs(model.modules) do
   accumMemoryByFieldName(module, mem_usage)
end
print(mem_usage)


-- sgdState = {
--    learningRate = 0.01,
--    --momentum     = 0.9,
--    --dampening    = 0,
--    --weightDecay  = 0.0005,
--    --nesterov     = true,
--    whichOptimMethod = 'rmsprop',
--    epochDropCount = 20,
--
--    -- Train stuff
--    options = opt,
--    accuracies = {},
-- }
--
-- -- Actual Training! -----------------------------
-- weights, gradients = model:getParameters()
-- function forwardBackwardBatch(batch)
--    model:training()
--    gradients:zero()
--    local inputs = batch[1]
--    local labels = batch[2]
--    local y = model:forward(inputs)
--    local loss_val = loss:forward(y, labels)
--    local df_dw = loss:backward(y, labels)
--    model:backward(inputs, df_dw)
--    return loss_val, gradients
-- end
--
--
-- function evalModel()
--    print("No evaluation...")
--    -- if sgdState.epochCounter > 10 then os.exit(1) end
--    -- model:evaluate()
--    -- local batch = dataset_test:sample(10000)
--    -- local output = model:forward(batch.inputs)
--    -- local _, indices = torch.sort(output, 2, true)
--    -- -- indices has shape (batchSize, nClasses)
--    -- local top1 = indices:select(2, 1)
--    -- local acc = (top1:eq(batch.outputs:long()):sum() / top1:size(1))
--    -- print("\n\nAccuracy: ", acc)
--    -- table.insert(sgdState.accuracies, acc)
-- end
--
--
-- TrainingHelpers.trainForever(
--    model,
--    forwardBackwardBatch,
--    weights,
--    sgdState,
--    function()
--       inputs,labels = data:getBatch()
--       return {inputs,labels}
--    end,
--    dataset_train:size(),
--    evalModel,
--    "snapshots/residual-mnist"
-- )
