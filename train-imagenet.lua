require 'residual-layers'
require 'nn'
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

-- Residual network
input = nn.Identity()()
model = addResidualLayer(input, 1, 8, 8)
model = nn.SpatialBatchNormalization(8)(model)
model = addResidualLayer(model, 8, 4, 8)
model = nn.SpatialBatchNormalization(8)(model)
model = addResidualLayer(model, 8, 4, 8)
model = nn.SpatialBatchNormalization(8)(model)
model = addResidualLayer(model, 8, 16, 32)
model = nn.SpatialBatchNormalization(32)(model)
-- model = addResidualLayer(model, 8, 4, 8)
-- model = addResidualLayer(model, 8, 4, 8)
-- model = addResidualLayer(model, 8, 4, 8)
model = addResidualLayer(model, 32, 4, 10)
model = nn.SpatialAveragePooling(28,28)(model)
model = nn.Reshape(10)(model)
model = nn.LogSoftMax()(model)

model = nn.gModule({input}, {model})

loss = nn.ClassNLLCriterion()
model:cuda()
loss:cuda()

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
