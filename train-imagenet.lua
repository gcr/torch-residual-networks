require 'residual-layers'
require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'train-helpers'
display = require 'display'

opt = lapp[[
      --batchSize       (default 64)     Batch size
      --nThreads        (default 4)       Data loader threads
      --dataTrainRoot   (default /mnt/imagenet/train)   Data root folder
      --dataValRoot     (default /mnt/imagenet/val)   Data root folder
      --verbose         (default true)
      --loadSize        (default 256)     Size of image when loading
      --fineSize        (default 224)     Size of image crop
]]
print(opt)

-- create data loader
local DataLoader = paths.dofile('data.lua')
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
input = nn.Identity()()
model = cudnn.SpatialConvolution(3, 64, 7,7, 2,2, 3,3)(input)
------> 64, 112,112
model = cudnn.ReLU(true)(model)
model = cudnn.SpatialMaxPooling(3,3,  2,2,  1,1)(model)
------> 64, 56,56
model = addResidualLayer2(model, 64)
model = nn.SpatialBatchNormalization(64)(model)
--model = addResidualLayer2(model, 64)
--model = nn.SpatialBatchNormalization(64)(model)
model = addResidualLayer2(model, 64, 128, 2)
------> 128, 28,28
model = addResidualLayer2(model, 128)
model = nn.SpatialBatchNormalization(128)(model)
--model = addResidualLayer2(model, 128)
--model = nn.SpatialBatchNormalization(128)(model)
model = addResidualLayer2(model, 128, 256, 2)
------> 256, 14,14
model = addResidualLayer2(model, 256)
model = nn.SpatialBatchNormalization(256)(model)
--model = addResidualLayer2(model, 256)
--model = nn.SpatialBatchNormalization(256)(model)
model = addResidualLayer2(model, 256, 512, 2)
------> 512, 7,7
model = addResidualLayer2(model, 512)
model = nn.SpatialBatchNormalization(512)(model)
--model = addResidualLayer2(model, 512)
--model = nn.SpatialBatchNormalization(512)(model)
model = cudnn.ReLU(true)(cudnn.SpatialConvolution(512, 1000, 7,7)(model))
------> 1000, 1,1
model = nn.Reshape(1000)(model)
------> 1000
model = nn.LogSoftMax()(model)

model = nn.gModule({input}, {model})

loss = nn.ClassNLLCriterion()
model:cuda()
loss:cuda()


-- DIRTY TRICK to use :)
--model.modules[2].weight:mul(0.1)

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



--[[
-- Show memory usage
print(#model:forward(torch.randn(64, 3, 224,224):cuda()))
inspectMemory(model)
mem_usage = {}
for i,module in ipairs(model.modules) do
   accumMemoryByFieldName(module, mem_usage)
end
print(mem_usage)
--]]


sgdState = {
   learningRate = 0.01,
   momentum     = 0.9,
   dampening    = 0,
   weightDecay    = 1e-6,
   nesterov     = true,
   --rho              = 0.5,
   --whichOptimMethod = 'rmsprop',
   epochDropCount = 20,

   -- Train stuff
   options = opt,
   accuracies = {},
}

-- Actual Training! -----------------------------
weights, gradients = model:getParameters()
function forwardBackwardBatch(batch)
   model:training()
   gradients:zero()
   local inputs = batch[1]:cuda()
   local labels = batch[2]:cuda()
   local y = model:forward(inputs)
   local loss_val = loss:forward(y, labels)
   local df_dw = loss:backward(y, labels)
   model:backward(inputs, df_dw)

   display.image(model.modules[2].weight, {win=24, title="First layer weights"})

   return loss_val, gradients
end


function evalModel()
   print("No evaluation...")
   -- if sgdState.epochCounter > 10 then os.exit(1) end
   local results = evaluateModel(model, dataVal)
   print(results)
   table.insert(sgdState.accuracies, acc)
end

--[[
local results = evaluateModel(model, dataVal)
print(results)
--]]

---[[
require 'graph'
graph.dot(model.fg, 'MLP', '/tmp/MLP')
os.execute('convert /tmp/MLP.svg /tmp/MLP.png')
display.image(image.load('/tmp/MLP.png'), {title="Network Structure", win=23})
--]]


---[[
TrainingHelpers.trainForever(
   model,
   forwardBackwardBatch,
   weights,
   sgdState,
   function()
      inputs,labels = dataTrain:getBatch()
      return {inputs,labels}
   end,
   dataTrain:size(),
   evalModel,
   "snapshots/imagenet-residual-experiment1"
)
--]]
