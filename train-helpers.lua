require 'optim'
TrainingHelpers = {}
local display = require 'display'

function TrainingHelpers.inspectLayer(layer, fields)
   function inspect(x)
      if x then
         x = x:double():view(-1)
         return {
            p5 = (x:kthvalue(1 + 0.05*x:size(1))[1]),
            mean = x:mean(),
            p95 = (x:kthvalue(1 + 0.95*x:size(1))[1]),
            var = x:var(),
         }
      end
   end
   local result = {name = tostring(layer)}
   for _,field in ipairs(fields) do
      result[field] = inspect(layer[field])
   end
   return result
end
function TrainingHelpers.printLayerInspection(li, fields)
   print("- "..tostring(li.name))
   if (string.find(tostring(li.name), "ReLU")
       or string.find(tostring(li.name), "BatchNorm")
       or string.find(tostring(li.name), "View")
       ) then
       -- Do not print these layers
   else
       for _,field in ipairs(fields) do
          local lf = li[field]
          if lf then
              print(string.format(
                       "%20s    5p: %+3e    Mean: %+3e    95p: %+3e    Var: %+3e",
                       field, lf.p5, lf.mean, lf.p95, lf.var))
          end
       end
   end
end
function TrainingHelpers.inspectModel(model)
   local results = {}
   for i,layer in ipairs(model.modules) do
      results[i] = TrainingHelpers.inspectLayer(layer, {"weight",
                                                        "gradWeight",
                                                        "bias",
                                                        "gradBias",
                                                        "output"})
   end
   return results
end
function TrainingHelpers.printInspection(inspection)
   print("\n\n\n")
   print(" \x1b[31m---------------------- Weights ---------------------- \x1b[0m")
   for i,layer in ipairs(inspection) do
      TrainingHelpers.printLayerInspection(layer, {"weight", "gradWeight"})
   end
   print(" \x1b[31m---------------------- Biases ---------------------- \x1b[0m")
   for i,layer in ipairs(inspection) do
      TrainingHelpers.printLayerInspection(layer, {"bias", "gradBias"})
   end
   print(" \x1b[31m---------------------- Outputs ---------------------- \x1b[0m")
   for i,layer in ipairs(inspection) do
      TrainingHelpers.printLayerInspection(layer, {"output"})
   end
end


function recordLoss(sgdState, loss_val)
   sgdState.lossLog = sgdState.lossLog or {}
   sgdState.lossLog[#sgdState.lossLog + 1] = {
      sgdState.nSampledImages,
      loss_val
   }
end
function displayLoss(sgdState, loss_val)
   display.plot(sgdState.lossLog, {labels={'Images Seen', 'Loss'},
                      title='Loss',
                      rollPeriod=10,
                      showRoller=true,
                      win=25})
end
function displayWeights(model)
    local layers = {}
    -- Go through each module and add its weight and its gradient.
    -- X axis = layer number.
    -- Y axis = weight / gradient.
    for i, li in ipairs(model.modules) do
        if not (string.find(tostring(li), "ReLU")
            or string.find(tostring(li), "BatchNorm")
            or string.find(tostring(li), "View")
            ) then
            if li.gradWeight then
                --print(tostring(li),li.weight:mean())
                layers[#layers+1] = {i,
                    -- Weight
                    {li.weight:mean() - li.weight:std(),
                    li.weight:mean(),
                    li.weight:mean() + li.weight:std()},
                    -- Gradient
                    {li.gradWeight:mean() - li.gradWeight:std(),
                    li.gradWeight:mean(),
                    li.gradWeight:mean() + li.gradWeight:std()},
                    -- Output
                    {li.output:mean() - li.output:std(),
                    li.output:mean(),
                    li.output:mean() + li.output:std()},
                }
            end
        end
    end
    -- Plot the result
    --
   display.plot(layers, {
                   labels={"Layer", "Weights", "Gradients", "Outputs"},
                   customBars=true, errorBars=true,
                   title='Network Weights',
                   rollPeriod=1,
                   win=26,
                   --annotations={"o"},
                   --axes={x={valueFormatter="function(x) {return x; }"}},
             })
end


function evaluateModel(model, datasetTest, epochSize)
   print("Evaluating...")
   model:evaluate()
   local correct1 = 0
   local correct5 = 0
   local total = 0
   while total < datasetTest:size() do
       local batch, labels = datasetTest:getBatch()
       local y = model:forward(batch:cuda()):float()
       local _, indices = torch.sort(y, 2, true)
       -- indices has shape (batchSize, nClasses)
       local top1 = indices:select(2, 1)
       local top5 = indices:narrow(2, 1,5)
       correct1 = correct1 + torch.eq(top1, labels):sum()
       correct5 = correct5 + torch.eq(top5, labels:view(-1, 1):expandAs(top5)):sum()
       total = total + indices:size(1)
       xlua.progress(total, datasetTest:size())
   end
   return {correct1=correct1/total, correct5=correct5/total}
end

function TrainingHelpers.trainForever(model, forwardBackwardBatch, weights, sgdState, epochSize, afterEpoch, filename)
   local d = Date{os.date()}
   local modelTag = string.format("%04d%02d%02d-%d",
      d:year(), d:month(), d:day(), torch.random())
   local newFilename = filename.."-"..modelTag
   print("Saving to "..newFilename)
   sgdState.epochSize = epochSize
   if sgdState.epochCounter == nil then
      sgdState.epochCounter = 0
   end
   if sgdState.nSampledImages == nil then
     sgdState.nSampledImages = 0
   end
   if sgdState.nEvalCounter == nil then
      sgdState.nEvalCounter = 0
   end
   local whichOptimMethod = optim.sgd
   if sgdState.whichOptimMethod then
       whichOptimMethod = optim[sgdState.whichOptimMethod]
   end
   while true do -- Each epoch
      collectgarbage(); collectgarbage()
      -- Run forward and backward pass on inputs and labels
      model:training()
      local loss_val, gradients, batchProcessed = forwardBackwardBatch()
      -- SGD step: modifies weights in-place
      whichOptimMethod(function() return loss_val, gradients end,
                       weights,
                       sgdState)
      -- Display progress and loss
      sgdState.nSampledImages = sgdState.nSampledImages + batchProcessed
      sgdState.nEvalCounter = sgdState.nEvalCounter + 1
      xlua.progress(sgdState.nSampledImages%epochSize, epochSize)

      recordLoss(sgdState, loss_val)
      if sgdState.nEvalCounter % 20 == 0 then
          displayLoss(sgdState, loss_val)
          displayWeights(model)
      --     print("\027[KLoss:", loss_val)
      --     print("Gradients:", gradients:norm())
      end
      -- if sgdState.nEvalCounter % 100 == 0 then
      --     local inspection = TrainingHelpers.inspectModel(model)
      --     inspection.nSampledImages = sgdState.nSampledImages
      --     table.insert(sgdState.inspectionLog, inspection)
      -- end
      --table.insert(sgdState.lossLog, {loss = loss_val, nSampledImages = sgdState.nSampledImages})
      if math.floor(sgdState.nSampledImages / epochSize) ~= sgdState.epochCounter then
         -- Epoch completed!
         xlua.progress(10,10)
         sgdState.epochCounter = math.floor(sgdState.nSampledImages / epochSize)
         if afterEpoch then afterEpoch() end

         print("\n\n----- Epoch "..sgdState.epochCounter.." -----")
         -- Every so often, decrease learning rate
         if sgdState.epochCounter % sgdState.epochDropCount == 0 then
            sgdState.learningRate = sgdState.learningRate * 0.1
            print("Dropped learning rate to", sgdState.learningRate)
         end
         -- Snapshot model (WARNING: Should be the last thing we do!)
         if filename then
            print("Snapshotting model to "..newFilename)
            torch.save(newFilename.."-model.tmp", model)
            os.rename(newFilename.."-model.tmp", newFilename.."-model.t7") -- POSIX guarantees automicity
            print("Snapshotting sgdState to "..newFilename)
            torch.save(newFilename.."-sgdState.tmp", sgdState)
            os.rename(newFilename.."-sgdState.tmp", newFilename.."-sgdState.t7") -- POSIX guarantees automicity
         end
      end
   end
end
