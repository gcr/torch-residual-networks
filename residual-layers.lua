require 'nn'
require 'nngraph'
require 'cudnn'
require 'cunn'

function addResidualLayer2(input,  nChannels, nOutChannels, stride)
   --[[

   Residual layers! Implements option (A) from Section 3.3. The input
   is passed through two 3x3 convolution filters. In parallel, if the
   number of input and output channels differ or if the stride is not
   1, then the input is downsampled or zero-padded to have the correct
   size and number of channels. Finally, the two versions of the input
   are added together.

               Input
                 |
         ,-------+-----.
   Downsampling      3x3 convolution+dimensionality reduction
        |               |
        v               v
   Zero-padding      3x3 convolution
        |               |
        `-----( Add )---'
                 |
              Output
   --]]
   nOutChannels = nOutChannels or nChannels
   stride = stride or 1
   -- Path 1: Convolution
   -- The first layer does the downsampling and the striding
   local net = cudnn.SpatialConvolution(nChannels, nOutChannels,
                                           3,3, stride,stride, 1,1)(input)
   net = nn.SpatialBatchNormalization(nOutChannels)(net)
   net = cudnn.ReLU(true)(net)
   net = cudnn.SpatialConvolution(nOutChannels, nOutChannels,
                                      3,3, 1,1, 1,1)(net)
   -- Should we put Batch Normalization here? I think not, because
   -- BN would force the output to have unit variance, which breaks the residual
   -- property of the network.
   -- What about ReLU here? I think maybe not for the same reason. Figure 2
   -- implies that they don't use it here

   -- Path 2: Identity / skip connection
   local skip = input
   if stride > 1 then
       -- optional downsampling
       skip = nn.SpatialAveragePooling(1, 1, stride,stride)(skip)
   end
   if nOutChannels > nChannels then
       -- optional padding
       skip = nn.Padding(1, (nOutChannels - nChannels), 3)(skip)
   elseif nOutChannels < nChannels then
       -- optional narrow, ugh.
       skip = nn.Narrow(2, 1, nOutChannels)(skip)
       -- NOTE this BREAKS with non-batch inputs!!
   end

   -- Add them together
   net = nn.CAddTable(){net, skip}
   net = nn.SpatialBatchNormalization(nOutChannels)(net)
   --net = cudnn.ReLU(true)(net)
   -- ^ don't put a ReLU here! see http://gitxiv.com/comments/7rffyqcPLirEEsmpX

   return net
end

--[[
function addResidualLayer3(input,  inChannels, hiddenChannels, outChannels)
   -- Downsampling and convolution path
   local net = cudnn.SpatialConvolution(inChannels, hiddenChannels,
                                           1,1)(input)
   net = cudnn.ReLU(true)(net)
   net = cudnn.SpatialConvolution(hiddenChannels, hiddenChannels,
                                      3,3, 1,1, 1,1)(net)
   --net = nn.Narrow
   net = cudnn.ReLU(true)(net)
   net = cudnn.SpatialConvolution(hiddenChannels, outChannels,
                                           1,1)(net)

   -- Add them together
   --return net
   return nn.CAddTable(){net, input}
end
--]]



--[[
-- Useful for memory debugging

function countElts(modules)
   local sum_elts = 0
   for k,v in pairs(modules) do
      if torch.isTensor(v) then
         sum_elts = sum_elts + v:numel()
      elseif torch.type(v) == 'table' then
         sum_elts = sum_elts + countElts(v)
      end
   end
   return sum_elts
end
function inspectMemory(net)
   local total_count = 0
   for i,module in ipairs(net.modules) do
      print(i..": "..tostring(module))
      local count_this_module = countElts(module)
      print(count_this_module)
      total_count = total_count + count_this_module
   end
   print("Total:",total_count)
   print("      ",total_count*8/1024./1024., " MB")
end

function accumMemoryByFieldName(module, accum)
   for k,v in pairs(module) do
      if torch.isTensor(v) then
         accum[k] = (accum[k] or 0) + (v:numel() * 8./1024./1024.)
      end
   end
end
--]]

--[[
-- Testing
input = nn.Identity()()
output = addResidualLayer2(input, 3, 6,  2)
net = nn.gModule({input},{output})

i = torch.randn(1, 3, 5,5):fill(1):cuda()
net:cuda()

net.modules[2].bias:fill(0)
net.modules[4].bias:fill(0)
net.modules[2].weight:fill(0)
net.modules[4].weight:fill(0)
--]]

-- -- Testing memory usage
-- i = torch.randn(5, 256, 224,224)
-- o = net:forward(i)
-- net:backward(i, o)
--
-- inspectMemory(net)
-- mem_usage = {}
-- for i,module in ipairs(net.modules) do
--    accumMemoryByFieldName(module, mem_usage)
-- end
-- print(mem_usage)
