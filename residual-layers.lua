require 'nn'
require 'nngraph'

function addResidualLayer(input,  inChannels, hiddenChannels, outChannels)
   -- Downsampling and convolution path
   local net = nn.SpatialConvolution(inChannels, hiddenChannels,
                                           1,1)(input)
   net = nn.ReLU(true)(net)
   net = nn.SpatialConvolution(hiddenChannels, hiddenChannels,
                                      3,3, 1,1, 1,1)(net)
   --net = nn.Narrow
   net = nn.ReLU(true)(net)
   net = nn.SpatialConvolution(hiddenChannels, outChannels,
                                           1,1)(net)

   -- Add them together
   --return net
   return nn.CAddTable(){net, input}
end

input = nn.Identity()()
net = addResidualLayer(input, 256,64,256)
net = nn.gModule({input}, {net})
graph.dot(net.fg, "MLP")


-- function countElts(modules)
--    local sum_elts = 0
--    for k,v in pairs(modules) do
--       if torch.isTensor(v) then
--          sum_elts = sum_elts + v:numel()
--       elseif torch.type(v) == 'table' then
--          sum_elts = sum_elts + countElts(v)
--       end
--    end
--    return sum_elts
-- end
-- function inspectMemory(net)
--    local total_count = 0
--    for i,module in ipairs(net.modules) do
--       print(i..": "..tostring(module))
--       local count_this_module = countElts(module)
--       print(count_this_module)
--       total_count = total_count + count_this_module
--    end
--    print("Total:",total_count)
--    print("      ",total_count*8/1024./1024., " MB")
-- end
--
-- function accumMemoryByFieldName(module, accum)
--    for k,v in pairs(module) do
--       if torch.isTensor(v) then
--          accum[k] = (accum[k] or 0) + (v:numel() * 8./1024./1024.)
--       end
--    end
-- end
--
--
--
--
-- input = nn.Identity()()
-- output = addResidualLayer(input, 256,64,256)
-- net = nn.gModule({input},{output})
--
-- -- Let's make some!
-- i = torch.randn(5, 256, 224,224)
-- o = net:forward(i)
-- net:backward(i, o)
--
-- inspectMemory(net)
--
-- mem_usage = {}
-- for i,module in ipairs(net.modules) do
--    accumMemoryByFieldName(module, mem_usage)
-- end
-- print(mem_usage)