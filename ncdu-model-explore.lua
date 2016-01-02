json = require 'cjson'

function buildNcduLayer(name, module)
   local result = nil
   if torch.isTensor(module) then
      if module:numel() ~= 0 then
         local strt = {name..': [' .. torch.typename(module) .. ' of size '}
         for i=1,module:nDimension() do
            table.insert(strt, module:size(i))
            if i ~= module:nDimension() then
               table.insert(strt, 'x')
            end
         end
         table.insert(strt, ']')
         result = {name = table.concat(strt),
                   dsize = module:storage():size() * module:storage():elementSize()
                }
      else
         result = {name = name..": [empty "..torch.typename(module).."]"}
      end
   elseif type(module)=="table" and module.modules then
      result = { {name = name..": "..string.gsub(tostring(module), "\n", " ")} }
      for i,m in ipairs(module.modules) do
         table.insert(result, buildNcduLayer(string.gsub(tostring(i), "\n", " "), m))
      end
   elseif type(module)=="table" then
      result = {{name=name..": "..string.gsub(tostring(module), "\n", " ")}}
      for k,v in pairs(module) do
         table.insert(result, buildNcduLayer(k,v))
      end
   else
      result = {name=name.." (primitive)", dsize=0}
   end
   return result
end

function buildNcdu(model)
   local result = {1,0,{timestamp=1451677436,progver="0.1",progname="ncdu-model-explore"}}
   table.insert(result, buildNcduLayer("model", model))
   return json.encode(result)
end

function exploreNcdu(model)
   local tmpname = os.tmpname()
   local tmphandle = io.open(tmpname, "w")
   tmphandle:write(buildNcdu(model))
   tmphandle:close()
   os.execute("ncdu -f "..tmpname)
   os.unlink(tmpname)
end


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

-- function accumMemoryByFieldName(module, accum)
--    for k,v in pairs(module) do
--       if torch.isTensor(v) then
--          accum[k] = (accum[k] or 0) + (v:numel() * 8./1024./1024.)
--       end
--    end
-- end


require 'nn'
model = nn.Sequential()
model:add(nn.SpatialConvolution(3,5,  10,10, 1,1))
model:add(nn.SpatialConvolution(5,10, 3,3,1,1))
model:forward(torch.randn(10, 3, 224,224))

print(exploreNcdu(model))