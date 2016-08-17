require 'torch'
require 'nn'

local subnetworkBuilder = {}

function subnetworkBuilder.get_generator()
  local generator = nn.Sequential()
  -- Add layers of your own interest
  
  return generator
end

function subnetworkBuilder.get_discriminator()
  local discriminator = nn.Sequential()
  -- Add layers of your own interest
  
  return discriminator
end

return subnetworkBuilder
