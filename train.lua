require 'nngraph'
require 'nn'
require 'subnetworkBuilder'

-- The generator gets a vector of length 200, a linear layer maps it into a vector of length 2560. Then reshaped to 40 x 8 x 8 features maps and the rest is conv layers
gen = subnetworkBuilder.get_generator()
discriminator = subnetworkBuilder.get_discriminator()

input = nn.Identity()()

discriminatorReal = discriminator:clone('weight', 'bias', 'gradWeight', 'gradBias') -- To be used for updating the discriminator weights only
discriminatorFake = discriminator:clone('weight', 'bias', 'gradWeight', 'gradBias') -- To be used for updating the discriminator weights only
discriminatorUpdateGenerator = discriminator:clone('weight', 'bias', 'gradWeight', 'gradBias') -- To be used for updating the Generator weights only

-- Prevent the discriminator weights get updated when wanting to update the Generator weights
discriminatorUpdateGenerator.accGradParameters = function() end
discriminatorUpdateGenerator.updateParameters = function() end -- The Optim package functions do not call updateParameters function though


-- This code snippet zeros-out the input gradients and prevents the gradients reach to previous network (the Generator)
tempLayer = discriminatorReal:get(1)
function tempLayer:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(input):zero()
    return self.gradInput
end
discriminatorReal.updateParameters = function() end -- If this function would have been called, gradients would get accumulated. But it doesn't when using the Opitm package functions!

-- This code snippet zeros-out the input gradients and prevents the gradients reach to previous network (the Generator)
tempLayerFake = discriminatorFake:get(1)
function tempLayerFake:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(input):zero()
    return self.gradInput
end

z = nn.Identity()() -- A vector of length 200
samplesGenerated = gen(z)
disOutputDecoder = discriminatorUpdateDecoder(samplesGenerated)
disOutputReal = discriminatorReal(input)
disOutputFake = discriminatorFake(samplesGenerated)

gMod = nn.gModule({input, z}, {disOutputReal, disOutputFake, disOutputDecoder})

params, gParams = gMod:getParameters()


batch_size = 32
noise = torch.Tensor(1, 200)
GANLabelsDisFake = torch.zeros(batch_size, 1)
GANLabelsDisReal = torch.ones(batch_size, 1)
GANLabelsGen = torch.ones(batch_size, 1)


GANCriterionReal = nn.BCECriterion()
GANCriterionReal.sizeAverage = false
GANCriterionFake = nn.BCECriterion()
GANCriterionFake.sizeAverage = false
GANCriterionGen = nn.BCECriterion()
GANCriterionGen.sizeAverage = false


-- Assuming the data are in a tensor named "data"
indices = commonFuncs.generateBatchIndices(data:size(1), batch_size)

config = {learningRate = 0.01}

for t,v in ipairs(indices) do
	noise:normal(0, 1)
	local inputs = {data:index(1,v), noise}

	local opfunc = function(x)
		if x ~= params then
			params:copy(x)
		end

		gMod:zeroGradParameters()

		local disReal, disFake, disGen = unpack(model:forward(inputs))

		local errReal = GANCriterionReal:forward(disReal, GANLabelsDisReal)
        local dGANReal_dw = GANCriterionReal:backward(disReal, GANLabelsDisReal)

        local errFake = GANCriterionFake:forward(disFake, GANLabelsDisFake)
        local dGANFake_dw = GANCriterionFake:backward(disFake, GANLabelsDisFake)

        local errGen = GANCriterionGen:forward(disGen, GANLabelsGen)
        local dGAN_dwGen = GANCriterionGen:backward(disGen, GANLabelsGen)

        local error_grads = {dGANReal_dw, dGANFake_dw, dGAN_dwGen}

        model:backward(inputs, error_grads)
        local batchError = errReal + errFake

        return batchError, gParams
	end

	x, err = optim.adam(opfunc, params, config)
end
