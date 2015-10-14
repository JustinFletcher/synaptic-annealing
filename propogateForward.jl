

@everywhere function propogateForward(inputPattern, synapseMatrix, actFun)

    # Initialize the propigation pattern to the input pattern. Pad the propPattern with zeros to size with weightMatrix.
    propPattern = [1 [actFun(inputPattern) zeros(1, size(synapseMatrix)[1]-length(inputPattern)-1)]]

    # For each "layer" of connection weights, except the last one, which we just want to "read."
    for layerIndex in 1:size(synapseMatrix)[3]-1

# 		propPatternT = repmat(transpose(propPattern), 1, size(synapseMatrix)[2])

# 		synapticWeights = synapseMatrix[:,:,layerIndex]

#      	@devec propPattern = tanh(sum((propPatternT .* synapticWeights), 1))
# 		println(synapticWeights)
# 		@devec postSynapseSignals = propPatternT .* synapticWeights
# 		println(postSynapseSignals)

# 		for col in 1:size(synapseMatrix)[2]

# 		end

# 		@devec integratedSignals = sum(postSynapseSignals, 1)

     	propPattern = tanh(sum((transpose(propPattern) .* synapseMatrix[:,:,layerIndex]), 1))

    end

    # Return the final propigation pattern, which is the output.
    return(transpose(propPattern))

end


# @everywhere function propogateForward(inputPattern, synapseMatrix, actFun)

# 	propPattern = inputPattern
#     # For each "layer" of connection weights, except the last one, which we just want to "read."
#     for layerIndex in 1:size(synapseMatrix)[3]-1
# 		propPattern = [1 propPattern zeros(1, maximum([0, size(synapseMatrix)[1]-length(propPattern)-1]))]

#         # Caluculate the sum along the columns of the propPattern distributed over the synapseMatrix.
# #         propPattern = actFun(sum((transpose([1 propPattern]) .* synapseMatrix[1:length(propPattern)+1,:,layerIndex]), 1))
#         propPattern = actFun(sum((transpose(propPattern) .* synapseMatrix[:,:,layerIndex]), 1))

#     end
#     # Return the final propigation pattern, which is the output.
#     return(transpose(propPattern))

# end


# @everywhere function propogateForward(inputPattern, synapseMatrix, actFun)


# 	propPattern = inputPattern
#     # For each "layer" of connection weights, except the last one, which we just want to "read."
#     for layerIndex in 1:size(synapseMatrix)[3]-1

# 		propPattern = [1 propPattern]

# 		propPattern = [propPattern zeros(1, maximum([0, size(synapseMatrix)[1]-length(propPattern)-1]))]

# 		propPattern = transpose(propPattern)

# 		synOutputMatrix = propPattern .* synapseMatrix[:,:,layerIndex]

# 		signalIntegrationVector = sum(synOutputMatrix, 1)

#         # Caluculate the sum along the columns of the propPattern distributed over the synapseMatrix.
#         propPattern = actFun(signalIntegrationVector)

#     end

#     # Return the final propigation pattern, which is the output.
#     return(transpose(propPattern))

# end

