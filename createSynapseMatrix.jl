# function createSynapseMatrix(layerSpecVector)

#     # Get the number of nodes in the largest layer.
#     maxNumNodes = maximum(layerSpecVector)

#     # Get the number of layers.
#     numLayers = length(layerSpecVector)

#     # Initialize an empty array to contain the weights.
#     weightsArray = zeros(maxNumNodes, maxNumNodes, numLayers)

#     for layer in 1:numLayers-1
#         thisLayerNumNeurons = layerSpecVector[layer]

#         nextLayerNumNeurons = layerSpecVector[layer+1]

#         weightsArray[1:thisLayerNumNeurons, 1:nextLayerNumNeurons, layer] = 0.2*(rand(thisLayerNumNeurons, nextLayerNumNeurons)-0.5)
#     end

#     weightsArray = weightsArray[:,:, 1:end-1]
#     return(weightsArray)
# end

# To revert remove +1...

function createSynapseMatrix(layerSpecVector)

    # Get the number of nodes in the largest layer.
    maxNumNodes = maximum(layerSpecVector)

    # Get the number of layers.
    numLayers = length(layerSpecVector)
	# add a +1 in the 1st dimension here to make all bias..
    # Initialize an empty array to contain the weights.
    weightsArray = zeros(maxNumNodes, maxNumNodes, numLayers)

    for layer in 1:numLayers-1
        thisLayerNumNeurons = layerSpecVector[layer]

        nextLayerNumNeurons = layerSpecVector[layer+1]

        weightsArray[1:thisLayerNumNeurons, 1:nextLayerNumNeurons, layer] = 0.2*(rand(thisLayerNumNeurons, nextLayerNumNeurons)-0.5)
    end

    return(weightsArray)
end

# diagm(0.2*(rand(10)-0.5))

