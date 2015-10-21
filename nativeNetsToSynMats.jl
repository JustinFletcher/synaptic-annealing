
function getNetworkSynapseMatrix(network)

	# Construct s matrix of zeros of the size needed.
	synapseMatrix = zeros(maximum(net.structure)+1, maximum(net.structure)+1, length(net.structure)-1)

	# Comprehension maps each layer matrix of the network object to the unified synapse matrix.
	[synapseMatrix[1:size(net.weights[layerInd])[1], 1:size(net.weights[layerInd])[2], layerInd] = net.weights[layerInd] for layerInd in 1:length(net.weights)]

	return(synapseMatrix)
end

function setNetworkSynapseMatrix(network, synMat)

	[network.weights[layerInd] = synMat[1:network.structure[layerInd]+1, 1:network.structure[layerInd+1], layerInd] for layerInd in 1:size(synMat)[3]]

	return(network)
end


