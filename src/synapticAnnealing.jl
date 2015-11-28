
function synapticAnnealing(convCriterion, cutoffEpochs, perturbSynapses, updateState, errorFunction, reportErrorFunction, initTemperature, initLearnRate, netIn, actFun, trainData, valData)

    println("New Synaptic Annealing")

#     Parse the input data tuple into train and val tuples.
#     (trainData, valData, inputCols, outputCols) = dataTuple

    # Create a local copy of the synapse matrix.
    network = netIn


    # Initiailize the minimum errors.
    minError = Inf
    trainErr = Inf
    valErr = Inf

    # Initialize the error vectors.
    trainingErrorVector = Float64[]
    validationErrorVector = Float64[]
    onlineErrorVector = Float64[]
    perturbationDistanceVector = Float64[]
    errorStack = Float64[]
    minValErrSynapseMatrix = Any[]

    # Initialize state variables.
    temperature = initTemperature
    learnRate = initLearnRate
    tunnelingField = 0
    epochsCool = 0
    maxConfigDist = sum(getNetworkSynapseMatrix(network).!=0) #(abs(actFun(Inf))+abs(actFun(-Inf)))*sum(getNetworkSynapseMatrix(network).!=0)
    numEpochs = 0
	anisotropicField = 0
    stateTuple = [temperature, initTemperature,  learnRate, tunnelingField, maxConfigDist, epochsCool, numEpochs, anisotropicField]

# 	network = groundNetwork(1000, network, errorFunction, perturbSynapses, stateTuple, trainData,inputCols, outputCols)
# 	network = groundWithBackProp(500, network,trainData,inputCols, outputCols)
    # Initialize the error.
    lastError = errorFunction(network, trainData)
    # Initialize the loop control variables.
    converged = false

    while !converged

		# Minibatch goes here.

        # Push the most recent error onto the error vector.
        trainErr = reportErrorFunction(network, trainData)
        push!(trainingErrorVector, trainErr)

        # Push the validation error set onto the vector.
        valErr = reportErrorFunction(network, valData)
        push!(validationErrorVector, valErr)

        # State capture: if this is the best net so far, save it to the disk.
        if valErr < minError
			minError = valErr
            minValErrSynapseMatrix = network
        end



        # Update the state of the annealing system.
        stateTuple = updateState(stateTuple)

		synapseMatrix = getNetworkSynapseMatrix(network)

        # Compute the synapse perturbation matrix.
        synapsePerturbationTuple = perturbSynapses(synapseMatrix, stateTuple)

        # Parse the perturbation tuple. For readability.
        (synapsePerturbation, perturbationDistance) = synapsePerturbationTuple

        # Append the perturbation distance to an output vector. For analysis.
        push!(perturbationDistanceVector, perturbationDistance)

        # Modify the synapse matrix using the perturbation matrix.
        synapseMatrix += synapsePerturbation

        # Compute the resultant error.
        perturbedError = errorFunction(setNetworkSynapseMatrix(network, synapseMatrix), trainData)

#         # State capture: if this is the best net so far, save it to the disk.
#         if perturbedError < minTrainError
# 			minTrainError = perturbedError
#         end

        # Computer the change in error.
        errorChange = perturbedError-lastError

        # Parse the temperature from the state tuple and normalize. For readability.
        temperatureNorm = stateTuple[1]/initTemperature

		# If this is a downhill move, take it.
        if (perturbedError<=lastError)
			lastError = perturbedError

		# If this is not a downhill or neural move...
		else
			# Then with probability exp(-DeltaE/T), or if it's the result of a tunnelling event, reject the move.
			if( (rand()>=exp(-(perturbedError-lastError)/temperatureNorm)) || (!(perturbationDistance<=stateTuple[3])) )
				synapseMatrix -= synapsePerturbation
			# If the uphill move is not rejected, set the error.
			else
				lastError = perturbedError
			end

        end
#         # If the move was a quantum one.
#         elseif !(perturbationDistance<=stateTuple[3])
#           # Repeal the move.
#           synapseMatrix -= synapsePerturbation
#         elseif (rand()<temperatureNorm)
#           # Repeal the move.
#           synapseMatrix -= synapsePerturbation
#         else
#           # If the annealing move is not repealed, set the error.
#           lastError = perturbedError
#         end

		network = setNetworkSynapseMatrix(network, synapseMatrix)

		# If this run is perfect, stop evaluating.
        if((trainErr == 0.0 )&&(valErr == 0.0))
          println("Perfect Run | Epoch "*stateTuple[7])
        end

        # Evaluate the convergence conditions.
        converged = (stateTuple[7]>=cutoffEpochs) || ((trainErr == 0.0 )&&(valErr == 0.0))

    end

# 	minValErrSynapseMatrix = synapseMatrix
# 	validationErrorVector = trainingErrorVector

    # Construct and return the output tuple.
    outputTuple = Any[minValErrSynapseMatrix, validationErrorVector, trainingErrorVector, perturbationDistanceVector]

    return(outputTuple)

end



function groundNetwork(cutoffEpochs, network, errorFunction, perturbSynapses, stateTuple, trainData,inputCols, outputCols)

	numEpochs = 0

    lastError = errorFunction(network, trainData, inputCols, outputCols)

    while !(numEpochs>=cutoffEpochs)

		numEpochs += 1

		synapseMatrix = getNetworkSynapseMatrix(network)

        # Compute the synapse perturbation matrix.
        synapsePerturbationTuple = perturbSynapses(synapseMatrix, stateTuple)

        # Parse the perturbation tuple. For readability.
        (synapsePerturbation, perturbationDistance) = synapsePerturbationTuple

        # Modify the synapse matrix using the perturbation matrix.
        synapseMatrix += synapsePerturbation

        # Compute the resultant error.
        perturbedError = errorFunction(setNetworkSynapseMatrix(network, synapseMatrix), trainData, inputCols, outputCols)


        if (perturbedError<=lastError)
			lastError = perturbedError
		else
			synapseMatrix -= synapsePerturbation
        end


		network = setNetworkSynapseMatrix(network, synapseMatrix)

    end
	return(network)
end

