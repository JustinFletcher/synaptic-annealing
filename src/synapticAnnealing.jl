
function synapticAnnealing(convCriterion, cutoffEpochs, perturbSynapses, updateState, errorFunction, reportErrorFunction, initTemperature, initLearnRate, netIn, actFun, trainData, valData, inputCols, outputCols)

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
    maxConfigDist = (abs(actFun(Inf))+abs(actFun(-Inf)))*sum(getNetworkSynapseMatrix(network).!=0)
    numEpochs = 0
	anisotropicField = 0
    stateTuple = [temperature, initTemperature,  learnRate, tunnelingField, maxConfigDist, epochsCool, numEpochs, anisotropicField]

# 	network = groundNetwork(1000, network, errorFunction, perturbSynapses, stateTuple, trainData,inputCols, outputCols)
# 	network = groundWithBackProp(500, network,trainData,inputCols, outputCols)
    # Initialize the error.
    lastError = errorFunction(network, trainData, inputCols, outputCols)
    # Initialize the loop control variables.
    converged = false

    while !converged

        # Push the most recent error onto the error vector.
        trainErr = reportErrorFunction(network, trainData, inputCols, outputCols)
        push!(trainingErrorVector, trainErr)

        # Push the validation error set onto the vector.
        valErr = reportErrorFunction(network, valData, inputCols, outputCols)
        push!(validationErrorVector, valErr)

        # State capture: if this is the best net so far, save it to the disk.
        if valErr < minError
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
        perturbedError = errorFunction(setNetworkSynapseMatrix(network, synapseMatrix), trainData, inputCols, outputCols)


        # Computer the change in error.
        errorChange = perturbedError-lastError

        # Parse the temperature from the state tuple and normalize. For readability.
        temperatureNorm = stateTuple[1]/initTemperature


        if (perturbedError<=lastError)
			lastError = perturbedError
		else
			if ((rand()>temperatureNorm) || (!(perturbationDistance<=stateTuple[3])))
				# Repeal the move.
				synapseMatrix -= synapsePerturbation
			else
				# If the annealing move is not repealed, set the error.
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
# 		println(network.weights[1][1,1])

        if((trainErr == 0.0 )&&(valErr == 0.0))
          println("Perfect Run")
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

