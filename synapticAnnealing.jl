
function synapticAnnealing(convCriterion, cutoffEpochs, perturbSynapses, updateState, errorFunction, reportErrorFunction, initTemperature, initLearnRate, synapseMatrixIn, actFun, trainData, valData, inputCols, outputCols)

    println("New Synaptic Annealing Run")

#     Parse the input data tuple into train and val tuples.
#     (trainData, valData, inputCols, outputCols) = dataTuple

    # Create a local copy of the synapse matrix.
    synapseMatrix = copy(synapseMatrixIn)

    # Initialize the error.
    lastError = errorFunction(synapseMatrix, actFun, trainData, inputCols, outputCols)

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
    minValErrSynapseMatrix = null

    # Developmental
    varianceVector = Float64[]

    # Initialize state variables.
    temperature = initTemperature
    learnRate = initLearnRate
    tunnelingField = 0
    epochsCool = 0
    maxConfigDist = (abs(actFun(Inf))+abs(actFun(-Inf)))*sum(synapseMatrix.!=0)
    numEpochs = 0
	anisotropicField = 0
    stateTuple = [temperature, initTemperature,  learnRate, tunnelingField, maxConfigDist, epochsCool, numEpochs, anisotropicField]

    # Initialize the loop control variables.
    converged = false

    while !converged

        # Push the most recent error onto the error vector.
        trainErr = reportErrorFunction(synapseMatrix, actFun, trainData, inputCols, outputCols)
        push!(trainingErrorVector, trainErr)

        # Push the validation error set onto the vector.
        valErr = reportErrorFunction(synapseMatrix, actFun, valData, inputCols, outputCols)
        push!(validationErrorVector, valErr)

        # State capture: if this is the best net so far, save it to the disk.
        if valErr < minError
            minValErrSynapseMatrix = synapseMatrix
        end

        # Update the state of the annealing system.
        stateTuple = updateState(stateTuple)

#         # Select a random element of the training data.
#         trainSample = trainData[rand(1:end),:]

        # Compute the synapse perturbation matrix.
        synapsePerturbationTuple = perturbSynapses(synapseMatrix, stateTuple)

        # Parse the perturbation tuple. For readability.
        (synapsePerturbation, perturbationDistance) = synapsePerturbationTuple

        # Append the perturbation distance to an output vector. For analysis.
        push!(perturbationDistanceVector, perturbationDistance)

		# Adjust the perturbation to keep it within the activation range.
# 		synapsePerturbation = synapsePerturbation + (1.-(synapseMatrix + synapsePerturbation)).*int((1-synapseMatrix + synapsePerturbation).<0)
# 		synapsePerturbation = synapsePerturbation + (-1-(synapseMatrix + synapsePerturbation)).*int((-1-synapseMatrix + synapsePerturbation).>0)

        # Modify the synapse matrix using the perturbation matrix.
        synapseMatrix += synapsePerturbation

        # Compute the resultant error.
        perturbedError = errorFunction(synapseMatrix, actFun, trainData, inputCols, outputCols)

#         push!(trainingErrorVector,  perturbedError)


        # Computer the change in error.
        errorChange = perturbedError-lastError

        # Parse the temperature from the state tuple and normalize. For readability.
        temperatureNorm = stateTuple[1]/initTemperature

# 		println(temperatureNorm, "||", errorChange, "||", acceptanceProbability)
#         # With probabability of acceptanceProbability, take the move.
#         if (rand()<acceptanceProbability)

#             # If the annealing move is not repealed, set the error.
#             lastError = perturbedError
#         else

#             # Repeal the move.
#             synapseMatrix -= synapsePerturbation
#         end

        # If the error has increased, then with probability=temperature, allow this move.
#         if ((perturbedError>lastError) && !(perturbationDistance<=stateTuple[3]))

# 			if (rand()>temperatureNorm)

# 				# Repeal the move.
# 				synapseMatrix -= synapsePerturbation

# 			else

# 				# If the annealing move is not repealed, set the error.
# 				lastError = perturbedError

# 			end

#         else

#             # If the annealing move is not repealed, set the error.
#             lastError = perturbedError

#         end

		if (perturbedError<lastError)
			lastError = perturbedError
		# If the move was a quantum one.
		elseif !(perturbationDistance<=stateTuple[3])
			# Repeal the move.
			synapseMatrix -= synapsePerturbation
		elseif (rand()>temperatureNorm)
			# Repeal the move.
			synapseMatrix -= synapsePerturbation
		else
			# If the annealing move is not repealed, set the error.
			lastError = perturbedError
		end



#         # Add the most recent error to the error history.
#         push!(errorStack, lastError)
#         push!(onlineErrorVector, lastError)

		if((trainErr == 0.0 )&&(valErr == 0.0))
			println("Perfect Run")
		end

        # Evaluate the convergence conditions.
        converged = (stateTuple[7]>=cutoffEpochs)  #|| ((valErr<convCriterion)&&(trainErr<convCriterion)) || ((trainErr == 0.0 )&&(valErr == 0.0))

    end

# 	minValErrSynapseMatrix = synapseMatrix
# 	validationErrorVector = trainingErrorVector

    # Construct and return the output tuple.
    outputTuple = Any[minValErrSynapseMatrix, validationErrorVector, trainingErrorVector, perturbationDistanceVector]

    return(outputTuple)

end

