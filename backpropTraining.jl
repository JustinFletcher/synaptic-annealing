
function backpropTraining(convCriterion, cutoffEpochs, perturbSynapses, updateState, errorFunction, reportErrorFunction, initTemperature, initLearnRate, netIn, actFun, trainData, valData, inputCols, outputCols)

    println("New Backprop Run")

#     Parse the input data tuple into train and val tuples.
#     (trainData, valData, inputCols, outputCols) = dataTuple

    # Create a local copy of the synapse matrix.
    net = netIn

    # Initiailize the minimum errors.
    minError = Inf
    trainErr = Inf
    valErr = Inf

    # Initialize the error vectors.
    trainingErrorVector = Float64[]
    validationErrorVector = Float64[]
    perturbationDistanceVector = Float64[]
    minValErrSynapseMatrix = null

	numEpochs = 0


    # Initialize the loop control variables.
    converged = false

    while !converged


		numEpochs += 1


        # Push the most recent error onto the error vector.
        trainErr = reportErrorFunction(net, trainData, inputCols, outputCols)
        push!(trainingErrorVector, trainErr)

        # Push the validation error set onto the vector.
        valErr = reportErrorFunction(net, valData, inputCols, outputCols)
        push!(validationErrorVector, valErr)


		for sampleRow in 1:size(trainData)[1]
			train(net, vec(trainData[sampleRow, inputCols]), vec(trainData[sampleRow, outputCols]))
		end


        # Evaluate the convergence conditions.
        converged = (numEpochs>=cutoffEpochs)  #|| ((valErr<convCriterion)&&(trainErr<convCriterion)) || ((trainErr == 0.0 )&&(valErr == 0.0))

    end

# 	minValErrSynapseMatrix = synapseMatrix
# 	validationErrorVector = trainingErrorVector

    # Construct and return the output tuple.
    outputTuple = Any[net, validationErrorVector, trainingErrorVector, perturbationDistanceVector]

    return(outputTuple)

end

