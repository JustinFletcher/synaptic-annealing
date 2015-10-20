
function synapticAnnealing_class(convCriterion, cutoffEpochs, perturbSynapses, updateState, errorFunction, reportErrorFunction, initTemperature, initLearnRate, synapseMatrixIn, actFun, trainData, valData, inputCols, outputCols)

    println("New Synaptic Annealing Run")

#     Parse the input data tuple into train and val tuples.
#     (trainData, valData, inputCols, outputCols) = dataTuple

    # Create a local copy of the synapse matrix.
    synapseMatrix = copy(synapseMatrixIn)

    numLayers = size(synapseMatrix)[3]-1
    sizeSynapseMatrix = size(synapseMatrix)[1]
    numTrainSamples = size(trainData)[1]
    numValSamples = size(valData)[1]

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


    #
    err = 0


    while !converged

        # Push the most recent error onto the error vector.
#         trainErr = reportErrorFunction(synapseMatrix, actFun, trainData, inputCols, outputCols)
        err = 0

        propPattern = zeros(1, sizeSynapseMatrix)

        initialPropOffset = 1+length(inputCols)

        for sampleRow in 1:size(trainData)[1]

            # Initialize the propigation pattern to the input pattern. Pad the propPattern with zeros to size with weightMatrix.
            propPattern[1] = 1
            propPattern[2:initialPropOffset] = trainData[sampleRow, inputCols]

            # For each "layer" of connection weights, except the last one, which we just want to "read."
            for layerIndex in 1:numLayers

                propPattern = tanh(sum((transpose(propPattern) .* synapseMatrix[:,:,layerIndex]), 1))

            end

            err += indmax(transpose(propPattern))!=indmax(trainData[sampleRow, outputCols])

        end
        # Return the average error.
        trainErr = err/numTrainSamples
        push!(trainingErrorVector, trainErr)

        # Push the validation error set onto the vector.
#         valErr = reportErrorFunction(synapseMatrix, actFun, valData, inputCols, outputCols)
        err = 0

        propPattern = zeros(1, sizeSynapseMatrix)

        initialPropOffset = 1+length(inputCols)

        for sampleRow in 1:size(valData)[1]

            # Initialize the propigation pattern to the input pattern. Pad the propPattern with zeros to size with weightMatrix.
            propPattern[1] = 1
            propPattern[2:initialPropOffset] = valData[sampleRow, inputCols]

            # For each "layer" of connection weights, except the last one, which we just want to "read."
            for layerIndex in 1:numLayers

                propPattern = tanh(sum((transpose(propPattern) .* synapseMatrix[:,:,layerIndex]), 1))

            end

            err += indmax(transpose(propPattern))!=indmax(valData[sampleRow, outputCols])

        end
        # Return the average error.
        valErr = err/numTrainSamples
        push!(validationErrorVector, valErr)

        # State capture: if this is the best net so far, save it to the disk.
        if valErr < minError
            minValErrSynapseMatrix = synapseMatrix
        end

        # Update the state of the annealing system.
        stateTuple = updateState(stateTuple)

        # Compute the synapse perturbation matrix.
        synapsePerturbationTuple = perturbSynapses(synapseMatrix, stateTuple)

        # Parse the perturbation tuple. For readability.
        (synapsePerturbation, perturbationDistance) = synapsePerturbationTuple

        # Append the perturbation distance to an output vector. For analysis.
        push!(perturbationDistanceVector, perturbationDistance)

        # Modify the synapse matrix using the perturbation matrix.
        synapseMatrix += synapsePerturbation



        # Compute the resultant error.
#         perturbedError = errorFunction(synapseMatrix, actFun, trainData, inputCols, outputCols)


        err = 0

        propPattern = zeros(1, sizeSynapseMatrix)

        initialPropOffset = 1+length(inputCols)

        for sampleRow in 1:size(trainData)[1]

            # Initialize the propigation pattern to the input pattern. Pad the propPattern with zeros to size with weightMatrix.
            propPattern[1] = 1
            propPattern[2:initialPropOffset] = trainData[sampleRow, inputCols]

            # For each "layer" of connection weights, except the last one, which we just want to "read."
            for layerIndex in 1:numLayers

                propPattern = tanh(sum((transpose(propPattern) .* synapseMatrix[:,:,layerIndex]), 1))

            end

            err += indmax(transpose(propPattern))!=indmax(trainData[sampleRow, outputCols])

        end
        # Return the average error.
        perturbedError = err/numTrainSamples

        # Computer the change in error.
        errorChange = perturbedError-lastError

        # Parse the temperature from the state tuple and normalize. For readability.
        temperatureNorm = stateTuple[1]/initTemperature


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

