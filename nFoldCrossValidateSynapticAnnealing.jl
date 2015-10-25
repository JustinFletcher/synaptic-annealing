function nFoldCrossValidateSynapticAnnealingPar(numFolds, synMatConfigVec, annealingFunction, convCriterion, cutoffEpochs, perturbSynapses, updateState, errorFunction, reportErrorFunction, initTemperature, initLearnRate, synMatIn, actFun, data, inputCols, outputCols)


    # Shuffle the data.
    data = shuffleData(data)

    # Build the folds, and shuffle the data.
    inFoldsVector,  outFoldsVector = buildFolds(data, numFolds)

    # Initialize the loop variables.
    minValErrorSynapseMatrix = null
    bestValError = Inf
    refList = Any[]

    for fold in 1:numFolds

        # Select the train and validation data from the folds.
        valData  = data[inFoldsVector[fold],:]
        trainData = data[outFoldsVector[fold],:]

        # Initialize a new synapse matrix.
        netIn = init_network(synMatConfigVec)
		netIn.learning_rate = 1
		netIn.propagation_function = tanh

        # Spawn an annealing task.
        ref = @spawn annealingFunction(convCriterion, cutoffEpochs, perturbSynapses, updateState, errorFunction, reportErrorFunction, initTemperature, initLearnRate, netIn, actFun, trainData, valData, inputCols, outputCols)

        # Append the rederence to the remote task to the list.
        push!(refList, ref)

    end

    # Initialize the loop variables.
    refNum = 0
    trainErrorFoldList = Any[]
    valErrorFoldList= Any[]
    perturbationDistanceFoldList = Any[]
    synapseMatrixFoldList = Any[]

    # Iterate over each reference.
    for remoteReference in refList

        # Increment the reference count.
        refNum += 1


        # Fetch and unpack the results from the remote job.
        outTuple = fetch(remoteReference)

        (minValErrSynapseMatrix, validationErrorVector, trainingErrorVector, perturbationDistanceVector) = outTuple

        # Push the results onto the storage lists.
        push!(trainErrorFoldList, trainingErrorVector)
        push!(valErrorFoldList, validationErrorVector)
        push!(perturbationDistanceFoldList, perturbationDistanceVector)
        push!(synapseMatrixFoldList, minValErrSynapseMatrix)

    end

    # Compute the fold-mean validation error vector.
    meanValErrorVec = vec(mean(vectorListToMatrix(valErrorFoldList), 1))

    # Compute the fold-mean training error vector.
    meanTrainErrorVec = vec(mean(vectorListToMatrix(trainErrorFoldList), 1))

    # Compute the fold-mean perturbation distance vector.
    meanPerturbationDistanceVec = 0

    # Return the results as a tuple.
    return(Any[meanValErrorVec, meanTrainErrorVec, meanPerturbationDistanceVec, synapseMatrixFoldList])
end

