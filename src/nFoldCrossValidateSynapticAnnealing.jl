function nFoldCrossValidateSynapticAnnealingPar(numFolds, synMatConfigVec, annealingFunction, convCriterion,
												cutoffEpochs, perturbSynapses, updateState, errorFunction,
												reportErrorFunction, initTemperature, initLearnRate, synMatIn,
												actFun, dataset, batchSize, reportFrequency)


    # Shuffle the data.
    # data = shuffleData(data)

    # Build the folds, and shuffle the data.
    inFoldsVector,  outFoldsVector = buildFolds(dataset.data, numFolds)

    # Initialize the loop variables.
    minValErrorSynapseMatrix = null
    bestValError = Inf
    refList = Any[]

    for fold in 1:numFolds

        # Select the train and validation data from the folds.
        valData  = ExperimentDataset.Dataset(dataset.data[inFoldsVector[fold],:], dataset.inputCols, dataset.outputCols, dataset.name)
        trainData = ExperimentDataset.Dataset(dataset.data[outFoldsVector[fold],:], dataset.inputCols, dataset.outputCols, dataset.name)

        # Initialize a new synapse matrix.
        netIn = init_network(synMatConfigVec)
		    netIn.learning_rate = 1
		    netIn.propagation_function = actFun

        # Spawn an annealing task.
        ref = @spawn annealingFunction(convCriterion, cutoffEpochs, perturbSynapses, updateState,
									   errorFunction, reportErrorFunction, initTemperature, initLearnRate,
									   netIn, actFun, trainData, valData,
						   			 batchSize, reportFrequency)

        # Append the rederence to the remote task to the list.
        push!(refList, ref)

    end

    # Initialize the loop variables.
    refNum = 0
    trainErrorFoldList = Any[]
    valErrorFoldList= Any[]
    perfectClassificationFoldList = Any[]
    synapseMatrixFoldList = Any[]

    # Iterate over each reference.
    for remoteReference in refList

        # Increment the reference count.
        refNum += 1


        # Fetch and unpack the results from the remote job.
        outTuple = fetch(remoteReference)

        # TODO: Make to accept data frame returns.

        (minValErrSynapseMatrix, validationErrorVector, trainingErrorVector, perfectClassificationVector) = outTuple

        # Push the results onto the storage lists.
        push!(trainErrorFoldList, trainingErrorVector)
        push!(valErrorFoldList, validationErrorVector)
        push!(perfectClassificationFoldList, perfectClassificationVector)
        push!(synapseMatrixFoldList, minValErrSynapseMatrix)

    end



    # Compute the fold-mean validation error vector.
    meanValErrorVec = vec(mean(vectorListToMatrix(valErrorFoldList), 1))

    # Compute the fold-mean validation error vector.
    stdValErrorVec = vec(std(vectorListToMatrix(valErrorFoldList), 1))

    # Compute the fold-mean training error vector.
    meanTrainErrorVec = vec(mean(vectorListToMatrix(trainErrorFoldList), 1))

    # Compute the fold-mean training error vector.
    stdTrainErrorVec = vec(std(vectorListToMatrix(trainErrorFoldList), 1))

    # Compute the fold-mean perturbation distance vector.
    perfectClassificationVec = vec(mean(vectorListToMatrix(perfectClassificationFoldList), 1)).*numFolds


    # Return the results as a tuple.
#     return(Any[meanValErrorVec, stdValErrorVec, meanTrainErrorVec, stdTrainErrorVec, meanPerturbationDistanceVec, synapseMatrixFoldList])

    # Return the results as a tuple.
    return(Any[meanValErrorVec, stdValErrorVec, meanTrainErrorVec, stdTrainErrorVec, perfectClassificationVec, synapseMatrixFoldList])
end

