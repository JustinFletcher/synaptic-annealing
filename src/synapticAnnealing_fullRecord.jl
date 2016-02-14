

# @everywhere include("$(pwd())\\src\\"*"AnnealingState.jl")
# state = AnnealingState.State(100,0,0,0)
# state.temperature
# temp = AnnealingState.updateState_classical_unit_cooling
# temp(state)
# state.temperature
# state.normTemperature

function synapticAnnealing_fullRecord(convCriterion, cutoffEpochs, perturbSynapses, updateState,
						   errorFunction, reportErrorFunction, initTemperature, initLearnRate,
						   netIn, actFun, trainData, valData,
						   batchSize, reportFrequency)

    println("New Synaptic Annealing")


    # Create a local copy of the synapse matrix.
    network = netIn

    # Initialize the error vectors.
    trainingErrorVector = Float64[]
    validationErrorVector = Float64[]
    onlineErrorVector = Float64[]
    perfectClassificationVector = Float64[]
    minValErrSynapseMatrix = Any[]
    synapseMatrixVector = Any[]
    moveAcceptanceVector = Any[]

	  # Initialize the state.
    maxConfigDist = 2*sum(getNetworkSynapseMatrix(network).!=0)
	  state = AnnealingState.State(initTemperature,initLearnRate,1,maxConfigDist)

    # Initialize the error.
    lastError = errorFunction(network, trainData, state, batchSize)
    state.minValError = Inf
    # Initialize the loop control variable.
    converged = false

    while !converged

      if ((state.epochsComplete%1)==0)

        # Push the most recent error onto the error vector.
        state.trainError = reportErrorFunction(network, trainData, state, size(trainData.data)[1])
        push!(trainingErrorVector, state.trainError./size(trainData.data)[1])

        # Push the validation error set onto the vector.
        state.valError = reportErrorFunction(network, valData, state, size(valData.data)[1])
        push!(validationErrorVector, state.valError./size(valData.data)[1])
        if ((state.epochsComplete%reportFrequency)==0)
            println("E @" * string(state.epochsComplete) * ": " * string(state.trainError./size(trainData.data)[1])  * " | " * string(state.valError./size(valData.data)[1]) )
        end

      end

      # State capture: if this is the best net so far, save it to the disk.
      if state.valError < state.minValError
        state.minValError = state.valError
        minValErrSynapseMatrix = network
      end

      # Update the state of the annealing system.
      updateState(state)

      # Parse the synapse matrix from the network.
      synapseMatrix = getNetworkSynapseMatrix(network)

      # Compute the synapse perturbation matrix.
      synapsePerturbationTuple = perturbSynapses(synapseMatrix, state)

      # Parse the perturbation tuple. For readability.
      (synapsePerturbation, perturbationDistance) = synapsePerturbationTuple

      # Append the perturbation distance to an output vector. For analysis.
      push!(perfectClassificationVector, int((state.trainError == 0.0 )&&(state.valError == 0.0)))

      # Modify the synapse matrix using the perturbation matrix.
      synapseMatrix += synapsePerturbation
      push!(synapseMatrixVector, synapseMatrix)


      # Compute the resultant error.
      perturbedError = errorFunction(setNetworkSynapseMatrix(network, synapseMatrix), trainData, state, batchSize)

      # Push a record of this online error to the storage vector.
      push!(onlineErrorVector, perturbedError./size(trainData.data)[1])

      # Computer the change in error.
      errorChange = perturbedError-lastError


      # If this is a downhill move, take it.
      if (perturbedError<=lastError)
        lastError = perturbedError
        push!(moveAcceptanceVector,1)
        # If this is not a downhill or neural move...
      else
        # Then with probability exp(-DeltaE/T), or if it's the result of a tunnelling event, reject the move.
        if( (rand()>=exp(-(errorChange)/state.temperature)) )#|| (perturbationDistance>state.learnRate) )
          synapseMatrix -= synapsePerturbation

          push!(moveAcceptanceVector,0)
          # If the uphill move is not rejected, set the error.
        else
          push!(moveAcceptanceVector,1)
          lastError = perturbedError
        end

      end

      # State capture: if this is the best net so far, save it to the disk.
      if perturbedError < state.minTrainError
        state.minTrainError = perturbedError
      end

      # Reconstruct the network from the latet synapse matrix.
      network = setNetworkSynapseMatrix(network, synapseMatrix)

      # If this run is perfect, tell the user.
      if((state.trainError == 0.0 )&&(state.valError == 0.0))
        println("Perfect Run")
      end

      # Evaluate the convergence conditions.
      converged = (state.epochsComplete>cutoffEpochs) || ((state.trainError == 0.0 )&&(state.valError == 0.0))

    end

    # TODO: Make this a data frame return.

    # Construct and return the output tuple.
    outputTuple = Any[minValErrSynapseMatrix, validationErrorVector, trainingErrorVector, perfectClassificationVector, onlineErrorVector, synapseMatrixVector, moveAcceptanceVector]
    return(outputTuple)

end


