
function synapticAnnealingEpoch(synapseMatrix, temperature, lastError, errorFunction, synapticChangeFcn, synapseChangeInputTuple, trainData, inputCols, outputCols, actFun)

    # Compute the synapse change matrix.
    synapseChange = synapticChangeFcn(synapseMatrix, synapseChangeInputTuple)

    # Modify the synapse matrix using the computed change values.
    synapseMatrix += synapseChange

    # Compute the resultant error.
    perturbedError = errorFunction(synapseMatrix, actFun, trainData, inputCols, outputCols)

    # Computer the change in error.
    errorChange = perturbedError-lastError

    # Compute the acceptance probability for this change.
    acceptanceProbability = minimum([1, exp(-(errorChange/temperature))])

#     println("temperature: ",temperature, "|errorChange: ",errorChange , "|ap: ",acceptanceProbability)

    # With probabability of acceptanceProbability, take the move.
    if (rand()<acceptanceProbability)
        # If the annealing move is not repealed, set the error.
        lastError = perturbedError
    else
        # Repeal the move.
        synapseMatrix -= synapseChange
    end


    return(synapseMatrix, lastError)
end


