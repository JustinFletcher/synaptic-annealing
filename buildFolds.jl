
function buildFolds(data, numFolds)

    println("*****************New Crossfold Run*********************")

    # Compute the numvewr of samples in this dataset.
    numSamples = size(data)[1]

    # Compute the foldSize.
    foldSize = int64((floor(numSamples)/(numFolds)))

    # Declare vectors which will store the
    inFoldsVector = Any[]
    outFoldsVector = Any[]

    # Initialize loop variables.
    foldIndex=1
    foldsComputed = 0


    while(foldsComputed < numFolds-1)

        # Construct the ranges for the folds
        inFoldRange = [foldIndex:(foldIndex+foldSize)]
        outFoldRange = [1:(foldIndex-1),(foldIndex+foldSize+1):(numSamples-1)]

        # Add the fold ranges to the fold vectors.
        push!(inFoldsVector, inFoldRange)
        push!(outFoldsVector, outFoldRange)

        # Increment loop variables.
        foldIndex += foldSize
        foldsComputed += 1

    end

    # Append the final fold ranges, which require special handling to account for uneven folds.
    push!(inFoldsVector, [foldIndex:numSamples])
    push!(outFoldsVector, [1:foldIndex-1])

    return(inFoldsVector, outFoldsVector)

end
