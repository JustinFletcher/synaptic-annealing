include("propogateForward.jl")

function getDataPreditions(synapseMatrix, actFun, data, inputCols, outputCols)

    outputData = zeros(size(data)[1], size(synapseMatrix)[1])


    # For every observation in the data set.
    for sampleRow in 1:size(data)[1]
       # println(data[sampleRow, inputCols])

        # Sum the differences between the correct output and the NN pattern. Take the abs() and accumulate.
        outputData[sampleRow, :] = propogateForward(data[sampleRow, inputCols], synapseMatrix, actFun)

    end

    # Return the average error.
    return(outputData)
end
