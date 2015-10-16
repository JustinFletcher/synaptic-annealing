

function getDataClassErr(synapseMatrix, actFun, data, inputCols, outputCols)

    # Get the number of samples.
    #numSamples = size(data)[1]

    # Initialize error to 0.
    err = 0

# 	numOutputs = length(outputCols)

# 	outputPadding = zeros(1, size(synapseMatrix)[1]-numOutputs)

    # For every observation in the data set.
    for sampleRow in 1:size(data)[1]

#         actualOutputRaw = propogateForward(data[sampleRow, inputCols], synapseMatrix, actFun)

#         actualOutput = actualOutputRaw.==maximum(actualOutputRaw)



# 		# Addd -1s here after size for all bias
#         err += transpose(data[sampleRow, outputCols])!=actualOutput[1:numOutputs,:]


		err += indmax(propogateForward(data[sampleRow, inputCols], synapseMatrix, actFun))!=indmax(data[sampleRow, outputCols])

	end

    # Return the average error.
    return(err/size(data)[1])
end

indmax(rand(100))


# temp = rand(1,20)
# tic()
# transpose(temp)
# t1 = toq()
# t1*150*3*1000000/60/60

function getDataClassErrPar(synapseMatrix, actFun, data, inputCols, outputCols)

    # Get the number of samples.
    #numSamples = size(data)[1]

    # Initialize error to 0.
    err = 0
# 	numOutputs = length(outputCols)

# 	outputPadding = zeros(1, size(synapseMatrix)[1]-numOutputs)

    # For every observation in the data set.
    err = @parallel (+) for sampleRow in 1:size(data)[1]

		indmax(propogateForward(data[sampleRow, inputCols], synapseMatrix, actFun))!=indmax(data[sampleRow, outputCols])
        #println("------------------------------")
        #sample = data[sampleRow, :]
        #println(sample)

#         actualOutputRaw = propogateForward(data[sampleRow, inputCols], synapseMatrix, actFun)

#         actualOutput = actualOutputRaw.==maximum(actualOutputRaw)

# 		# Addd -1s here after size for all bias
#         transpose(data[sampleRow, outputCols])!=actualOutput[1:numOutputs,:]

        # Format the data for the arithmatic.
        #correctOutput = int([sample[outputCols], transpose(zeros(1, size(synapseMatrix)[1]-length(sample[outputCols])))])

#         #include("propogateForward.jl")
#         #println("correctOutput ",correctOutput )
#         actualOutputRaw = propogateForward(data[sampleRow, inputCols], synapseMatrix, actFun)

#         actualOutput = int(actualOutputRaw.==maximum(actualOutputRaw))
#         #println("actualOutputRaw",actualOutputRaw)

#           # println("output", output)
#         #   println("output", int(output.==maximum(output)))
# #         println(int(int(paddedData)==int(output.==maximum(output))))

# 		# Addd -1s here after size for all bias
#         int(int(transpose([data[sampleRow, outputCols] zeros(1, size(synapseMatrix)[1]-length(data[sampleRow, outputCols]))]))!=actualOutput)
    end

    # Return the average error.
    return(err/size(data)[1])
end
# To undo the changes for universal viasing, remove the 1s.

function getDataRegErr(synapseMatrix, actFun, data, inputCols, outputCols)


    # Get the number of samples.
    #numSamples = size(data)[1]

    err = 0

    # For every observation in the data set.
    for sampleRow in 1:size(data)[1]

        #sample = data[sampleRow, :]

        # Format the data for the arithmatic.
		# Addd -1s here after size for all bias
        paddedData = transpose([data[sampleRow, outputCols] (zeros(1, size(synapseMatrix)[1]-length(data[sampleRow, outputCols])))])

# 		println(paddedData)
# 		println(propogateForward(data[sampleRow, inputCols], synapseMatrix, actFun))
        # Sum the differences between the correct output and the NN pattern. Take the abs() and accumulate.
        err += sqrt(sum((paddedData-propogateForward(data[sampleRow, inputCols], synapseMatrix, actFun)).^2))^2

    end

    # Return the average error.
    return(err/size(data)[1])
end
