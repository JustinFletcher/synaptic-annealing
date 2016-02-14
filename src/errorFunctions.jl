
function getDataClassSTUNErr(net, dataset, state, batchSize)

    # Initialize error to 0.
    err = 0

    # For every observation in the data set.
    for sampleRow in (1:size(dataset.data)[1])[(vec(1:size(dataset.data)[1])[randperm(length(1:size(dataset.data)[1]))[1:min(batchSize, length(1:size(dataset.data)[1]))]])]

		err += indmax(transpose(net_eval(net, vec(dataset.data[sampleRow, dataset.inputCols]))))!=indmax(dataset.data[sampleRow, dataset.outputCols])

	end

    # Return the average error.

    return((1-exp(-0.0000001*(err-state.minTrainError))))
end


function getDataClassErr(net, dataset, state, batchSize)

    # Initialize error to 0.
    err = 0

	# Get random subsed of the row indexes of the dataset.
# 	(size(dataset.data)[1])[randperm(length(size(dataset.data)[1]))[1:batchSize]]
    # For every observation in the data set.
# 	println("a")
# 	(1:(size(dataset.data)[1]))[10, 20]
# 	println("b")



# 	println((1:(size(dataset.data)[1]))[randperm(length(1:(size(dataset.data)[1])))[1:batchSize]])
# 	a = 1:size(dataset.data)[1]

    for sampleRow in (1:size(dataset.data)[1])[(vec(1:size(dataset.data)[1])[randperm(length(1:size(dataset.data)[1]))[1:min(batchSize, length(1:size(dataset.data)[1]))]])]

        err += indmax(transpose(net_eval(net, vec(dataset.data[sampleRow, dataset.inputCols]))))!=indmax(dataset.data[sampleRow, dataset.outputCols])

    end

    # Return the average error.
    return(err)
end



# function getDataClassErr(net, dataset, state, batchSize)

#     # Initialize error to 0.
#     err = 0

#     # For every observation in the data set.
#     for sampleRow in 1:size(dataset.data)[1]

# 		err += indmax(transpose(net_eval(net, vec(dataset.data[sampleRow, dataset.inputCols]))))!=indmax(dataset.data[sampleRow, dataset.outputCols])

# 	end

#     # Return the average error.
#     return(err/size(dataset.data)[1])
# end


function getDataClassErr_backprop(net, dataset, state, batchSize)

    # Initialize error to 0.
    err = 0

    # For every observation in the data set.
    for sampleRow in 1:size(dataset.data)[1]

		err += indmax(net_eval(net, vec(dataset.data[sampleRow, dataset.inputCols])))!=indmax(dataset.data[sampleRow, dataset.outputCols])

	end

    # Return the average error.
    return(err/size(dataset.data)[1])
end




function getDataClassErrPar(synapseMatrix, actFun, dataset, state)

    # Initialize error to 0.
    err = 0

    # For every observation in the data set.
    err = @parallel (+) for sampleRow in 1:size(dataset.data)[1]
		indmax(net_eval(net, vec(dataset.data[sampleRow, dataset.inputCols])))!=indmax(dataset.data[sampleRow, dataset.outputCols])

    end
    # Return the average error.
    return(err/size(dataset.data)[1])
end

# function getDataRegErr(net, dataset, state, batchSize)


#     err = 0

#     # For every observation in the data set.
#     for sampleRow in (1:size(dataset.data)[1])[(vec(1:size(dataset.data)[1])[randperm(length(1:size(dataset.data)[1]))[1:min(batchSize, length(1:size(dataset.data)[1]))]])]

# 		    err += (sqrt(sum((dataset.data[sampleRow, dataset.outputCols]-transpose(net_eval(net, vec(dataset.data[sampleRow, dataset.inputCols])))).^2))).^2
#     end

#     # Return the average error.
#     return(err)
# end

function getDataRegErr(net, dataset, state, batchSize)


    err = 0

    # For every observation in the data set.
    for sampleRow in (1:size(dataset.data)[1])[(vec(1:size(dataset.data)[1])[randperm(length(1:size(dataset.data)[1]))[1:min(batchSize, length(1:size(dataset.data)[1]))]])]

		    #err += sum(abs(dataset.data[sampleRow, dataset.outputCols].-transpose(net_eval(net, vec(dataset.data[sampleRow, dataset.inputCols])))))
        err += sqrt(sum(((net_eval(net, vec(dataset.data[sampleRow, dataset.inputCols])))-transpose(dataset.data[sampleRow, dataset.outputCols])).^2)).^2
    end

    # Return the average error.
    return(err)
end
