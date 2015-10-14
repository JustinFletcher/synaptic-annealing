using StatsBase
using PyPlot

# ---- Utility Functions  ----

function putdata(data, idStr)
    # This function is a convienience interface to the serialize package.
    out = open(idStr,"w")
    serialize(out,data);
    close(out)
end

function getdata(idStr)
    # This function is a convienience interface to the serialize package.
    in = open(idStr,"r")
    data = deserialize(in)
    close(in)
    return(data)
end

# ---- Dataset Construction Function Declarations ----

function inverse(x)
  return(1./(x))
end

function constructInverseDataset(inverseDataRange, trainToTestRatio)

    # Construct the data.
    inverseData = [inverseDataRange inverse(inverseDataRange)]

    # Randomly partition the data set into test and train subsets. Start by shuffling the data.
    randIndVec = sample(1:size(inverseData)[1], size(inverseData)[1])
    inverseData = inverseData[randIndVec, :]

    # Now select a subset of the data to be train and test.
    inverseDataTrain = inverseData[1:int64(floor(size(inverseData)[1]*trainToTestRatio)), :]
    inverseDataTest = inverseData[int64(floor(size(inverseData)[1]*trainToTestRatio))+1:end, :]

    return(inverseDataTrain,inverseDataTest)

end

function normalizeData(dataIn)
  data = copy(dataIn)
  for col in 1:size(data)[2]
    data[:,col] = ((data[:,col]./maximum(data[:,col])))
  end
  return(data)
end


# ---- Generate the Iris Dataset ----

irisData = readdlm(".\\iris.dat", ',' , Any)

function uniqueElements(vector)
    knownList = Array(Any, 0)
    for element in vector
        if (! in(element, knownList))
            knownList=vcat(knownList, element)
        end
    end
    return(knownList)
end

function transformClassData(data,classCol)

    colVector  = (1:size(data)[2])[(1:size(data)[2]).!=classCol]

    transformedData = copy(data)
    transformedData = transformedData[:, colVector]
    # Get each unique class value
    for class in uniqueElements(data[:, classCol])
        transformedData = [transformedData int(data[:, classCol] .== class)]
    end
    # Build Boolean (1, 0 encoded) vector representing class membership for each class
    # Remove the original class Column
    # Append the new class membership columns
    return(float64(transformedData))
end

irisDataClassed = transformClassData(irisData, [5])




# ---- Synapse Matrix Creation Function ----

function createSynapseMatrix(layerSpecVector)

    # Get the number of nodes in the largest layer.
    maxNumNodes = maximum(layerSpecVector)

    # Get the number of layers.
    numLayers = length(layerSpecVector)

    # Initialize an empty array to contain the weights.
    weightsArray = zeros(maxNumNodes, maxNumNodes, numLayers)

    for layer in 1:numLayers-1
        thisLayerNumNeurons = layerSpecVector[layer]

        nextLayerNumNeurons = layerSpecVector[layer+1]

        weightsArray[1:thisLayerNumNeurons, 1:nextLayerNumNeurons, layer] = 0.2*(rand(thisLayerNumNeurons, nextLayerNumNeurons)-0.5)
    end

    return(weightsArray)
end

# ---- Performance Calculation Function ----

function propogateForward(inputPattern, synapseMatrix, actFun)

    # Initialize the propigation pattern to the input pattern. Pad the propPattern with zeros to size with weightMatrix.
    propPattern = [1 [actFun(inputPattern) zeros(1, size(synapseMatrix)[1]-length(inputPattern)-1)]]
    # For each "layer" of connection weights, except the last one, which we just want to "read."
    for layerIndex in 1:size(synapseMatrix)[3]-1
        # Caluculate the sum along the columns of the propPattern distributed over the synapseMatrix.
        propPattern = actFun(sum((transpose(propPattern) .* synapseMatrix[:,:,layerIndex]), 1))
    end

    # Return the final propigation pattern, which is the output.
    return(transpose(propPattern))

end

# ---- Temperature Schedule Function ----

function sigmoidTemperatureSchedule(numEpochs)

    k=0.9
    x0=12.5
    return(1-(1./(1+exp(-k .* (20*((1:numEpochs)./numEpochs).-x0)    ))))
end

function linearTemperatureSchedule_eightTenths(numEpochs)
    lineValues = (1-(1:numEpochs)./(0.8*numEpochs))
    return( lineValues .- (lineValues .* int(lineValues.<0)) )
end

function linearTemperatureSchedule_sixTenths(numEpochs)
    lineValues = (1-(1:numEpochs)./(0.6*numEpochs))
    return( lineValues .- (lineValues .* int(lineValues.<0)) )
end

plot(sigmoidTemperatureSchedule(1000))
plot(linearTemperatureSchedule_sixTenths(1000))


# ---- Decay Functions ----

function expDecay(x)
    return(0.1+(0.5*exp(-(0.1*x))))
end

function constVal(x)
    return(10)
end

function sigmoidDecay_Tuned(epoch)

    k=3
    x0=8
    decayVal = (1- 0.99*(1./(1+exp(-k .* (10.*(epoch)-x0)))) )
    return(decayVal)
end

plot(sigmoidDecay_Tuned(0:0.001:1))
# ---- Error Calculation Functions ----



function getDataClassErrBack(synapseMatrix, actFun, data, inputCols, outputCols)

    # Get the number of samples.
    numSamples = size(data)[1]

    # Initialize error to 0.
    err = 0

    # For every observation in the data set.
    for sampleRow in 1:numSamples
        #println("------------------------------")
        sample = data[sampleRow, :]
        #println(sample)

        # Format the data for the arithmatic.
        correctOutput = int([sample[outputCols], transpose(zeros(1, size(synapseMatrix)[1]-length(sample[outputCols])))])

        #println("correctOutput ",correctOutput )
        actualOutputRaw = propogateForward(transpose(sample[inputCols]), synapseMatrix, actFun)

        actualOutput = int(actualOutputRaw.==maximum(actualOutputRaw))
        #println("actualOutputRaw",actualOutputRaw)

          # println("output", output)
        #   println("output", int(output.==maximum(output)))
#         println(int(int(paddedData)==int(output.==maximum(output))))

        err += int(correctOutput!=actualOutput)
    end

    # Return the average error.
    return(err/numSamples)
end

function getDataClassErr(synapseMatrix, actFun, data, inputCols, outputCols)

    # Get the number of samples.
    #numSamples = size(data)[1]

    # Initialize error to 0.
    err = 0

    # For every observation in the data set.
    for sampleRow in 1:size(data)[1]
        #println("------------------------------")
        #sample = data[sampleRow, :]
        #println(sample)

        # Format the data for the arithmatic.
        #correctOutput = int([sample[outputCols], transpose(zeros(1, size(synapseMatrix)[1]-length(sample[outputCols])))])

        #println("correctOutput ",correctOutput )
        actualOutputRaw = propogateForward(data[sampleRow, inputCols], synapseMatrix, actFun)

        actualOutput = int(actualOutputRaw.==maximum(actualOutputRaw))
        #println("actualOutputRaw",actualOutputRaw)

          # println("output", output)
        #   println("output", int(output.==maximum(output)))
#         println(int(int(paddedData)==int(output.==maximum(output))))

        err += int(int(transpose([data[sampleRow, outputCols] zeros(1, size(synapseMatrix)[1]-length(data[sampleRow, outputCols]))]))!=actualOutput)
    end

    # Return the average error.
    return(err/size(data)[1])
end

function getDataRegErrBack(synapseMatrix, actFun, data, inputCols, outputCols)


    # Get the number of samples.
    numSamples = size(data)[1]
    err = 0

    # For every observation in the data set.
    for sampleRow in 1:numSamples

        sample = data[sampleRow, :]

        # Format the data for the arithmatic.
        paddedData = [sample[outputCols], transpose(zeros(1, size(synapseMatrix)[1]-length(sample[outputCols])))]

        # Sum the differences between the correct output and the NN pattern. Take the abs() and accumulate.
        err += sqrt(sum((paddedData-propogateForward(transpose(sample[inputCols]), synapseMatrix, actFun)).^2))^2

    end

    # Return the average error.
    return(err/numSamples)
end

function getDataRegErr(synapseMatrix, actFun, data, inputCols, outputCols)


    # Get the number of samples.
    #numSamples = size(data)[1]

    err = 0

    # For every observation in the data set.
    for sampleRow in 1:size(data)[1]

        #sample = data[sampleRow, :]

        # Format the data for the arithmatic.
        paddedData = transpose([data[sampleRow, outputCols] (zeros(1, size(synapseMatrix)[1]-length(data[sampleRow, outputCols])))])

        # Sum the differences between the correct output and the NN pattern. Take the abs() and accumulate.
        err += sqrt(sum((paddedData-propogateForward(data[sampleRow, inputCols], synapseMatrix, actFun)).^2))^2

    end

    # Return the average error.
    return(err/size(data)[1])
end






# ---- Annealing Training Function ----

function annealSynapses_allWeight_fixedStep_Quantum(numEpochs, errorFunction, reportErrorFunction, temperatureScheduleFcn, learnRateDecay, synapseMatrixIn, actFun, trainData, valData, inputCols, outputCols)

    # Create a local copy of the synapse matrix.
    synapseMatrix = copy(synapseMatrixIn)

    # Initialize the error.
    lastError = errorFunction(synapseMatrix, actFun, trainData, inputCols, outputCols)

    # Compute the temperature schedule.
    temperatureSchedule = temperatureScheduleFcn(numEpochs)

    # Initialize the error vectors.
    trainingErrorVector = Float64[]
    validationErrorVector = Float64[]

    for epoch in 1:numEpochs

        # Push the most recent error onto the error vector.
        push!(trainingErrorVector, reportErrorFunction(synapseMatrix, actFun, trainData, inputCols, outputCols))

        # Push the validation error set onto the vector.
        push!(validationErrorVector, reportErrorFunction(synapseMatrix, actFun, valData, inputCols, outputCols))

        # Select the appropriate temperature value for this epoch.
        temperature = temperatureSchedule[epoch]


        #stepSize = 1+(100*int(rand()<((1-learnRateDecay(epoch/numEpochs)/100))))

        # Construct a matrix of values which will modify the weights of the synapses.
        # Scalar Multiple .* Random Matrix Intersect Exists Synapse Which Adds To One .* Random Negator
        #randMat = rand(size(synapseMatrix)).*int(bool(synapseMatrix))
        #synapseChange = stepSize.*((randMat./(sum(randMat))).*((2*int(rand(size(synapseMatrix)).>0.5))-1))


        function synapticChange(synapseMatrix, learnRateDecay, epoch, numEpochs)

            stepSize = 1+(100*rand()*int(rand()<(0.1*(1-(learnRateDecay(epoch/numEpochs))))))
            randMat = rand(size(synapseMatrix)).*int(bool(synapseMatrix))
            synapseChange = stepSize.*((randMat./(sum(randMat))).*((2*int(rand(size(synapseMatrix)).>0.5))-1))
            return(synapseChange)
        end


        synapseChange = synapticChange(synapseMatrix, learnRateDecay, epoch, numEpochs)

        # Here add Small probability of increasing the multiple by 100


        # Multipy by the learn rate.
        # Mask by the existance of connections.

        # Randomly step in each dimension.
        synapseMatrix += synapseChange

        # Compute the error for this random step.
        thisError = errorFunction(synapseMatrix, actFun, trainData, inputCols, outputCols)

        # If the error has increased, then with probability=temperature, allow this move.
        if( (thisError>lastError) && (rand()>temperature) )

            # Repeal the move.
            synapseMatrix -= synapseChange

        else

            # If the annealing move is not repealed, set the error.
            lastError = thisError

        end

    end

    # Return the annealed synapse matrix and the error vector.
    return(synapseMatrix, trainingErrorVector, validationErrorVector)

end

function annealSynapses_allWeight_fixedStep(numEpochs, errorFunction, reportErrorFunction, temperatureScheduleFcn, learnRateDecay, synapseMatrixIn, actFun, trainData, valData, inputCols, outputCols)

    # Create a local copy of the synapse matrix.
    synapseMatrix = copy(synapseMatrixIn)

    # Initialize the error.
    lastError = errorFunction(synapseMatrix, actFun, trainData, inputCols, outputCols)

    # Compute the temperature schedule.
    temperatureSchedule = temperatureScheduleFcn(numEpochs)

    # Initialize the error vectors.
    trainingErrorVector = Float64[]
    validationErrorVector = Float64[]

    for epoch in 1:numEpochs

        # Push the most recent error onto the error vector.
        push!(trainingErrorVector, reportErrorFunction(synapseMatrix, actFun, trainData, inputCols, outputCols))

        # Push the validation error set onto the vector.
        push!(validationErrorVector, reportErrorFunction(synapseMatrix, actFun, valData, inputCols, outputCols))

        # Select the appropriate temperature value for this epoch.
        temperature = temperatureSchedule[epoch]

        stepSize = learnRateDecay(epoch/numEpochs)

        # Construct a matrix of values which will modify the weights of the synapses.
        # Scalar Multiple .* Random Matrix Intersect Exists Synapse Which Adds To One .* Random Negator
        randMat = rand(size(synapseMatrix)).*int(bool(synapseMatrix))
        synapseChange = stepSize.*((randMat./(sum(randMat))).*((2*int(rand(size(synapseMatrix)).>0.5))-1))

        # Here add Small probability of increasing the multiple by 100


        # Multipy by the learn rate.
        # Mask by the existance of connections.

        # Randomly step in each dimension.
        synapseMatrix += synapseChange

        # Compute the error for this random step.
        thisError = errorFunction(synapseMatrix, actFun, trainData, inputCols, outputCols)

        # If the error has increased, then with probability=temperature, allow this move.
        if( (thisError>lastError) && (rand()>temperature) )

            # Repeal the move.
            synapseMatrix -= synapseChange

        else

            # If the annealing move is not repealed, set the error.
            lastError = thisError

        end

    end

    # Return the annealed synapse matrix and the error vector.
    return(synapseMatrix, trainingErrorVector, validationErrorVector)

end

function annealSynapses_allWeight(numEpochs, errorFunction, reportErrorFunction, temperatureScheduleFcn, learnRateDecay, synapseMatrixIn, actFun, trainData, valData, inputCols, outputCols)

    # Create a local copy of the synapse matrix.
    synapseMatrix = copy(synapseMatrixIn)

    # Initialize the error.
    lastError = errorFunction(synapseMatrix, actFun, trainData, inputCols, outputCols)

    # Compute the temperature schedule.
    temperatureSchedule = temperatureScheduleFcn(numEpochs)

    # Initialize the error vectors.
    trainingErrorVector = Float64[]
    validationErrorVector = Float64[]

    for epoch in 1:numEpochs

        # Push the most recent error onto the error vector.
        push!(trainingErrorVector, reportErrorFunction(synapseMatrix, actFun, trainData, inputCols, outputCols))

        # Push the validation error set onto the vector.
        push!(validationErrorVector, reportErrorFunction(synapseMatrix, actFun, valData, inputCols, outputCols))

        # Select the appropriate temperature value for this epoch.
        temperature = temperatureSchedule[epoch]



        # Construct a matrix of values which will modify the weights of the synapses.
        synapseChange = learnRateDecay(epoch/numEpochs).*(2*(rand(size(synapseMatrix))-0.5)).*int(bool(synapseMatrix))

        # Create a random matrix adding to one.

        # Muliply by a scale factor, which is the stepSize parameter.
        # Randomly negate half the elements.


        # Multipy by the learn rate.
        # Mask by the existance of connections.

        # Randomly step in each dimension.
        synapseMatrix += synapseChange

        # Compute the error for this random step.
        thisError = errorFunction(synapseMatrix, actFun, trainData, inputCols, outputCols)

        # If the error has increased, then with probability=temperature, allow this move.
        if( (thisError>lastError) && (rand()>temperature) )

            # Repeal the move.
            synapseMatrix -= synapseChange

        else

            # If the annealing move is not repealed, set the error.
            lastError = thisError

        end

    end

    # Return the annealed synapse matrix and the error vector.
    return(synapseMatrix, trainingErrorVector, validationErrorVector)

end


function annealSynapses_singleWeight(numEpochs, errorFunction, reportErrorFunction, temperatureScheduleFcn, learnRateDecay, synapseMatrixIn, actFun, trainData, valData, inputCols, outputCols)

    # Create a local copy of the synapse matrix.
    synapseMatrix = copy(synapseMatrixIn)

    # Initialize the error.
    lastError = errorFunction(synapseMatrix, actFun, trainData, inputCols, outputCols)

    # Compute the temperature schedule.
    temperatureSchedule = temperatureScheduleFcn(numEpochs)

    # Initialize the error vectors.
    trainingErrorVector = Float64[]
    validationErrorVector = Float64[]

    for epoch in 1:numEpochs

        # Push the most recent error onto the error vector.
        push!(trainingErrorVector,   reportErrorFunction(synapseMatrix, actFun, trainData, inputCols, outputCols))

        # Push the validation error set onto the vector.
        push!(validationErrorVector, reportErrorFunction(synapseMatrix, actFun, valData, inputCols, outputCols))

        # Select the appropriate temperature value for this epoch.
        temperature = temperatureSchedule[epoch]

        # Calculate the update value.
        #updateWeightChange = learnRateDecay(epoch/numEpochs).*(2*(rand()-0.5))

        randMat = (rand(size(synapseMatrix))).*int(bool(synapseMatrix))
        synapseChange = learnRateDecay(epoch/numEpochs).*int(randMat.==(maximum(randMat))).*(2*(rand()-0.5))
        # Select a random synapse to modify.
        #updateWeightLocation = [rand(0:size(synapseMatrix)[1]),rand(1:size(synapseMatrix)[2]),rand(1:size(synapseMatrix)[3])]

        # Randomly select a weight and modify it by updateStep.
        synapseMatrix += synapseChange


        # Compute the error for this random step.
        thisError = errorFunction(synapseMatrix, actFun, trainData, inputCols, outputCols)

        # If the error has increased, then with probability=temperature, allow this move. The not inverts the choice to form a repeal.
        if( (thisError>lastError) && (rand()>temperature) )

            # Repeal the move.
            synapseMatrix -= synapseChange

        else

            # If the annealing move is not repealed, set the error.
            lastError = thisError

        end

    end

    # Return the annealed synapse matrix and the error vector.
    return(synapseMatrix, trainingErrorVector, validationErrorVector)

end







# ---- Cross Validation Functions ----



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

function shuffleData(data)
    return(data[sample(1:size(data)[1], size(data)[1]), :])
end

function nFoldCrossValidateSynapticAnnealing(numFolds, numEpochs, synMatConfigVec, annealFunction, errorFunction, reportErrorFunction,  tempSched, learnRateDecay, actFun, data, inputCols, outputCols)

    # Shuffle the data.
    data = shuffleData(data)

    # Build the folds, and shuffle the data.
    inFoldsVector, outFoldsVector = buildFolds(data, numFolds)

    # Initialize the loop variables.
    trainErrorFoldMatrix = zeros(numEpochs, numFolds)
    valErrorFoldMatrix = zeros(numEpochs, numFolds)
    synapseMatrixFoldMatrix = zeros(maximum(synMatConfigVec),maximum(synMatConfigVec),length(synMatConfigVec),numFolds)
    minValErrorSynapseMatrix = null
    bestValError = Inf

    for fold in 1:numFolds

        # Select the train and validation data from the folds.
        valData  = data[inFoldsVector[fold],:]
        trainData = data[outFoldsVector[fold],:]

        # Initialize a new synapse matrix.
        synapseMatrix = createSynapseMatrix(synMatConfigVec)

        # Anneal the synapse matrix.
        synapseMatrixOut, trainErrorVec, valErrorVec = annealFunction(numEpochs,  errorFunction, reportErrorFunction,  tempSched, learnRateDecay, synapseMatrix, tanh, trainData, valData, inputCols, outputCols)

        # Append the trainErrorVec and valErrorVec to the loop storage variables for each.
        trainErrorFoldMatrix[:, fold] = trainErrorVec
        valErrorFoldMatrix[:, fold] = valErrorVec

        synapseMatrixFoldMatrix[:,:,:, fold] = synapseMatrixOut

        if(valErrorVec[end] < bestValError)
            bestValError = valErrorVec[end]
            minValErrorSynapseMatrix = synapseMatrixOut
        end
    end

    minValErrorSynapseMatrix = synapseMatrixFoldMatrix

    # Compute the fold-mean validation error vector.
    meanValErrorVec = mean(valErrorFoldMatrix, 2)

    # Compute the fold-mean training error vector.
    meanTrainErrorVec = mean(trainErrorFoldMatrix, 2)

    # Select the synapse matrix which achieved the min validation error.
    return(meanValErrorVec, meanTrainErrorVec, minValErrorSynapseMatrix)

end




# ---- Plotting Functions ----

function plotAnnealResults(meanTrainErrorVec, meanValErrorVec, titleStr)
    plot(meanTrainErrorVec, label="Training Classification Error", alpha=0.7)
    plot(meanValErrorVec, label="Validation Classification Error", alpha=0.7)
    ylim(0, maximum([maximum(meanTrainErrorVec), maximum(meanValErrorVec)]))
    xlabel("Training Epoch")
    ylabel("5-Fold Cross-Validated Mean Classification Error")
    legend(loc=3)
    title(titleStr)
end



function plotTempFunctions(linFun, sigFun)
    fcnRange = 1000

    linFcnOut = linFun(fcnRange)

    sigFcnOut = sigFun(fcnRange)

    subplot(2,1,1)
    title("Temperature Schedules")
    plot((1:fcnRange)./fcnRange, linFcnOut, label="Linear Temperature Schedule")
    ylabel("Temperature")
    legend(loc=3)

    subplot(2,1,2)
    plot((1:fcnRange)./fcnRange, sigFcnOut, label="Sigmoid Temperature Schedule")
    ylabel("Temperature")
    xlabel("Fraction of Epochs Completed")
    legend(loc=3)

end

plotTempFunctions(linearTemperatureSchedule, sigmoidTemperatureSchedule)




###################################################################################################################################################################################################
# ---- Experiment Development Area ----

# Construct the iris dataset
irisData = readdlm(".\\iris.dat", ',' , Any)
irisDataClassed = transformClassData(irisData, [5])
irisDataClassed = normalizeData(irisDataClassed)

meanValErrorVec, meanTrainErrorVec, minValErrorSynapseMatrix = @time nFoldCrossValidateSynapticAnnealing(5, 1000, [5,10,3], annealSynapses_allWeight_fixedStep_Quantum, getDataRegErr, getDataClassErr, linearTemperatureSchedule_sixTenths, sigmoidDecay_Tuned, tanh, irisDataClassed, [1:4], [5:7])

putdata(meanTrainErrorVec, "meanTrainErrorVec_synapticAnnealing_randAllDecay_5fold_10K_n5_10_3_linea")
putdata(meanValErrorVec, "meanValErrorVec_synapticAnnealing_randAllDecay_5fold_10K_n5_10_3_linea")
putdata(minValErrorSynapseMatrix, "synapseMat_synapticAnnealing_randAllDecay_5fold_10K_n5_10_3_linea")

meanTrainErrorVec = getdata("meanTrainErrorVec_synapticAnnealing_randAllDecay_5fold_10K_n5_10_3_linea")
meanValErrorVec = getdata("meanValErrorVec_synapticAnnealing_randAllDecay_5fold_10K_n5_10_3_linea")
minValErrorSynapseMatrix = getdata("synapseMat_synapticAnnealing_randAllDecay_5fold_10K_n5_10_3_linea")

finalMeanValError = meanValErrorVec[end]

minClassError = getDataClassErr(minValErrorSynapseMatrix, tanh, irisDataClassed, [1:4], [5:7])

plotAnnealResults( meanTrainErrorVec, meanValErrorVec, "Training and Validation Classification Error of a Synaptic\n Annealing Neural Network using a Linear Temperature\n Schedule, Decaying Learn Rate After Annealing")


###################################################################################################################################################################################################





###################################################################################################################################################################################################
# ---- Experiment Development Area ----

# Construct the iris dataset
path()
irisData = readdlm("trainLHC.dat", ',' , Any)
irisDataClassed = transformClassData(irisData, [5])
irisDataClassed = normalizeData(irisDataClassed)

meanValErrorVec, meanTrainErrorVec, minValErrorSynapseMatrix = @time nFoldCrossValidateSynapticAnnealing(5, 1000, [5,10,3], annealSynapses_allWeight_fixedStep_Quantum, getDataRegErr, getDataClassErr, linearTemperatureSchedule_sixTenths, sigmoidDecay_Tuned, tanh, irisDataClassed, [1:4], [5:7])

putdata(meanTrainErrorVec, "meanTrainErrorVec_synapticAnnealing_randAllDecay_5fold_10K_n5_10_3_linea")
putdata(meanValErrorVec, "meanValErrorVec_synapticAnnealing_randAllDecay_5fold_10K_n5_10_3_linea")
putdata(minValErrorSynapseMatrix, "synapseMat_synapticAnnealing_randAllDecay_5fold_10K_n5_10_3_linea")

meanTrainErrorVec = getdata("meanTrainErrorVec_synapticAnnealing_randAllDecay_5fold_10K_n5_10_3_linea")
meanValErrorVec = getdata("meanValErrorVec_synapticAnnealing_randAllDecay_5fold_10K_n5_10_3_linea")
minValErrorSynapseMatrix = getdata("synapseMat_synapticAnnealing_randAllDecay_5fold_10K_n5_10_3_linea")

finalMeanValError = meanValErrorVec[end]

minClassError = getDataClassErr(minValErrorSynapseMatrix, tanh, irisDataClassed, [1:4], [5:7])

plotAnnealResults( meanTrainErrorVec, meanValErrorVec, "Training and Validation Classification Error of a Synaptic\n Annealing Neural Network using a Linear Temperature\n Schedule, Decaying Learn Rate After Annealing")


###################################################################################################################################################################################################










# ---- Experiment: Thermal Annealing with Random (Not Constant Euclidean Distance) Step Size in ONE Dimension  ----

# Construct the iris dataset
irisData = readdlm(".\\iris.dat", ',' , Any)
irisDataClassed = transformClassData(irisData, [5])
irisDataClassed = normalizeData(irisDataClassed)

meanValErrorVec, meanTrainErrorVec, minValErrorSynapseMatrix = @time nFoldCrossValidateSynapticAnnealing(5, 10000, [5,10,3], annealSynapses_singleWeight, getDataRegErr, getDataClassErr, linearTemperatureSchedule_eightTenths, function(x) return(0.1) end, tanh, irisDataClassed, [1:4], [5:7])

#putdata(meanTrainErrorVec, "meanTrainErrorVec_synapticAnnealing_randOne_5fold_10K_n5_10_3_linear_stepConst_zeroPointOne")
#putdata(meanValErrorVec, "meanValErrorVec_synapticAnnealing_randOne_5fold_10K_n5_10_3_linear_stepConst_zeroPointOne")
#putdata(minValErrorSynapseMatrix, "synapseMat_synapticAnnealing_randOne_5fold_10K_n5_10_3_linear_stepConst_zeroPointOne")

meanTrainErrorVec = getdata("meanTrainErrorVec_synapticAnnealing_randOne_5fold_10K_n5_10_3_linear_stepConst_zeroPointOne")
meanValErrorVec = getdata("meanValErrorVec_synapticAnnealing_randOne_5fold_10K_n5_10_3_linear_stepConst_zeroPointOne")
minValErrorSynapseMatrix = getdata("synapseMat_synapticAnnealing_randOne_5fold_10K_n5_10_3_linear_stepConst_zeroPointOne")

finalMeanValError = meanValErrorVec[end]

minClassError =  getDataClassErr(minValErrorSynapseMatrix, tanh, irisDataClassed, [1:4], [5:7])

plotAnnealResults(meanTrainErrorVec, meanValErrorVec, "Training and Validation Classification Error of a Synaptic\n Annealing Neural Network using a Linear Temperature\n Schedule, Varying a Single Synapse at each Epoch")






# ---- Experiment: Thermal Annealing with Random (Not Constant ED) Step Size in ALL Dimensions  ----

# Construct the iris dataset
irisData = readdlm(".\\iris.dat", ',' , Any)
irisDataClassed = transformClassData(irisData, [5])
irisDataClassed = normalizeData(irisDataClassed)

meanValErrorVec, meanTrainErrorVec, minValErrorSynapseMatrix = @time nFoldCrossValidateSynapticAnnealing(5, 10000, [5,10,3], annealSynapses_allWeight, getDataRegErr, getDataClassErr, linearTemperatureSchedule_eightTenths, function(x) return(1) end, tanh, irisDataClassed, [1:4], [5:7])

#putdata(meanTrainErrorVec, "meanTrainErrorVec_synapticAnnealing_randAll_5fold_10K_n5_10_3_linear")
#putdata(meanValErrorVec, "meanValErrorVec_synapticAnnealing_randAll_5fold_10K_n5_10_3_linear")
#putdata(minValErrorSynapseMatrix, "synapseMat_synapticAnnealing_randAll_5fold_10K_n5_10_3_linear")

meanTrainErrorVec = getdata("meanTrainErrorVec_synapticAnnealing_randAll_5fold_10K_n5_20_3_linear")
meanValErrorVec = getdata("meanValErrorVec_synapticAnnealing_randAll_5fold_10K_n5_20_3_linear")
minValErrorSynapseMatrix = getdata("synapseMat_synapticAnnealing_randAll_5fold_10K_n5_20_3_linear")

finalMeanValError = meanValErrorVec[end]

minClassError = getDataClassErr(minValErrorSynapseMatrix, tanh, irisDataClassed, [1:4], [5:7])

plotAnnealResults(meanTrainErrorVec, meanValErrorVec, "Training and Validation Classification Error of a Synaptic\n Annealing Neural Network using a Linear Temperature\n Schedule, Varying All Synapses at each Epoch")







# ---- Experiment: Fixed Magnitude Step Size ----

# Construct the iris dataset
irisData = readdlm(".\\iris.dat", ',' , Any)
irisDataClassed = transformClassData(irisData, [5])
irisDataClassed = normalizeData(irisDataClassed)
#sigmoidDecay_Tuned
meanValErrorVec, meanTrainErrorVec, minValErrorSynapseMatrix = @time nFoldCrossValidateSynapticAnnealing(5, 10000, [5,10,3], annealSynapses_allWeight_fixedStep, getDataRegErr, getDataClassErr, linearTemperatureSchedule_sixTenths, function(x) return(1) end, tanh, irisDataClassed, [1:4], [5:7])

finalMeanValError = meanValErrorVec[end]

minClassError = getDataClassErr(minValErrorSynapseMatrix, tanh, irisDataClassed, [1:4], [5:7])

plotAnnealResults(meanTrainErrorVec, meanValErrorVec, "Training and Validation Classification Error of a Synaptic\n Annealing Neural Network using a Linear Temperature\n Schedule, Varying All Synapses at Each Epoch")

# Save Data
#putdata(meanTrainErrorVec, "meanTrainErrorVec_synapticAnnealing_fixedAll_5fold_10K_n5_10_3_linear")
#putdata(meanValErrorVec, "meanValErrorVec_synapticAnnealing_fixedAll_5fold_10K_n5_10_3_linear")
#putdata(minValErrorSynapseMatrix, "synapseMat_synapticAnnealing_fixedAll_5fold_10K_n5_10_3_linear")

# Retrieve Data
meanTrainErrorVec = getdata("meanTrainErrorVec_synapticAnnealing_fixedAll_5fold_10K_n5_10_3_linear")
meanValErrorVec = getdata("meanValErrorVec_synapticAnnealing_fixedAll_5fold_10K_n5_10_3_linear")
minValErrorSynapseMatrix = getdata("synapseMat_synapticAnnealing_fixedAll_5fold_10K_n5_10_3_linear")






# ---- Experiment: Reduce Step Size as Alpha ----




# Construct the iris dataset
irisData = readdlm(".\\iris.dat", ',' , Any)
irisDataClassed = transformClassData(irisData, [5])
irisDataClassed = normalizeData(irisDataClassed)

meanValErrorVec, meanTrainErrorVec, minValErrorSynapseMatrix = @time nFoldCrossValidateSynapticAnnealing(5, 15000, [5,10,3], annealSynapses_allWeight_fixedStep, getDataRegErr, getDataClassErr, linearTemperatureSchedule_eightTenths, sigmoidDecay_Tuned, tanh, irisDataClassed, [1:4], [5:7])

#putdata(meanTrainErrorVec, "meanTrainErrorVec_synapticAnnealing_randAllDecay_5fold_10K_n5_10_3_linear")
#putdata(meanValErrorVec, "meanValErrorVec_synapticAnnealing_randAllDecay_5fold_10K_n5_10_3_linear")
#putdata(minValErrorSynapseMatrix, "synapseMat_synapticAnnealing_randAllDecay_5fold_10K_n5_10_3_linear")

meanTrainErrorVec = getdata("meanTrainErrorVec_synapticAnnealing_randAllDecay_5fold_10K_n5_10_3_linear")
meanValErrorVec = getdata("meanValErrorVec_synapticAnnealing_randAllDecay_5fold_10K_n5_10_3_linear")
minValErrorSynapseMatrix = getdata("synapseMat_synapticAnnealing_randAllDecay_5fold_10K_n5_10_3_linear")

finalMeanValError = meanValErrorVec[end]

minClassError = getDataClassErr(minValErrorSynapseMatrix, tanh, irisDataClassed, [1:4], [5:7])

plotAnnealResults(meanTrainErrorVec, meanValErrorVec, "Training and Validation Classification Error of a Synaptic\n Annealing Neural Network using a Linear Temperature\n Schedule, Decaying Learn Rate After Annealing")

# Special storage for joint plotting procedure
meanValErrorVec_anneal = meanValErrorVec

sigmoidDecay_Tuned([0.001:0.001:1])
linearTemperatureSchedule_sixTenths(1000)
plot([0.001:0.001:1], linearTemperatureSchedule_sixTenths(1000), label="Linear Temperature Schedule")
plot([0.001:0.001:1], sigmoidDecay_Tuned([0.001:0.001:1]), label="Sigmoid Step Size Decay")

title("Simulated Annealing Parameters Through Epochs")
ylabel("Decay")
xlabel("Fraction of Epochs Completed")


# ---- Experiment: Reduce Step Size to Enable Online Learning ----



# ---- Resource Analysis ----

sampleRes = 10
maxEpochs = 100
epochComplexityVec = zeros(length(1:sampleRes:maxEpochs),2)

iter = 0
for epochTrial in 1:sampleRes:maxEpochs
    iter+=1
    tic()
    nFoldCrossValidateSynapticAnnealing(5, int(epochTrial), [5,10,3], annealSynapses_allWeight, getDataRegErr, getDataClassErr, linearTemperatureSchedule_eightTenths, function(x) return(1) end, tanh, irisDataClassed, [1:4], [5:7])

    time=toq()
    println([int(epochTrial), time[1]])
    println(iter)
    epochComplexityVec[iter,:] = [int(epochTrial) time[1]/(5*int(epochTrial))]
end

plot(epochComplexityVec[:,1],epochComplexityVec[:,2])




sampleRes = 10
maxSamples = 100
samplesComplexityVec = zeros(length(10:sampleRes:maxSamples+10),2)
irisDataClassed[1:10,:]
iter = 0
for numSamples in 10:sampleRes:maxSamples+10
    iter+=1
    dataSubset = irisDataClassed[(1:(numSamples)), :]

    tic()
    nFoldCrossValidateSynapticAnnealing(5, 100, [5,10,3], annealSynapses_allWeight, getDataRegErr, getDataClassErr, linearTemperatureSchedule_eightTenths, function(x) return(1) end, tanh, dataSubset, [1:4], [5:7])
    time=toq()

    samplesComplexityVec[iter, :] = [numSamples, time[1]]
end
samplesComplexityVec

plot(samplesComplexityVec[:,1],samplesComplexityVec[:,2]/(5*100))
title("Variation in CPU Time per Epoch with Increasing Sample Size")
ylabel("Average CPU Seconds Per Epoch")
xlabel("Number of Samples in the Dataset")




sampleRes = 1
maxNeurons = 75
neuronsComplexityVec = zeros(length(1:sampleRes:maxNeurons),2)
iter = 0
for numNeurons in 1:sampleRes:maxNeurons
    iter+=1

    tic()
    nFoldCrossValidateSynapticAnnealing(5, 10, [5,int(numNeurons),3], annealSynapses_allWeight, getDataRegErr, getDataClassErr, linearTemperatureSchedule_eightTenths, function(x) return(1) end, tanh, irisDataClassed, [1:4], [5:7])
    time=toq()

    neuronsComplexityVec[iter, :] = [numNeurons, time[1]]
end
neuronsComplexityVec

plot(neuronsComplexityVec[:,1],neuronsComplexityVec[:,2]/(10*5))
title("Variation in CPU Time per Epoch with Increasing Hidden Layer Size")
ylabel("Average CPU Seconds Per Epoch")
xlabel("Number of Neurons in the Hidden Layer")



sampleRes = 1
maxLayers= 20
layersComplexityVec = zeros(length(1:sampleRes:maxLayers),2)
iter = 0

for numLayers in 1:sampleRes:maxLayers
    iter+=1
    println(iter)
    tic()
    nFoldCrossValidateSynapticAnnealing(5, 10, [5,int([5*ones(int(numLayers))]),3], annealSynapses_allWeight, getDataRegErr, getDataClassErr, linearTemperatureSchedule_eightTenths, function(x) return(1) end, tanh, irisDataClassed, [1:4], [5:7])
    time=toq()

    layersComplexityVec[iter, :] = [numLayers, time[1]]
end
layersComplexityVec

plot(layersComplexityVec[:,1],layersComplexityVec[:,2]/(10*5))
title("Variation in CPU Time per Epoch with Increasing Hidden Layers")
ylabel("Average CPU Seconds Per Epoch")
xlabel("Number of Hidden Layers (5 Neurons Each)")



plot(layersComplexityVec[:,1]*5,layersComplexityVec[:,2]/(10*5), label="Mulitiple Hidden Layers")
plot(neuronsComplexityVec[:,1],neuronsComplexityVec[:,2]/(10*5), label="Single Hidden Layer")
title("Variation in CPU Time per Epoch with\n Increasing Number of Hidden Neurons")
ylabel("Average CPU Seconds Per Epoch")
xlabel("Number of Neurons in the Hidden Layer")
legend(loc=2)


# Calculate individual epoch time.
# Construct the iris dataset
irisData = readdlm(".\\iris.dat", ',' , Any)
irisData = transformClassData(irisData, [5])
irisData = normalizeData(irisData)

function calcEpochTime_annealing(trainingFunction, numEpochs, synMatConfigVec, errorFunction, reportErrorFunction, temperatureScheduleFcn, learnRateDecay, actFun, trainData, valData, inputCols, outputCols)

    synapseMatrixIn = createSynapseMatrix(synMatConfigVec)
    tic()
    trainingFunction(numEpochs, errorFunction, reportErrorFunction, temperatureScheduleFcn,  function(x) return(1) end, synapseMatrixIn, actFun, trainData, valData, inputCols, outputCols)
    time = toq()[1]
    return(time/numEpochs)
end

epochTimeAvg_anneal = calcEpochTime_annealing(annealSynapses_allWeight_fixedStep, 100, [5,10,3], getDataRegErr, getDataClassErr, linearTemperatureSchedule_eightTenths, learnRateDecay, tanh, irisData[31:end, :], irisData[1:30, :], [1:4], [5:7])
epochTimeAvg_anneal


function annealSynapses_allWeight_fixedStep(numEpochs, errorFunction, reportErrorFunction, temperatureScheduleFcn, learnRateDecay, synapseMatrixIn, actFun, trainData, valData, inputCols, outputCols)

# ---- Experiment: Temperature Schedules ----


# Construct the iris dataset
irisData = readdlm(".\\iris.dat", ',' , Any)
irisDataClassed = transformClassData(irisData, [5])
irisDataClassed = normalizeData(irisDataClassed)

meanValErrorVec_lin, meanTrainErrorVec_lin, minValErrorSynapseMatrix_lin = @time nFoldCrossValidateSynapticAnnealing(30, 500, [5,10,3], annealSynapses_allWeight_fixedStep, getDataRegErr, getDataClassErr, linearTemperatureSchedule_eightTenths, sigmoidDecay_Tuned, tanh, irisDataClassed, [1:4], [5:7])

meanValErrorVec_sig, meanTrainErrorVec_sig, minValErrorSynapseMatrix_sig = @time nFoldCrossValidateSynapticAnnealing(30, 500, [5,10,3], annealSynapses_allWeight_fixedStep, getDataRegErr, getDataClassErr, sigmoidTemperatureSchedule, sigmoidDecay_Tuned, tanh, irisDataClassed, [1:4], [5:7])


plot(meanValErrorVec_lin, label="Linear Temperature Classification Error", alpha=0.7)
plot(meanValErrorVec_sig, label="Sigmoid Temperature Classification Error", alpha=0.7)
ylim(0, maximum([maximum(meanValErrorVec_lin), maximum(meanValErrorVec_sig)]))
xlabel("Training Epoch")
ylabel("10-Fold Cross-Validated Mean Classification Error")
legend(loc=3)
title("Validation Classification Error of a Synaptic Annealing\n Neural Network using Different Temperature Schedules")








# ---- Inverese Data Set Regression ----

# Construct the dataset
inverseDataTrain, inverseDataTest = constructInverseDataset(100:1:200, .8)

# Construct the initial synapse matrix.
synapseMatrix = createSynapseMatrix([1,5,1])

synapseMatrix, trainErrorVector, valErrorVector = @time annealSynapses_allWeight(1000, calculateDatasetRegressionError, linearTemperatureSchedule, synapseMatrix, tanh, inverseDataTrain, inverseDataTest, [1], [2])

valErrorVector[end]

plot(trainErrorVector)plot(valErrorVector)




# ---- Run Time Evaluations ----


meanValErrorVec_anneal
epochTimeAvg_anneal
timeVec_anneal = epochTimeAvg_anneal.*[i for i in 1:length(meanValErrorVec_anneal)]

meanValErrorVec_backprop
epochTimeAvg_backprop
timeVec_backprop = epochTimeAvg_backprop.*[i for i in 1:length(meanValErrorVec_backprop)]

epochTimeAvg_backprop/epochTimeAvg_anneal

plot(timeVec_anneal, meanValErrorVec_anneal, label="Simulated Annealing", alpha=0.7)
plot(timeVec_backprop, meanValErrorVec_backprop, label="Back Propagation", alpha=0.7)
ylim(0, 1)
xlabel("CPU Time (s)")
ylabel("5-Fold Cross-Validated Mean Classification Error")
legend(loc=1)
title("Classification Error of Neural Network Training Algorithms through Time\n")






# ---- Goals ----

# Experiment for runtime
# # Experiment for simulated annealing optimizing a basis vector configuration of a temperature schedule.
# Experiment for best step size
# # Experiment for best step size decay??
# Create synapse matrix visualization.
# Attempt quantum annealing
# annealable quantities: Number of runs, matrix configuration, temperature schedule, learn reate decay.

# Is reheating at smaller stepsizes a valid approach? Should allow exploration of fine-grain details?
## Shorten step size once mean of MSE has stableized and increase temperature to explore the more nuanced topology of the error manifold.

# Is tunneling a valid phenominon?
## Low probability random roll to look at a large euclidean distance synapse-space step. If lower MSE, take it.

# The synapse space error manifold is fractal. Thus an oscilliatory heating and cooling temperature schedulem with an exponentially decaying step size should produce very low MSEs.
# Or a psuedo-cyclic structure wherein the temperature is abruptly increased and the step size radically decreased each time the MSE levels out until doing so produces no effect.

# Should I do hold-out set testing on Iris?


# Quantum Synaptic Annealing Neural Networks.


  # Joint plotting


