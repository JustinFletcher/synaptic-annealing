
function synapticAnnealing(convCriterion, cutoffEpochs, synapticChangeFcn, errorFunction, reportErrorFunction, initTemperature, temperatureFcn, initLearnRate, learnRateFcn, synapseMatrixIn, actFun, trainData, valData, inputCols, outputCols)

    println("New Synaptic Annealing Run")

    # Create a local copy of the synapse matrix.
    synapseMatrix = copy(synapseMatrixIn)

    # Initialize the error.
    error = errorFunction(synapseMatrix, actFun, trainData, inputCols, outputCols)

    # Initiailize the minimum errors.
    minError = Inf
    trainErr = Inf
    valErr = Inf

    # Initialize the error vectors.
    trainingErrorVector = Float64[]
    validationErrorVector = Float64[]
    onlineErrorVector = Float64[]
    errorStack = Float64[]
    minValErrSynapseMatrix = null

    # Developmental
    varianceVector = Float64[]

    # Initialize state variables.
    temperature = initTemperature
    learnRate = initLearnRate
    tunnelingField = 0
    epochsCool = 0
    tunnelingMaxLength = (abs(actFun(Inf))+abs(actFun(-Inf)))*sum(synapseMatrix.!=0)

    # Initialize the loop control variables.
    converged = false
    numEpochs = 0

    while !converged

        # Iterate epoch count.
        numEpochs += 1
#         println(numEpochs)

        # Push the most recent error onto the error vector.
        trainErr = reportErrorFunction(synapseMatrix, actFun, trainData, inputCols, outputCols)
        push!(trainingErrorVector, trainErr)

        # Push the validation error set onto the vector.
        valErr = reportErrorFunction(synapseMatrix, actFun, valData, inputCols, outputCols)
        push!(validationErrorVector, valErr)

        # State capture: if this is the best net so far, save it to the disk.
        if valErr < minError
            minValErrSynapseMatrix = synapseMatrix
        end

        # Control the quantum kinetic energy term.
        epochsCool += int(temperature==0)
        if epochsCool > 3*int(initTemperature)
            tunnelingField = 0.5
        end

        # Update the temperature value.
        temperature = temperatureFcn(temperature)
        temperatureNorm = temperature/initTemperature


        # Contruct the tuple which serves as the synaps matrix update functions.
        synapseChangeInputTuple = [learnRate, tunnelingField, tunnelingMaxLength]
        stateTuple = [temperatureNorm, learnRate, tunnelingField, tunnelingMaxLength]

#         # Select a random element of the training data.
#         trainSample = trainData[rand(1:end),:]

#         println("----------", trainSample)
#         # Run an epoch of the annealing function.
#         (synapseMatrix, error) = synapticAnnealingEpoch(synapseMatrix, temperatureNorm, error, errorFunction, synapticChangeFcn, synapseChangeInputTuple, trainSample, inputCols, outputCols, actFun)

        # Run an epoch of the annealing function.
        (synapseMatrix, error) = synapticAnnealingEpoch(synapseMatrix, temperatureNorm, error, errorFunction, synapticChangeFcn, synapseChangeInputTuple, trainData, inputCols, outputCols, actFun)

        # Add the most recent error to the error history.
#         push!(errorStack, error)
#         push!(onlineErrorVector, error)

        # Evaluate the convergence conditions.
        converged = (numEpochs>=cutoffEpochs)  || ((valErr<convCriterion)&&(trainErr<convCriterion)) || ((trainErr == 0.0 )&&(valErr == 0.0))


    end


    # Return the annealed synapse matrix and the error vector.
    return(minValErrSynapseMatrix, validationErrorVector, trainingErrorVector)

end

