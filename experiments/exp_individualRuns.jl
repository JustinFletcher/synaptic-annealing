# Pkg.init()
# Pkg.add("PyCall")
# Pkg.build("PyCall")
# Pkg.add("PyPlot")
# Pkg.add("StatsBase")
# Pkg.update()

using PyPlot

addprocs(28)
rmprocs(workers())

# Include utility libraries.
@everywhere include("getput.jl")
@everywhere include("vectorListMean")
@everywhere include("vectorListToMatrix.jl")

# Include data maniputlation libraries.
@everywhere include("normalizeData.jl")
@everywhere include("orthogonalizeDataClasses.jl")


# Include synaptic annealing libraries.
@everywhere include("createSynapseMatrix.jl")
@everywhere include("propogateForward.jl")
@everywhere include("temperatureLib.jl")
@everywhere include("decayLib.jl")
@everywhere include("errorFunctions.jl")
@everywhere include("shuffleData.jl")
@everywhere include("getDataPredictions.jl")
@everywhere include("plotAnnealResults.jl")
@everywhere include("synapticAnnealingEpoch.jl")
@everywhere include("annealingTraversalFunctions.jl")
@everywhere include("stateUpdateFunctions.jl")
@everywhere include("synapticAnnealing.jl")

# Include cross val annealing libraries.
@everywhere include("buildFolds.jl")
@everywhere include("nFoldCrossValidateSynapticAnnealing")

ion()


###################################################################################################################################################################################################
# ---- Experiment Development Area ----

# Construct the iris dataset.
irisData = readdlm("C:\\Users\\serg\\OneDrive\\afit\\rs\\synapticAnnealing\\iris.dat", ',' , Any)
irisDataClassed = orthogonalizeDataClasses(irisData, [5])
irisDataClassed = normalizeData(irisDataClassed)
irisDataClassed = shuffleData(irisDataClassed)

# Partition the iris dataset.
irisDataTrain = irisDataClassed[1:100, :]
irisDataVal = irisDataClassed[101:125, :]
irisDataTest = irisDataClassed[126:end, :]

synapseMatrixIn = createSynapseMatrix([5,10,3])


outTuple = @time synapticAnnealing(0.0, 50, quantumSynapticChange, updateState_q,
                                   getDataClassErr, getDataClassErr,
                                   5, 1,
                                   synapseMatrixIn, tanh,
                                   irisDataTrain, irisDataVal,
                                   [1:4], [5:7])



outTuple = @time synapticAnnealing(0.0, 500, fixedStepSizeOmniDimSynapticChange, updateState,
                                   getDataClassErr, getDataClassErr,
                                   50, 1,
                                   synapseMatrixIn, tanh,
                                   irisDataTrain, irisDataVal,
                                   [1:4], [5:7])



outTuple = @time synapticAnnealing(0.0, 500, omniDimSynapticChange, updateState,
                                   getDataClassErr, getDataClassErr,
                                   50, 1,
                                   synapseMatrixIn, tanh,
                                   irisDataTrain, irisDataVal,
                                   [1:4], [5:7])

outTuple = @time synapticAnnealing(0.0, 500, singleDimSynapticChange, updateState,
                                   getDataClassErr, getDataClassErr,
                                   50, 1,
                                   synapseMatrixIn, tanh,
                                   irisDataTrain, irisDataVal,
                                   [1:4], [5:7])


(minValErrorSynapseMatrix, validationErrorVector, trainingErrorVector, perturbationDistanceVector) = outTuple

finalValError = validationErrorVector[end]

# Calculate the classification error of the minimum validation error synapse matrix, on the entire data set.
temp =0
for t = 1:1000
	temp += getDataClassErr(createSynapseMatrix([5,10,3]), tanh, irisDataClassed, [1:4], [5:7])
end
temp/1000

plotAnnealResults(trainingErrorVector,validationErrorVector, "Training and Validation Classification Error of a Quantum Synaptic\n Annealing Neural Network using a Linear Temperature\n Schedule.")

###################################################################################################################################################################################################

outTuple = @time synapticAnnealing(0.0, 500, quantumSynapticChange, updateState,
                                   getDataClassErr, getDataClassErr,
                                   50, 1,
                                   synapseMatrixIn, tanh,
                                   irisDataTrain, irisDataVal,
                                   [1:4], [5:7])
