Pkg.init()
Pkg.add("PyCall")
Pkg.build("PyCall")
Pkg.add("PyPlot")
Pkg.build("PyPlot")
# Pkg.add("StatsBase")
Pkg.add("BackpropNeuralNet")
Pkg.add("MNIST")
Pkg.update()

Pkg.build()

using PyPlot
rmprocs(workers())
addprocs(6)
# @everywhere using Devectorize

@everywhere using BackpropNeuralNet
@everywhere cd("\\fletcher-thesis")

pwd()

@everywhere include("$(pwd())\\src\\"*"ExperimentDataset.jl")
@everywhere include("$(pwd())\\src\\"*"AnnealingState.jl")

# Include utility libraries.
@everywhere include("$(pwd())\\src\\"*"getput.jl")
@everywhere include("$(pwd())\\src\\"*"vectorListMean.jl")
@everywhere include("$(pwd())\\src\\"*"vectorListToMatrix.jl")

# Include data maniputlation libraries.
@everywhere include("$(pwd())\\src\\"*"normalizeData.jl")
@everywhere include("$(pwd())\\src\\"*"removeDataMean.jl")
@everywhere include("$(pwd())\\src\\"*"orthogonalizeDataClasses.jl")
@everywhere include("$(pwd())\\src\\"*"shuffleData.jl")

# Include synaptic annealing libraries.
@everywhere include("$(pwd())\\src\\"*"createSynapseMatrix.jl")
@everywhere include("$(pwd())\\src\\"*"propogateForward.jl")
@everywhere include("$(pwd())\\src\\"*"plotAnnealResults.jl")
@everywhere include("$(pwd())\\src\\"*"annealingTraversalFunctions.jl")
@everywhere include("$(pwd())\\src\\"*"synapticAnnealing.jl")
@everywhere include("$(pwd())\\src\\"*"errorFunctions.jl")
@everywhere include("$(pwd())\\src\\"*"getDataPredictions.jl")
@everywhere include("$(pwd())\\src\\"*"nativeNetsToSynMats.jl")

# Include cross val annealing libraries.
@everywhere include("$(pwd())\\src\\"*"buildFolds.jl")
@everywhere include("$(pwd())\\src\\"*"nFoldCrossValidateSynapticAnnealing.jl")

# Include cross val annealing libraries.
@everywhere include("$(pwd())\\src\\"*"backpropTraining.jl")
@everywhere include("$(pwd())\\src\\"*"nFoldCrossValidateBackprop.jl")

ion()

#########################################################################################################################

# workspace()

#########################################################################################################################
# Construct the iris dataset

irisData = readdlm("$(pwd())\\data\\iris.dat", ',' , Any)
irisDataClassed = orthogonalizeDataClasses(irisData, [5])
irisDataClassed = normalizeData(irisDataClassed)
irisDataClassed = shuffleData(irisDataClassed)
irisDataClassed = removeDataMean(irisDataClassed,[1:4])

irisDatapath = "$(pwd())\\data\\iris.dat"
dataInputDimensions = [1:4]
dataOutputDimensions = [5]

irisDataset = ExperimentDataset.Dataset(irisDatapath, dataInputDimensions, dataOutputDimensions)
irisDataset.data[:,irisDataset.outputCols] = (irisDataset.data[:,irisDataset.outputCols]+1)/2
irisDataset = ExperimentDataset.Dataset(irisDataset.data, dataInputDimensions, dataOutputDimensions)


###################################################################################################################################################

lcvfData = readdlm("$(pwd())\\data\\lcvfData.csv", ',' , Any)
lcvfDataClassed = orthogonalizeDataClasses(lcvfData, [195])
lcvfDataClassed = normalizeData(lcvfDataClassed)
lcvfDataClassed = shuffleData(lcvfDataClassed)


lcvfDatapath = "$(pwd())\\data\\lcvfData.csv"
dataInputDimensions = [1:194]
dataOutputDimensions = [195]

lcvfDataset = ExperimentDataset.Dataset(lcvfDatapath, dataInputDimensions, dataOutputDimensions)

###################################################################################################################################################
using MNIST

# Function to orthogonalize MNIST. Credit: github.com/yarlett
function digits_to_indicators(digits)
	digit_indicators = zeros(Float64, (10, length(digits)))
	for j = 1:length(digits)
		digit_indicators[int(digits[j])+1, j] = 1.0
	end
	digit_indicators
end

# Load MNIST training and testing data.
mnistTrainInput, mnistTrainClasses = traindata()
mnistTrainInput ./= 255.0
mnistTrainClasses = digits_to_indicators(mnistTrainClasses)

# XTE, YTE = testdata()
# XTE ./= 255.0
# YTE = digits_to_indicators(YTE)

# Make the classes antisemetric for consistency.
mnistTrainClassesAntisymmetric = (mnistTrainClasses*2)-1

# Transpose the MNIST training data for consistency.
mnistTrainInput = transpose(mnistTrainInput)
mnistTrainClassesAntisymmetric = transpose(mnistTrainClassesAntisymmetric)

mnistTrainData = [mnistTrainInput mnistTrainClassesAntisymmetric]

dataInputDimensions = [1:size(mnistTrainInput)[2]]
dataOutputDimensions = size(mnistTrainInput)[2]+1

mnistDataset = ExperimentDataset.Dataset(mnistTrainData[1:150, :], dataInputDimensions, dataOutputDimensions)



###################################################################################################################################################

dataSet = irisDataset

###################################################################################################################################################

dataSet = lcvfDataset

###################################################################################################################################################

###################################################################################################################################################

dataSet = mnistDataset

###################################################################################################################################################

numFolds = 6

maxRuns = 1000000

initTemp = 500

numHiddenLayers = 1

matrixConfig = [length(dataSet.inputCols), repmat([length(dataSet.inputCols)], numHiddenLayers), length(dataSet.outputCols)]

matrixConfig = [length(dataSet.inputCols), 50, length(dataSet.outputCols)]

synMatIn = null

batchSize = 150

reportFrequency = 100


###################################################################################################################################################


outTuple_csa = @time nFoldCrossValidateSynapticAnnealingPar(numFolds, matrixConfig, synapticAnnealing,
                                                          0.0, maxRuns, csaSynapticChange, AnnealingState.updateState_csa,
                                                          getDataClassErr, getDataClassErr,
                                                          initTemp, 1,
                                                          synMatIn, tanh,
                                                          dataSet, batchSize, reportFrequency)

# putdata(outTuple_q, "outTuple_q")
# outTuple_q = getdata("outTuple_q")
(meanValErrorVec_csa, meanTrainErrorVec_csa, meanPerturbDistanceVec_csa, minValErrorSynapseMatrix_csa) = outTuple_csa
meanPerturbDistanceVec_csa
plotAnnealResults(meanTrainErrorVec_csa, meanValErrorVec_csa, reportFrequency, "CSA - Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")



####################################
###################################################################################################################################################


outTuple_qi = @time nFoldCrossValidateSynapticAnnealingPar(numFolds, matrixConfig, synapticAnnealing,
                                                          0.0, maxRuns, quantumIsotropicSynapticChange, AnnealingState.updateState_oscillatory,
                                                          getDataClassErr, getDataClassErr,
                                                          initTemp, 1,
                                                          synMatIn, tanh,
                                                          dataSet, batchSize, reportFrequency)

# putdata(outTuple_q, "outTuple_q")
# outTuple_q = getdata("outTuple_q")
(meanValErrorVec_qi, meanTrainErrorVec_qi, meanPerturbDistanceVec_qi, minValErrorSynapseMatrix_qi) = outTuple_qi
meanPerturbDistanceVec_qi
plotAnnealResults(meanTrainErrorVec_qi, meanValErrorVec_qi, reportFrequency, "GSA - Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")



###################################################################################################################################################


###################################################################################################################################################


outTuple_qua = @time nFoldCrossValidateSynapticAnnealingPar(numFolds, matrixConfig, synapticAnnealing,
                                                          0.0, maxRuns,  quantumUniformlyAnisotropicSynapticChange, AnnealingState.updateState_oscillatory,
                                                          getDataClassErr, getDataClassErr,
                                                          initTemp, 1,
                                                          synMatIn,tanh,
                                                          dataSet, batchSize, reportFrequency)

# putdata(outTuple_q, "outTuple_q")
# outTuple_q = getdata("outTuple_q")

(meanValErrorVec_qua, meanTrainErrorVec_qua, meanPerturbDistanceVec_qua, minValErrorSynapseMatrix_qua) = outTuple_qua

plotAnnealResults(meanTrainErrorVec_qua, meanValErrorVec_qua, reportFrequency, "QUA- Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")



###################################################################################################################################################

###################################################################################################################################################


outTuple_qva = @time nFoldCrossValidateSynapticAnnealingPar(numFolds, matrixConfig, synapticAnnealing,
                                                          0.0, maxRuns,  quantumVariablyAnisotropicSynapticPerturbation, AnnealingState.updateState_oscillatory,
                                                          getDataClassErr, getDataClassErr,
                                                          initTemp, 1,
                                                          synMatIn, tanh,
                                                          dataSet, batchSize, reportFrequency)

# putdata(outTuple_q, "outTuple_qva")
# outTuple_q = getdata("outTuple_qva")

(meanValErrorVec_qva, meanTrainErrorVec_qva, meanPerturbDistanceVec_qva, minValErrorSynapseMatrix_qva) = outTuple_qva

 meanPerturbDistanceVec_qva

plotAnnealResults(meanTrainErrorVec_qva, meanValErrorVec_qva, reportFrequency, "QVA - Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")


###################################################################################################################################################

