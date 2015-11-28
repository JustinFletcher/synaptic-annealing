# Pkg.init()
# Pkg.add("PyCall")
# Pkg.build("PyCall")
# Pkg.add("PyPlot")
# Pkg.add("StatsBase")
# Pkg.add("BackpropNeuralNet")
# Pkg.add("MNIST")
Pkg.build("BackpropNeuralNet")
Pkg.update()




@everywhere using PyPlot
rmprocs(workers())
addprocs(25)

# @everywhere using Devectorize

@everywhere using BackpropNeuralNet

@everywhere cd("\\fletcher-thesis")

pwd()
# Include utility libraries.

@everywhere include("$(pwd())\\src\\"*"getput.jl")
@everywhere include("$(pwd())\\src\\"*"vectorListMean.jl")
@everywhere include("$(pwd())\\src\\"*"vectorListToMatrix.jl")

# Include data maniputlation libraries.
@everywhere include("$(pwd())\\src\\"*"normalizeData.jl")
@everywhere include("$(pwd())\\src\\"*"orthogonalizeDataClasses.jl")
@everywhere include("$(pwd())\\src\\"*"shuffleData.jl")

# Include synaptic annealing libraries.
@everywhere include("$(pwd())\\src\\"*"createSynapseMatrix.jl")
@everywhere include("$(pwd())\\src\\"*"propogateForward.jl")
@everywhere include("$(pwd())\\src\\"*"plotAnnealResults.jl")
@everywhere include("$(pwd())\\src\\"*"annealingTraversalFunctions.jl")
@everywhere include("$(pwd())\\src\\"*"stateUpdateFunctions.jl")
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

@everywhere include("$(pwd())\\src\\"*"ExperimentDataset.jl")

@everywhere include("$(pwd())\\src\\"*"AnnealingState.jl")

state = AnnealingState.State(0,0,0,1)


ion()


#########################################################################################################################

# workspace()

#########################################################################################################################
# Construct the iris dataset

irisData = readdlm("$(pwd())\\data\\iris.dat", ',' , Any)
irisDataClassed = orthogonalizeDataClasses(irisData, [5])
irisDataClassed = normalizeData(irisDataClassed)
irisDataClassed = shuffleData(irisDataClassed)

irisDatapath = "$(pwd())\\data\\iris.dat"
dataInputDimensions = [1:4]
dataOutputDimensions = [5]

irisDataset = ExperimentDataset.Dataset(irisDatapath, dataInputDimensions, dataOutputDimensions)

irisDataset = ExperimentDataset.Dataset(irisDataset.data, dataInputDimensions, dataOutputDimensions)


###################################################################################################################################################

lcvfData = readdlm("$(pwd())\\data\\lcvfData.csv", ',' , Any)
lcvfDataClassed = orthogonalizeDataClasses(lcvfData, [195])
lcvfDataClassed = normalizeData(lcvfDataClassed)
lcvfDataClassed = shuffleData(lcvfDataClassed)


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

dataSet = lcvfDataClassed[1:100, :]

###################################################################################################################################################

###################################################################################################################################################

dataSet = mnistDataset


###################################################################################################################################################



numFolds = 25

maxRuns = 1000

initTemp = 800

numHiddenLayers = 1

matrixConfig = [length(dataSet.inputCols), repmat([length(dataSet.inputCols)], numHiddenLayers), length(dataSet.outputCols)]

matrixConfig = [length(dataSet.inputCols), 50,  length(dataSet.outputCols)]


###################################################################################################################################################

dataSetBackProp = irisDataClassed
dataSetBackProp[:,dataOutputDimensions] =  (dataSetBackProp[:,dataOutputDimensions]+1)/2

dataSetBackProp

outTuple_bp = @time nFoldCrossValidateBackpropPar(numFolds, matrixConfig, synapticAnnealing,
                                                          0.0, maxRuns, quantumAnisotropicSynapticPerturbation, updateState_oscillatory,
                                                          getDataClassErr, getDataClassErr,
                                                          initTemp, 1,
                                                          tanh,
                                                          dataSetBackProp, dataInputDimensions, dataOutputDimensions)

putdata(outTuple_bp, "outTuple_bp")
outTuple_bp = getdata("outTuple_bp")

(meanValErrorVec_bp, meanTrainErrorVec_bp, meanPerturbDistanceVec_bp, minValErrorSynapseMatrix_bp) = outTuple_bp

plotAnnealResults(meanTrainErrorVec_bp, meanValErrorVec_bp, "Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")


###################################################################################################################################################
# synMatIn = minValErrorSynapseMatrix_aq[1]

int((7*60*60)/(196.1/10000))

int((8*60*60)/(26/1))
# minValErrorSynapseMatrix_bp[1].weights

# synMatIn[:,1:4,1] = minValErrorSynapseMatrix_bp[1].weights[1]
# synMatIn[1:end-1,1:3,2] = minValErrorSynapseMatrix_bp[1].weights[2][1:end-1,:]

synMatIn = null
###################################################################################################################################################

function softmax(x)
	return(log(1+exp(x)))
end


outTuple_aq = @time nFoldCrossValidateSynapticAnnealingPar(numFolds, matrixConfig, synapticAnnealing,
                                                          0.0, maxRuns, quantumAnisotropicSynapticPerturbation, updateState_oscillatory,
                                                          getDataClassErr, getDataClassErr,
                                                          initTemp, 1,
               											  synMatIn, softmax,
                                                          dataSet)

# putdata(outTuple_aq, "outTuple_aq")
outTuple_aq = getdata("outTuple_aq")

(meanValErrorVec_aq, meanTrainErrorVec_aq, meanPerturbDistanceVec_aq, minValErrorSynapseMatrix_aq) = outTuple_aq

plotAnnealResults(meanTrainErrorVec_aq, meanValErrorVec_aq, "Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")



###################################################################################################################################################

outTuple_qo = @time nFoldCrossValidateSynapticAnnealingPar(numFolds, matrixConfig, synapticAnnealing,
                                                          0.0, maxRuns, quantumSynapticChange, updateState_quantum_only,
                                                          getDataClassErr, getDataClassErr,
                                                          initTemp, 1,
                                                          synMatIn,tanh,
                                                          dataSet)

# putdata(outTuple_qo, "outTuple_qo")
# outTuple_qo = getdata("outTuple_qo")

(meanValErrorVec_qo, meanTrainErrorVec_qo, meanPerturbDistanceVec_qo, minValErrorSynapseMatrix_qo) = outTuple_qo

plotAnnealResults(meanTrainErrorVec_qo, meanValErrorVec_qo, "Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")


###################################################################################################################################################


outTuple_q = @time nFoldCrossValidateSynapticAnnealingPar(numFolds, matrixConfig, synapticAnnealing,
                                                          0.0, maxRuns, quantumSynapticChange, updateState_oscillatory,
                                                          getDataClassErr, getDataClassErr,
                                                          initTemp, 1,
                                                          synMatIn,tanh,
                                                          dataSet, dataInputDimensions, dataOutputDimensions)

# putdata(outTuple_q, "outTuple_q")
# outTuple_q = getdata("outTuple_q")

(meanValErrorVec_q, meanTrainErrorVec_q, meanPerturbDistanceVec_q, minValErrorSynapseMatrix_q) = outTuple_q

plotAnnealResults(meanTrainErrorVec_q, meanValErrorVec_q, "Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")



###################################################################################################################################################



outTuple_f = @time nFoldCrossValidateSynapticAnnealingPar(numFolds, matrixConfig, synapticAnnealing,
                                                         0.0, maxRuns, fixedStepSizeOmniDimSynapticChange, updateState,
                                                         getDataClassErr, getDataClassErr,
                                                         initTemp, 1*prod(matrixConfig),
                                                         synMatIn, tanh,
                                                         dataSet, dataInputDimensions, dataOutputDimensions)

# putdata(outTuple_f, "outTuple_f")
# outTuple_f = getdata("outTuple_f")

(meanValErrorVec_f, meanTrainErrorVec_f, meanPerturbDistanceVec_f, minValErrorSynapseMatrix_f) = outTuple_f

plotAnnealResults(meanTrainErrorVec_f, meanValErrorVec_f, "Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")


# plotAnnealResults(meanTrainErrorVec_f, meanPerturbDistanceVec_f./maximum(meanPerturbDistanceVec_f), "Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")




###################################################################################################################################################


outTuple_o = @time nFoldCrossValidateSynapticAnnealingPar(numFolds, matrixConfig, synapticAnnealing,
                                                          0.0, maxRuns, omniDimSynapticChange, updateState,
                                                          getDataClassErr, getDataClassErr,
                                                          initTemp, 1,
                                                          synMatIn,tanh,
                                                          dataSet, dataInputDimensions, dataOutputDimensions)

# putdata(outTuple_o, "outTuple_o")
# outTuple_o = getdata("outTuple_o")

(meanValErrorVec_o, meanTrainErrorVec_o, meanPerturbDistanceVec_o, minValErrorSynapseMatrix_o) = outTuple_o

plotAnnealResults(meanTrainErrorVec_o, meanValErrorVec_o, "Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")



# plotAnnealResults(meanTrainErrorVec_o, meanValErrorVec_o, "Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")





###################################################################################################################################################



outTuple_s =nFoldCrossValidateSynapticAnnealingPar(numFolds, matrixConfig, synapticAnnealing,
                                                   0.0, maxRuns, singleDimSynapticChange, updateState,
                                                   getDataClassErr, getDataClassErr,
                                                   initTemp, 1,
                                                   synMatIn,tanh,
                                                   dataSet, dataInputDimensions, dataOutputDimensions)

putdata(outTuple_s, "outTuple_s")
outTuple_s = getdata("outTuple_s")

(meanValErrorVec_s, meanTrainErrorVec_s, meanPerturbDistanceVec_s, minValErrorSynapseMatrix_s) = outTuple_s

plotAnnealResults(meanTrainErrorVec_s, meanValErrorVec_s, "Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")




function minlimitvec(vector)


	minlimitedvector = Any[]
	minelement = Inf

	for element = vector

		if element < minelement
			minelement = element
		end

		push!(minlimitedvector, minelement)

	end

	return(minlimitedvector)

end


# plot(meanTrainErrorVec_aq, label="Anisotropic Quantum Training Error", alpha=0.7)
plot(minlimitvec(meanValErrorVec_bp), label="Back Prop Validation Error", alpha=0.7)
# plot(meanTrainErrorVec_aq, label="Anisotropic Quantum Training Error", alpha=0.7)
plot(minlimitvec(meanValErrorVec_aq), label="Anisotropic Quantum Validation Error", alpha=0.7)
# plot(meanTrainErrorVec_q, label="Quantum Training Error", alpha=0.7)
plot(minlimitvec(meanValErrorVec_q), label="Quantum Validation Error", alpha=0.7)
# plot(meanTrainErrorVec_q, label="Quantum Training Error", alpha=0.7)
plot(minlimitvec(meanValErrorVec_qo), label="Quantum ONLY Validation Error", alpha=0.7)
# plot(meanTrainErrorVec_o, label="Omnidimensional Training Error", alpha=0.7)
plot(minlimitvec(meanValErrorVec_o), label="Omnidimensional Validation Error", alpha=0.7)
# plot(meanTrainErrorVec_f, label="Fixed-Step Training Error", alpha=0.7)
plot(minlimitvec(meanValErrorVec_f), label="Fixed-Step Validation Error", alpha=0.7)
# plot(meanTrainErrorVec_s, label="Single-Step Training Error", alpha=0.7)
plot(minlimitvec(meanValErrorVec_s), label="Single-Step Validation Error", alpha=0.7)


ylim(0, 1)

xlabel("Training Epoch")
ylabel("25-Fold Cross-Validated Mean Classification Error")
legend(loc=1)
title("Average Classification Error of Various Configuration Space Traversal Techniques")
plt[:show]()


# # plot(meanTrainErrorVec_aq, label="Anisotropic Quantum Training Error", alpha=0.7)
# plot(1:100:maxRuns+1, meanValErrorVec_aq, label="Anisotropic Quantum Validation Error", alpha=0.7)
# # plot(meanTrainErrorVec_q, label="Quantum Training Error", alpha=0.7)
# plot(1:100:maxRuns+1,meanValErrorVec_q, label="Quantum Validation Error", alpha=0.7)
# # plot(meanTrainErrorVec_o, label="Omnidimensional Training Error", alpha=0.7)
# plot(1:100:maxRuns+1,meanValErrorVec_o, label="Omnidimensional Validation Error", alpha=0.7)
# # plot(meanTrainErrorVec_f, label="Fixed-Step Training Error", alpha=0.7)
# plot(1:100:maxRuns+1,meanValErrorVec_f, label="Fixed-Step Validation Error", alpha=0.7)
# # plot(meanTrainErrorVec_s, label="Single-Step Training Error", alpha=0.7)
# plot(1:100:maxRuns+1,meanValErrorVec_s, label="Single-Step Validation Error", alpha=0.7)
# ylim(0, 1)

# xlabel("Training Epoch")
# ylabel("25-Fold Cross-Validated Mean Classification Error")
# legend(loc=1)
# title("Average Classification Error of Various Configuration Space Traversal Techniques")
# plt[:show]()

