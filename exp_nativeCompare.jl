# Pkg.init()
# Pkg.add("PyCall")
# Pkg.build("PyCall")
# Pkg.add("PyPlot")
# Pkg.add("StatsBase")
# Pkg.add("BackpropNeuralNet")
Pkg.build("BackpropNeuralNet")
# Pkg.update()

using PyPlot
rmprocs(workers())
addprocs(2)

# @everywhere using Devectorize

@everywhere using BackpropNeuralNet

@everywhere cd(homedir()*"\\OneDrive\\afit\\rs\\synaptic-annealing")

# Include utility libraries.
@everywhere include("getput.jl")
@everywhere include("vectorListMean.jl")
@everywhere include("vectorListToMatrix.jl")

# Include data maniputlation libraries.
@everywhere include("normalizeData.jl")
@everywhere include("orthogonalizeDataClasses.jl")
@everywhere include("shuffleData.jl")

# Include synaptic annealing libraries.
@everywhere include("createSynapseMatrix.jl")
@everywhere include("propogateForward.jl")
@everywhere include("plotAnnealResults.jl")
@everywhere include("annealingTraversalFunctions.jl")
@everywhere include("stateUpdateFunctions.jl")
@everywhere include("synapticAnnealing.jl")
@everywhere include("errorFunctions.jl")
@everywhere include("getDataPredictions.jl")

# Include native propigation annealing libraries.
@everywhere include("nativeNetsToSynMats.jl")
@everywhere include("nFoldCrossValidateNativeProp.jl")
@everywhere include("synapticAnnealing_nativePropigation.jl")

# Include cross val annealing libraries.
@everywhere include("buildFolds.jl")
@everywhere include("nFoldCrossValidateSynapticAnnealing.jl")

# Include cross val annealing libraries.
@everywhere include("backpropTraining.jl")
@everywhere include("nFoldCrossValidateBackprop.jl")

ion()




###################################################################################################################################################

# Construct the iris dataset
irisData = readdlm(homedir()*"\\OneDrive\\afit\\rs\\synaptic-annealing\\iris.dat", ',' , Any)
irisDataClassed = orthogonalizeDataClasses(irisData, [5])
irisDataClassed = normalizeData(irisDataClassed)
irisDataClassed = shuffleData(irisDataClassed)


###################################################################################################################################################

# Pull the physics training data.
tauThreeMuTrainData = readdlm(homedir()*"\\OneDrive\\afit\\CSCE823\\project\\tauThreeMuTrainData.dat", ',' , Any)
# Clip the last two columns, and the first row.
tauThreeMuTrainData = tauThreeMuTrainData[2:end, 1:end-2]
# Clip the Second from the last column, and the first column
tauThreeMuTrainData = [tauThreeMuTrainData[:, 2:end-2] tauThreeMuTrainData[:, end]]

tauThreeMuTrainData = orthogonalizeDataClasses(tauThreeMuTrainData, [47])
tauThreeMuTrainData = normalizeData(tauThreeMuTrainData)
tauThreeMuTrainData = shuffleData(tauThreeMuTrainData)

subsetSize = int(0.1*size(tauThreeMuTrainData)[1])
tauThreeMuTrainData_subsetTrain = tauThreeMuTrainData[1:subsetSize, :]
tauThreeMuTrainData_subsetTest = tauThreeMuTrainData[(subsetSize+1):end, :]


###################################################################################################################################################

lcvfData = readdlm(homedir()*"\\OneDrive\\afit\\rs\\synaptic-annealing\\lcvfData.csv", ',' , Any)
lcvfDataClassed = orthogonalizeDataClasses(lcvfData, [195])
lcvfDataClassed = normalizeData(lcvfDataClassed)
lcvfDataClassed = shuffleData(lcvfDataClassed)



# ###################################################################################################################################################
# # Partition the iris dataset.
# irisDataTrain = irisDataClassed[1:100, :]
# irisDataVal = irisDataClassed[101:125, :]
# irisDataTest = irisDataClassed[126:end, :]

# synapseMatrixIn = createSynapseMatrix([5,10,3])

# outTuple = @time synapticAnnealing(0.0, 500, fixedStepSizeOmniDimSynapticChange, updateState,
#                                    getDataClassErr, getDataClassErr,
#                                    100, 1,
#                                    synapseMatrixIn, tanh,
#                                    irisDataTrain, irisDataVal,
#                                    [1:4], [5:7])

# (minValErrorSynapseMatrix, validationErrorVector, trainingErrorVector, perturbationDistanceVector) = outTuple

# finalValError = validationErrorVector[end]

# plotAnnealResults(trainingErrorVector,validationErrorVector, "Training and Validation Classification Error of a Quantum Synaptic\n Annealing Neural Network using a Linear Temperature\n Schedule.")


###################################################################################################################################################

dataSet = irisDataClassed

dataInputDimensions = [1:4]

dataOutputDimensions = [5:7]




###################################################################################################################################################

dataSet = lcvfDataClassed

dataInputDimensions = [1:194]

dataOutputDimensions = [195:229]


###################################################################################################################################################


numFolds = 25

maxRuns = 100000

initTemp = 2000

numHiddenLayers = 1

matrixConfig = [length(dataInputDimensions)+1, repmat([length(dataInputDimensions)], numHiddenLayers),length(dataOutputDimensions)]

matrixConfigNative = [length(dataInputDimensions), repmat([length(dataInputDimensions)], numHiddenLayers),length(dataOutputDimensions)]

###################################################################################################################################################
outTuple_bp = @time nFoldCrossValidateBackpropPar(numFolds, matrixConfigNative, synapticAnnealing,
                                                          0.0, maxRuns, quantumAnisotropicSynapticPerturbation, updateState_oscillatory,
                                                          getDataRegErr_nativeProp, getDataClassErr_backprop,
                                                          initTemp, 1,
                                                          tanh,
                                                          dataSet, dataInputDimensions, dataOutputDimensions)

putdata(outTuple_bp, "outTuple_bp")
outTuple_bp = getdata("outTuple_bp")

(meanValErrorVec_bp, meanTrainErrorVec_bp, meanPerturbDistanceVec_bp, minValErrorSynapseMatrix_bp) = outTuple_bp

plotAnnealResults(meanTrainErrorVec_bp, meanValErrorVec_bp, "Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")


###################################################################################################################################################
# synMatIn = minValErrorSynapseMatrix_aq[1]



# minValErrorSynapseMatrix_bp[1].weights

# synMatIn[:,1:4,1] = minValErrorSynapseMatrix_bp[1].weights[1]
# synMatIn[1:end-1,1:3,2] = minValErrorSynapseMatrix_bp[1].weights[2][1:end-1,:]

synMatIn = null
###################################################################################################################################################

outTuple_aq_synMat = @time nFoldCrossValidateSynapticAnnealingPar(numFolds, matrixConfig, synapticAnnealing,
                                                          0.0, maxRuns, quantumAnisotropicSynapticPerturbation, updateState_oscillatory,
                                                          getDataRegErr, getDataClassErr,
                                                          initTemp, 1,
                                                          synMatIn,tanh,
                                                          dataSet, dataInputDimensions, dataOutputDimensions)

# putdata(outTuple_aq, "outTuple_aq_lcvf10800")
# outTuple_aq = getdata("outTuple_aq_lcvf10800")

(meanValErrorVec_aq_synMat, meanTrainErrorVec_aq_synMat, meanPerturbDistanceVec_aq_synMat, minValErrorSynapseMatrix_aq_synMat) = outTuple_aq_synMat

plotAnnealResults(meanTrainErrorVec_aq_synMat, meanValErrorVec_aq_synMat, "Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")



outTuple_aq_natProp = @time nFoldCrossValidateNativePropPar(numFolds, matrixConfigNative, synapticAnnealing_nativePropigation,
                                                         0.0, maxRuns, quantumAnisotropicSynapticPerturbation, updateState_oscillatory,
                                                         getDataClassErr_nativeProp, getDataClassErr_nativeProp,
                                                         initTemp, 1,
                                                         synMatIn,tanh,
                                                         dataSet, dataInputDimensions, dataOutputDimensions)
# putdata(outTuple_f, "outTuple_f")
# outTuple_f = getdata("outTuple_f")

(meanValErrorVec_aq_natProp, meanTrainErrorVec_aq_natProp, meanPerturbDistanceVec_aq_natProp, minValErrorSynapseMatrix_aq_natProp) = outTuple_aq_natProp

plotAnnealResults(meanTrainErrorVec_aq_natProp, meanValErrorVec_aq_natProp, "Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")


###################################################################################################################################################

outTuple_qo = @time nFoldCrossValidateSynapticAnnealingPar(numFolds, matrixConfig, synapticAnnealing,
                                                          0.0, maxRuns, quantumSynapticChange, updateState_q_only,
                                                          getDataClassErr, getDataClassErr,
                                                          initTemp, 1,
                                                          synMatIn,tanh,
                                                          dataSet, dataInputDimensions, dataOutputDimensions)

putdata(outTuple_qo, "outTuple_qo_lcvf1000")
outTuple_qo = getdata("outTuple_qo_lcvf1000")

(meanValErrorVec_qo, meanTrainErrorVec_qo, meanPerturbDistanceVec_qo, minValErrorSynapseMatrix_qo) = outTuple_qo

plotAnnealResults(meanTrainErrorVec_qo, meanValErrorVec_qo, "Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")


###################################################################################################################################################


outTuple_q = @time nFoldCrossValidateSynapticAnnealingPar(numFolds, matrixConfig, synapticAnnealing,
                                                          0.0, maxRuns, quantumSynapticChange, updateState_oscillatory,
                                                          getDataClassErr, getDataClassErr,
                                                          initTemp, 1,
                                                          synMatIn,tanh,
                                                          dataSet, dataInputDimensions, dataOutputDimensions)

putdata(outTuple_q, "outTuple_q")
outTuple_q = getdata("outTuple_q")

(meanValErrorVec_q, meanTrainErrorVec_q, meanPerturbDistanceVec_q, minValErrorSynapseMatrix_q) = outTuple_q

plotAnnealResults(meanTrainErrorVec_q, meanValErrorVec_q, "Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")




###################################################################################################################################################




outTuple_f_synMat = @time nFoldCrossValidateSynapticAnnealingPar(numFolds, matrixConfig, synapticAnnealing,
                                                         0.0, maxRuns, fixedStepSizeOmniDimSynapticChange, updateState,
                                                         getDataClassErr, getDataClassErr,
                                                         initTemp,1,
                                                         synMatIn,tanh,
                                                         dataSet, dataInputDimensions, dataOutputDimensions)

# putdata(outTuple_f, "outTuple_f")
# outTuple_f = getdata("outTuple_f")

(meanValErrorVec_f_synMat, meanTrainErrorVec_f_synMat, meanPerturbDistanceVec_f_synMat, minValErrorSynapseMatrix_f_synMat) = outTuple_f_synMat

plotAnnealResults(meanTrainErrorVec_f_synMat, meanValErrorVec_f_synMat, "Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")



outTuple_f_natProp = @time nFoldCrossValidateNativePropPar(numFolds, matrixConfigNative, synapticAnnealing_nativePropigation,
                                                         0.0, maxRuns, fixedStepSizeOmniDimSynapticChange, updateState,
                                                         getDataClassErr_nativeProp, getDataClassErr_nativeProp,
                                                         initTemp, 1,
                                                         synMatIn,tanh,
                                                         dataSet, dataInputDimensions, dataOutputDimensions)

# putdata(outTuple_f, "outTuple_f")
# outTuple_f = getdata("outTuple_f")

(meanValErrorVec_f_natProp, meanTrainErrorVec_f_natProp, meanPerturbDistanceVec_f_natProp, minValErrorSynapseMatrix_f_natProp) = outTuple_f_natProp

plotAnnealResults(meanTrainErrorVec_f_natProp, meanValErrorVec_f_natProp, "Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")


# plotAnnealResults(meanTrainErrorVec_f, meanPerturbDistanceVec_f./maximum(meanPerturbDistanceVec_f), "Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")




###################################################################################################################################################


outTuple_o = @time nFoldCrossValidateSynapticAnnealingPar(numFolds, matrixConfig, synapticAnnealing,
                                                          0.0, maxRuns, omniDimSynapticChange, updateState,
                                                          getDataClassErr, getDataClassErr,
                                                          initTemp, 1,
                                                          synMatIn,tanh,
                                                          dataSet, dataInputDimensions, dataOutputDimensions)

putdata(outTuple_o, "outTuple_o")
outTuple_o = getdata("outTuple_o")

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

putdata(outTuple_q, "outTuple_s")
outTuple_q = getdata("outTuple_s")

(meanValErrorVec_s, meanTrainErrorVec_s, meanPerturbDistanceVec_s, minValErrorSynapseMatrix_s) = outTuple_s

# plotAnnealResults(meanTrainErrorVec_s, meanValErrorVec_s, "Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")



runTime = toq()
runTime/60/60

runTime


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

