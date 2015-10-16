# Pkg.init()
# Pkg.add("PyCall")
# Pkg.build("PyCall")
# Pkg.add("PyPlot")
# Pkg.add("StatsBase")
# Pkg.update()

using PyPlot
rmprocs(workers())
addprocs(25)

@everywhere using Devectorize
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

# Include cross val annealing libraries.
@everywhere include("buildFolds.jl")
@everywhere include("nFoldCrossValidateSynapticAnnealing.jl")

ion()


###################################################################################################################################################
# ---- Experiment Development Area ----

# Construct the iris dataset
irisData = readdlm(homedir()*"\\OneDrive\\afit\\rs\\synaptic-annealing\\iris.dat", ',' , Any)
irisDataClassed = orthogonalizeDataClasses(irisData, [5])
irisDataClassed = normalizeData(irisDataClassed)
irisDataClassed = shuffleData(irisDataClassed)



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


numFolds = 10

maxRuns = 2

initTemp = 800

# dataSet = irisDataClassed

# dataInputDimensions = [1:4]

# dataOutputDimensions = [5:7]


dataSet = lcvfDataClassed

dataInputDimensions = [1:194]

dataOutputDimensions = [195:229]


matrixConfig = [length(dataInputDimensions)+1,length(dataInputDimensions),length(dataOutputDimensions)]

tic()

###################################################################################################################################################

outTuple_aq = @time nFoldCrossValidateSynapticAnnealingPar(numFolds, matrixConfig, synapticAnnealing,
                                                          0.0, maxRuns, quantumAnisotropicSynapticPerturbation, updateState_oscillatory,
                                                          getDataClassErr, getDataClassErr,
                                                          initTemp, 1,
                                                          tanh,
                                                          dataSet, dataInputDimensions, dataOutputDimensions)

putdata(outTuple_aq, "outTuple_aq")
outTuple_aq = getdata("outTuple_aq")

(meanValErrorVec_aq, meanTrainErrorVec_aq, meanPerturbDistanceVec_aq, minValErrorSynapseMatrix_aq) = outTuple_aq

plotAnnealResults(meanTrainErrorVec_aq, meanValErrorVec_aq, "Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")

###################################################################################################################################################

outTuple_aq = @time nFoldCrossValidateSynapticAnnealingPar(numFolds, matrixConfig, synapticAnnealing,
                                                          0.0, maxRuns, quantumSynapticPerturbation, updateState_q_only,
                                                          getDataClassErr, getDataClassErr,
                                                          initTemp, 1,
                                                          tanh,
                                                          dataSet, dataInputDimensions, dataOutputDimensions)

putdata(outTuple_aq, "outTuple_aq")
outTuple_aq = getdata("outTuple_aq")

(meanValErrorVec_aq, meanTrainErrorVec_aq, meanPerturbDistanceVec_aq, minValErrorSynapseMatrix_aq) = outTuple_aq

plotAnnealResults(meanTrainErrorVec_aq, meanValErrorVec_aq, "Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")


8751/60/60
###################################################################################################################################################


outTuple_q = @time nFoldCrossValidateSynapticAnnealingPar(numFolds, matrixConfig, synapticAnnealing,
                                                          0.0, maxRuns, quantumSynapticChange, updateState_oscillatory,
                                                          getDataClassErr, getDataClassErr,
                                                          initTemp, 1,
                                                          tanh,
                                                          dataSet, dataInputDimensions, dataOutputDimensions)

putdata(outTuple_q, "outTuple_q")
outTuple_q = getdata("outTuple_q")

(meanValErrorVec_q, meanTrainErrorVec_q, meanPerturbDistanceVec_q, minValErrorSynapseMatrix_q) = outTuple_q

# plotAnnealResults(meanTrainErrorVec_q, meanValErrorVec_q, "Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")




###################################################################################################################################################




outTuple_f = @time nFoldCrossValidateSynapticAnnealingPar(numFolds, matrixConfig, synapticAnnealing,
                                                         0.0, maxRuns, fixedStepSizeOmniDimSynapticChange, updateState,
                                                         getDataClassErr, getDataClassErr,
                                                         initTemp,1,
                                                         tanh,
                                                         dataSet, dataInputDimensions, dataOutputDimensions)

putdata(outTuple_f, "outTuple_f")
outTuple_f = getdata("outTuple_f")

(meanValErrorVec_f, meanTrainErrorVec_f, meanPerturbDistanceVec_f, minValErrorSynapseMatrix_f) = outTuple_f

plotAnnealResults(meanTrainErrorVec_f, meanValErrorVec_f, "Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")


# plotAnnealResults(meanTrainErrorVec_f, meanPerturbDistanceVec_f./maximum(meanPerturbDistanceVec_f), "Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")






Matrix_o) = outTuple_o

# plotAnnealResults(meanTrainErrorVec_o, meanValErrorVec_o, "Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")





###################################################################################################################################################



outTuple_s =nFoldCrossValidateSynapticAnnealingPar(numFolds, matrixConfig, synapticAnnealing,
                                                   0.0, maxRuns, singleDimSynapticChange, updateState,
                                                   getDataClassErr, getDataClassErr,
                                                   initTemp, 1,
                                                   tanh,
                                                   dataSet, dataInputDimensions, dataOutputDimensions)

putdata(outTuple_q, "outTuple_s")
outTuple_q = getdata("outTuple_s")

(meanValErrorVec_s, meanTrainErrorVec_s, meanPerturbDistanceVec_s, minValErrorSynapseMatrix_s) = outTuple_s

# plotAnnealResults(meanTrainErrorVec_s, meanValErrorVec_s, "Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")



runTime = toq()
runTime/60/60

runTime

# plot(meanTrainErrorVec_aq, label="Anisotropic Quantum Training Error", alpha=0.7)
plot(meanValErrorVec_aq, label="Anisotropic Quantum Validation Error", alpha=0.7)
# plot(meanTrainErrorVec_q, label="Quantum Training Error", alpha=0.7)
plot(meanValErrorVec_q, label="Quantum Validation Error", alpha=0.7)
# plot(meanTrainErrorVec_o, label="Omnidimensional Training Error", alpha=0.7)
plot(meanValErrorVec_o, label="Omnidimensional Validation Error", alpha=0.7)
# plot(meanTrainErrorVec_f, label="Fixed-Step Training Error", alpha=0.7)
plot(meanValErrorVec_f, label="Fixed-Step Validation Error", alpha=0.7)
# plot(meanTrainErrorVec_s, label="Single-Step Training Error", alpha=0.7)
plot(meanValErrorVec_s, label="Single-Step Validation Error", alpha=0.7)
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

