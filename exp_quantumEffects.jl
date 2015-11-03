# Pkg.init()
# Pkg.add("PyCall")
# Pkg.build("PyCall")
# Pkg.add("PyPlot")
# Pkg.add("StatsBase")

Pkg.add("Devectorize")
Pkg.update()

using PyPlot
rmprocs(workers())
addprocs(25)

@everywhere using Devectorize
@everywhere cd(homedir()*"\\OneDrive\\afit\\rs\\synapticAnnealing")

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

tempSynMat = createSynapseMatrix([5,10,3])
randMat = rand(size(tempSynMat)).*int(bool(tempSynMat))

normMat = randMat./sum(randMat)
sum(normMat)

anisotropicField = 0.0
anisotropicField = 0.25
anisotropicField = 0.5
anisotropicField = 0.75
anisotropicField = 1

anisotropicMat = normMat.^(1/(1-(0.99*anisotropicField)))

sum(anisotropicMat)

anisotropicNormMat = anisotropicMat./sum(anisotropicMat)

sum(anisotropicNormMat)

###################################################################################################################################################
# ---- Experiment Development Area ----

# Construct the iris dataset
irisData = readdlm(homedir()*"\\OneDrive\\afit\\rs\\synapticAnnealing\\iris.dat", ',' , Any)
irisDataClassed = orthogonalizeDataClasses(irisData, [5])
irisDataClassed = normalizeData(irisDataClassed)
irisDataClassed = shuffleData(irisDataClassed)

###################################################################################################################################################

# Pull the physics training data.
tauThreeMuTrainData = readdlm("C:\\Users\\serg\\OneDrive\\afit\\CSCE823\\project\\tauThreeMuTrainData.dat", ',' , Any)
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




@time getDataClassErr(createSynapseMatrix([5,4,3]), tanh, irisDataClassed, [1:4], [5:7])

###################################################################################################################################################
###### Experiment: how does quantum-only compare to QSA?
outTuple_q_only = @time nFoldCrossValidateSynapticAnnealingPar(5, [47,46,2], synapticAnnealing,
                                                          0.0, 100, quantumSynapticChange, updateState_q_only,
                                                          getDataClassErrPar, getDataClassErrPar,
                                                          1, 1, tanh,
                                                          tauThreeMuTrainData_subsetTrain, [1:46], [47:48])

(meanValErrorVec_q_only, meanTrainErrorVec_q_only, meanPerturbDistanceVec_q_only, minValErrorSynapseMatrix_q_only) = outTuple_q_only

plotAnnealResults(meanTrainErrorVec_q_only, meanValErrorVec_q_only, "Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")

# putdata(outTuple_q, "outTuple_q")
# outTuple_q = getdata("outTuple_q")

outTuple_q = @time nFoldCrossValidateSynapticAnnealingPar(25, [47,46,2], synapticAnnealing,
                                                          0.0, 15000, quantumSynapticChange, updateState_q,
                                                          getDataClassErr, getDataClassErr,
                                                          100, 1, tanh,
                                                          tauThreeMuTrainData_subsetTrain, [1:46], [47:48])

(meanValErrorVec_q, meanTrainErrorVec_q, meanPerturbDistanceVec_q, minValErrorSynapseMatrix_q) = outTuple_q

plotAnnealResults(meanTrainErrorVec_q, meanValErrorVec_q, "Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")

# putdata(outTuple_q, "outTuple_q")
# outTuple_q = getdata("outTuple_q")


outTuple_q = @time nFoldCrossValidateSynapticAnnealingPar(25, [5,10,10,10,3], synapticAnnealing,
                                                          0.0, 100, quantumAnisotropicSynapticPerturbation, updateState_q_unified,
                                                          getDataClassErr, getDataClassErr,
                                                          8000, 1, tanh,
                                                          irisDataClassed, [1:4], [5:7])

(meanValErrorVec_q, meanTrainErrorVec_q, meanPerturbDistanceVec_q, minValErrorSynapseMatrix_q) = outTuple_q

plotAnnealResults(meanTrainErrorVec_q, meanValErrorVec_q, "Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")

putdata(outTuple_q, "outTuple_qasa_15K_f100_ph3")
# putdata(outTuple_q, "outTuple_qa_1M_f100_ph3")
outTuple_q = getdata("outTuple_qa_1M_f100_ph3")


outTuple_q = @time nFoldCrossValidateSynapticAnnealingPar(5, [5,10,3], synapticAnnealing,
                                                          0.0, 1000, quantumSynapticChange, updateState_q_unified,
                                                          getDataClassErr, getDataClassErr,
                                                          8000, 1, tanh,
                                                          irisDataClassed, [1:4], [5:7])

(meanValErrorVec_q, meanTrainErrorVec_q, meanPerturbDistanceVec_q, minValErrorSynapseMatrix_q) = outTuple_q

plotAnnealResults(meanTrainErrorVec_q, meanValErrorVec_q, "Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")

# putdata(outTuple_q, "outTuple_q")
# outTuple_q = getdata("outTuple_q")


plot(meanTrainErrorVec_q, label="Quantum Anisotropic Training Error", alpha=0.7)
plot(meanValErrorVec_q, label="Quantum Anisotropic Validation Error", alpha=0.7)
ylim(0, 0.2)
xlabel("Training Epoch")
ylabel("25-Fold Cross-Validated Mean Classification Error")
legend(loc=3)
title("Average Classification Error of Various Configuration Space Traversal Techniques")
plt[:show]()






###################################################################################################################################################
###### Experiment: how do quanum effects interact with synapse matrix configuration. How does SA or backprop do?
outTuple_q_only = @time nFoldCrossValidateSynapticAnnealingPar(5, [5,4,3], synapticAnnealing,
                                                          0.0, 100, quantumSynapticChange, updateState_q_only,
                                                          getDataClassErr, getDataClassErr,
                                                          1, 1, tanh,
                                                          irisDataClassed, [1:4], [5:7])

(meanValErrorVec_q_only, meanTrainErrorVec_q_only, meanPerturbDistanceVec_q_only, minValErrorSynapseMatrix_q_only) = outTuple_q_only

plotAnnealResults(meanTrainErrorVec_q_only, meanValErrorVec_q_only, "Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")

# putdata(outTuple_q, "outTuple_q")
# outTuple_q = getdata("outTuple_q")

outTuple_q_only = @time nFoldCrossValidateSynapticAnnealingPar(25, [5,4,4,4,3], synapticAnnealing,
                                                          0.0, 1000, quantumSynapticChange, updateState_q_only,
                                                          getDataClassErr, getDataClassErr,
                                                          1, 1, tanh,
                                                          irisDataClassed, [1:4], [5:7])

(meanValErrorVec_q_only, meanTrainErrorVec_q_only, meanPerturbDistanceVec_q_only, minValErrorSynapseMatrix_q_only) = outTuple_q_only

plotAnnealResults(meanTrainErrorVec_q_only, meanValErrorVec_q_only, "Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")

# putdata(outTuple_q, "outTuple_q")
# outTuple_q = getdata("outTuple_q")

outTuple_q_only = @time nFoldCrossValidateSynapticAnnealingPar(25, [5,4,4,4,4,4,4,4,4,4,4,3], synapticAnnealing,
                                                          0.0, 1000, quantumSynapticChange, updateState_q_only,
                                                          getDataClassErr, getDataClassErr,
                                                          1, 1, tanh,
                                                          irisDataClassed, [1:4], [5:7])

(meanValErrorVec_q_only, meanTrainErrorVec_q_only, meanPerturbDistanceVec_q_only, minValErrorSynapseMatrix_q_only) = outTuple_q_only

plotAnnealResults(meanTrainErrorVec_q_only, meanValErrorVec_q_only, "Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")

# putdata(outTuple_q, "outTuple_q")
# outTuple_q = getdata("outTuple_q")

outTuple_q = @time nFoldCrossValidateSynapticAnnealingPar(25, [5,4,3], synapticAnnealing,
                                                          0.0, 1000, quantumSynapticChange, updateState_q,
                                                          getDataClassErr, getDataClassErr,
                                                          100, 1, tanh,
                                                          irisDataClassed, [1:4], [5:7])

(meanValErrorVec_q, meanTrainErrorVec_q, meanPerturbDistanceVec_q, minValErrorSynapseMatrix_q) = outTuple_q

plotAnnealResults(meanTrainErrorVec_q, meanValErrorVec_q, "Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")

# putdata(outTuple_q, "outTuple_q")
# outTuple_q = getdata("outTuple_q")
