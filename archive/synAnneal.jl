# Pkg.init()
# Pkg.add("PyCall")
# Pkg.build("PyCall")
# Pkg.add("PyPlot")
# Pkg.add("StatsBase")
# Pkg.update()

using PyPlot

addprocs(28)

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
@everywhere include("synapticAnnealing.jl")

# Include cross val annealing libraries.
@everywhere include("buildFolds.jl")
@everywhere include("nFoldCrossValidateSynapticAnnealing.jl")

ion()

###################################################################################################################################################################################################
# ---- Experiment Development Area ----

# Construct the iris dataset
irisData = readdlm("C:\\Users\\serg\\OneDrive\\afit\\rs\\synapticAnnealing\\iris.dat", ',' , Any)
irisDataClassed = orthogonalizeDataClasses(irisData, [5])
irisDataClassed = normalizeData(irisDataClassed)
irisDataClassed = shuffleData(irisDataClassed)


(meanValErrorVec_q, meanTrainErrorVec_q, minValErrorSynapseMatrix_q) = @time nFoldCrossValidateSynapticAnnealing(50, [5,10,3], synapticAnnealing,
                                                                                                      0.0, 10000, quantumSynapticChange,
                                                                                                      getDataClassErr, getDataClassErr,
                                                                                                      1000, decayUnityZeroBound,
                                                                                                      1, expDecay,
                                                                                                      tanh,
                                                                                                      irisDataClassed,
                                                                                                      [1:4], [5:7])

putdata(meanTrainErrorVec_q, "meanTrainErrorVec_q2")
putdata(meanValErrorVec_q, "meanValErrorVec_q2")
putdata(minValErrorSynapseMatrix_q, "synapseMat_q2")

meanTrainErrorVec_q = getdata("meanTrainErrorVec_q2")
meanValErrorVec_q = getdata("meanValErrorVec_q2")
minValErrorSynapseMatrix_q = getdata("synapseMat_q2")

finalValError = meanValErrorVec_q[end]
plotAnnealResults(meanTrainErrorVec_q, meanValErrorVec_q, "Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")






(meanValErrorVec_o, meanTrainErrorVec_o, minValErrorSynapseMatrix_o) = @time nFoldCrossValidateSynapticAnnealing(50, [5,10,3], synapticAnnealing,
                                                                                                      0.0, 10000, omniDimSynapticChange,
                                                                                                      getDataClassErr, getDataClassErr,
                                                                                                      1000, decayUnityZeroBound,
                                                                                                      1, expDecay,
                                                                                                      tanh,
                                                                                                      irisDataClassed,
                                                                                                      [1:4], [5:7])

putdata(meanTrainErrorVec_o, "meanTrainErrorVec_o2")
putdata(meanValErrorVec_o, "meanValErrorVec_o2")
putdata(minValErrorSynapseMatrix_o, "synapseMat_o2")


meanTrainErrorVec_o = getdata("meanTrainErrorVec_o2")
meanValErrorVec_o = getdata("meanValErrorVec_o2")
minValErrorSynapseMatrix_o = getdata("synapseMat_o2")

finalValError = meanValErrorVec_o[end]
plotAnnealResults(meanTrainErrorVec_o, meanValErrorVec_o, "Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")




(meanValErrorVec_f, meanTrainErrorVec_f, minValErrorSynapseMatrix_f) = @time nFoldCrossValidateSynapticAnnealing(50, [5,10,3], synapticAnnealing,
                                                                                                      0.0, 10000, fixedStepSizeOmniDimSynapticChange,
                                                                                                      getDataClassErr, getDataClassErr,
                                                                                                      1000, decayUnityZeroBound,
                                                                                                      1, expDecay,
                                                                                                      tanh,
                                                                                                      irisDataClassed,
                                                                                                      [1:4], [5:7])

putdata(meanTrainErrorVec_f, "meanTrainErrorVec_f2")
putdata(meanValErrorVec_f, "meanValErrorVec_f2")
putdata(minValErrorSynapseMatrix_f, "synapseMat_f2")


meanTrainErrorVec_f = getdata("meanTrainErrorVec_f2")
meanValErrorVec_f = getdata("meanValErrorVec_f2")
minValErrorSynapseMatrix_f = getdata("synapseMat_f2")

finalValError = meanValErrorVec_f[end]
plotAnnealResults(meanTrainErrorVec_f, meanValErrorVec_f, "Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")




(meanValErrorVec, meanTrainErrorVec, minValErrorSynapseMatrix) = nFoldCrossValidateSynapticAnnealingPar(5, [5,10,3], synapticAnnealing,
                                                                                                      0.0, 10, singleDimSynapticChange,
                                                                                                      getDataClassErr, getDataClassErr,
                                                                                                      500, decayUnityZeroBound,
                                                                                                      1, expDecay,
                                                                                                      tanh,
                                                                                                      irisDataClassed,
                                                                                                      [1:4], [5:7])


plot(meanTrainErrorVec_q, label="Quantum Training Error", alpha=0.7)
# plot(meanValErrorVec_q, label="Quantum Validation Error", alpha=0.7)
plot(meanTrainErrorVec_o, label="Omnidimensional Training Error", alpha=0.7)
# plot(meanValErrorVec_o, label="Omnidimensional Validation Error", alpha=0.7)
plot(meanTrainErrorVec_f, label="Fixed-Step Training Error", alpha=0.7)
# plot(meanValErrorVec_f, label="Fixed-Step Validation Error", alpha=0.7)
ylim(0, maximum([maximum(meanTrainErrorVec_q), maximum(meanValErrorVec_q)]))
xlabel("Training Epoch")
ylabel("50-Fold Cross-Validated Mean Classification Error")
legend(loc=3)
title("Average Classification Error of Various Configuration Space Traversal Techniques")
plt[:show]()



putdata(meanTrainErrorVec, "meanTrainErrorVec_s")
putdata(meanValErrorVec, "meanValErrorVec_s")
putdata(minValErrorSynapseMatrix, "synapseMat_s")

meanTrainErrorVec = getdata("meanTrainErrorVec_s")
meanValErrorVec = getdata("meanValErrorVec_s")
minValErrorSynapseMatrix = getdata("synapseMat_s")

finalValError = meanValErrorVec[end]
plotAnnealResults(meanTrainErrorVec, meanValErrorVec, "Training and Validation Classification Error\n of a Synaptic Annealing Neural Network")


