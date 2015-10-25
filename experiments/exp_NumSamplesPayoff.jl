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
# ---- Experiment Development Area ----

# Construct the iris dataset
irisData = readdlm(homedir()*"\\OneDrive\\afit\\rs\\synapticAnnealing\\iris.dat", ',' , Any)
irisDataClassed = orthogonalizeDataClasses(irisData, [5])
irisDataClassed = normalizeData(irisDataClassed)
irisDataClassed = shuffleData(irisDataClassed)

tempSamples = 6000

tempDims = 48

tempData = rand(tempSamples,tempDims)

@time getDataClassErr(createSynapseMatrix([tempDims+1,tempDims,1]), tanh, tempData, [1:tempDims-1], [tempDims])

@time getDataClassErrPar(createSynapseMatrix([tempDims+1,tempDims,1]), tanh, tempData, [1:tempDims-1], [tempDims])

