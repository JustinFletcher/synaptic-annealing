Pkg.init()
Pkg.add("PyCall")
Pkg.build("PyCall")
Pkg.add("PyPlot")
Pkg.build("PyPlot")
# Pkg.add("StatsBase")
Pkg.add("BackpropNeuralNet")
Pkg.add("MNIST")


using PyPlot

rmprocs(workers())
addprocs(25)
# @everywhere using Devectorize

@everywhere using BackpropNeuralNet
@everywhere cd("\\fletcher-thesis")

pwd()

@everywhere fileList = ("$(pwd())\\src\\"*"ExperimentDataset.jl"),
                        ("$(pwd())\\src\\"*"AnnealingState.jl"),
                        ("$(pwd())\\src\\"*"getput.jl"),
                        ("$(pwd())\\src\\"*"vectorListMean.jl"),
                        ("$(pwd())\\src\\"*"vectorListToMatrix.jl"),
                        ("$(pwd())\\src\\"*"plotLib.jl"),
                        ("$(pwd())\\src\\"*"gammaStats.jl"),
                        ("$(pwd())\\src\\"*"movingWindowAverage.jl"),
                        ("$(pwd())\\src\\"*"normalizeData.jl"),
                        ("$(pwd())\\src\\"*"removeDataMean.jl"),
                        ("$(pwd())\\src\\"*"orthogonalizeDataClasses.jl"),
                        ("$(pwd())\\src\\"*"shuffleData.jl"),
                        ("$(pwd())\\src\\"*"createSynapseMatrix.jl"),
                        ("$(pwd())\\src\\"*"propogateForward.jl"),
                        ("$(pwd())\\src\\"*"annealingTraversalFunctions.jl"),
                        ("$(pwd())\\src\\"*"synapticAnnealing.jl"),
                        ("$(pwd())\\src\\"*"errorFunctions.jl"),
                        ("$(pwd())\\src\\"*"getDataPredictions.jl"),
                        ("$(pwd())\\src\\"*"nativeNetsToSynMats.jl"),
                        ("$(pwd())\\src\\"*"buildFolds.jl"),
                        ("$(pwd())\\src\\"*"nFoldCrossValidateSynapticAnnealing.jl"),
                        ("$(pwd())\\src\\"*"backpropTraining.jl"),
                        ("$(pwd())\\src\\"*"nFoldCrossValidateBackprop.jl")


@everywhere include(fileList)




@everywhere include("$(pwd())\\src\\"*"ExperimentDataset.jl")
@everywhere include("$(pwd())\\src\\"*"AnnealingState.jl")

# Include utility libraries.
@everywhere include("$(pwd())\\src\\"*"getput.jl")
@everywhere include("$(pwd())\\src\\"*"vectorListMean.jl")
@everywhere include("$(pwd())\\src\\"*"vectorListToMatrix.jl")
@everywhere include("$(pwd())\\src\\"*"plotLib.jl")
@everywhere include("$(pwd())\\src\\"*"gammaStats.jl")
@everywhere include("$(pwd())\\src\\"*"movingWindowAverage.jl")

# Include data maniputlation libraries.
@everywhere include("$(pwd())\\src\\"*"normalizeData.jl")
@everywhere include("$(pwd())\\src\\"*"removeDataMean.jl")
@everywhere include("$(pwd())\\src\\"*"orthogonalizeDataClasses.jl")
@everywhere include("$(pwd())\\src\\"*"shuffleData.jl")

# Include synaptic annealing libraries.
@everywhere include("$(pwd())\\src\\"*"createSynapseMatrix.jl")
@everywhere include("$(pwd())\\src\\"*"propogateForward.jl")
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

irisDataset = ExperimentDataset.Dataset(irisDatapath, dataInputDimensions, dataOutputDimensions, "Iris")
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

lcvfDataset = ExperimentDataset.Dataset(lcvfDatapath, dataInputDimensions, dataOutputDimensions, "LCVF")

###################################################################################################################################################


###################################################################################################################################################

wineData = readdlm("$(pwd())\\data\\wine.dat", ',' , Any)
wineDataClassed = orthogonalizeDataClasses(wineData, [195])
wineDataClassed = normalizeData(wineDataClassed)
wineDataClassed = shuffleData(wineDataClassed)


wineDatapath = "$(pwd())\\data\\wine.dat"
dataInputDimensions = [1:13]
dataOutputDimensions = [14]

wineDataset = ExperimentDataset.Dataset(wineDatapath, dataInputDimensions, dataOutputDimensions, "Wine")

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

mnistDataset = ExperimentDataset.Dataset(mnistTrainData[1:150, :], dataInputDimensions, dataOutputDimensions, "MNIST")


###################################################################################################################################################

dataSet = irisDataset

###################################################################################################################################################

dataSet = lcvfDataset

###################################################################################################################################################

dataSet = wineDataset

###################################################################################################################################################

dataSet = mnistDataset

###################################################################################################################################################

dataSetList = Any[wineDataset, irisDataset]
###################################################################################################################################################


numFolds = 25

maxRuns = 1000

initTemp = 500

numHiddenLayers = 1

matrixConfig = [length(dataSet.inputCols), repmat([length(dataSet.inputCols)], numHiddenLayers), length(dataSet.outputCols)]

matrixConfig = [length(dataSet.inputCols), 50, length(dataSet.outputCols)]

synMatIn = null

batchSize = 150

batchSize = size(dataSet.data)[1]

reportFrequency = 10

generate = false
saveData = true
loadData = true
view = true
savePlots = true


###################################################################################################################################################


function runSynapticAnnealingExperiment(expTitle, generate, saveData, loadData, view, savePlots, windowSize, xmax, ymax,
                                        visitDistTag, anisotropicityTag, experimentTag,
                                        numFolds, matrixConfig, annealingFunction,
                                        cutoffError, maxRuns, neighborhoodFunction, stateUpdate,
                                        trainErrorFunction, reportErrorFunction,
                                        initTemp, initLearnRate,
                                        synMatIn, actFun,
                                        dataSet, batchSize, reportFrequency)

    # TODO: Replace with an acculator.
    annealingOutput = null

    # Turn off interactive mode, if it's on, to tightly control plotting.
    ioff()

    # Destroy any lingering figures that may interfer with result saving.
    close("all")

    viewFig = figure("ViewFig")
    validationErrorFig = figure("ValidationErrorFig")
    completeRateFig = figure("CompleteRateFig")

    colorList = ["blue","green","red","cyan"]
    datasetNum = 0


    for dataSet in dataSetList

        datasetNum+=1
        # TODO: need to parameterize by percentage.
        # Set the batch size for this dataset.
        batchSize = size(dataSet.data)[1]

        # TODO: need to paramerize by hidden layer generation function.
        # Create a matrix configuration for this dataset.
        matrixConfig = [length(dataSet.inputCols), 50, length(dataSet.outputCols)]

        # If we've been asked to generate new data for this experiment.
        if (generate)

            annealingOutput = @time nFoldCrossValidateSynapticAnnealingPar(numFolds, matrixConfig, annealingFunction,
                                                                           cutoffError, maxRuns, neighborhoodFunction, stateUpdate,
                                                                           trainErrorFunction, reportErrorFunction,
                                                                           initTemp, initLearnRate,
                                                                           synMatIn, actFun,
                                                                           dataSet, batchSize, reportFrequency)

        end

        # If we've generated new data, and that data is
        if (generate && saveData)
            putdata(annealingOutput, expTitle*"_"*dataSet.name)
        end

        if (loadData)
            annealingOutput = getdata(expTitle*"_"*dataSet.name)
        end

        # Unpack the annealing output.
        (meanValErrorVec, stdValErrorVec, meanTrainErrorVec, stdTrainErrorVec, meanPerturbDistanceVec, minValErrorSynapseMatrix) = annealingOutput

        # Calculate the stats.
        (toptExpVal, toptStd) = calcPerfectClassStats(meanPerturbDistanceVec)


        # If we're viewing, display the plots.
        if (view)

            figure("ViewFig")
            subplot(2,2,1)
            plotCompleteRate(meanPerturbDistanceVec./numFolds, expTitle, dataSet.name, xmax, ymax)
            subplot(2,2,2)
            plotCentralMovingAverageValError(meanValErrorVec, stdValErrorVec,  windowSize, reportFrequency, expTitle, dataSet.name, xmax, ymax, colorList[datasetNum])
            subplot(2,2,3)
            plotAnnealResults(meanTrainErrorVec, meanValErrorVec, reportFrequency, expTitle, dataSet.name, xmax, ymax)
            subplot(2,2,4)
            plotGammaDistPDFfromVector(meanPerturbDistanceVec, expTitle, xmax)

        end

        if (savePlots)

            # Check if the desired director exists, if it doesn't, make it.
            if(!isdir("$(pwd())\\plots\\"*experimentTag))
                mkdir("$(pwd())\\plots\\"*experimentTag)
            end

            figure("ValidationErrorFig")
            plotCentralMovingAverageValError(meanValErrorVec, stdValErrorVec, windowSize, reportFrequency, expTitle, dataSet.name, xmax, ymax, colorList[datasetNum])

              # Check if the desired director exists, if it doesn't, make it.
            if(!isdir("$(pwd())\\plots\\"*experimentTag*"\\"*"classPerf\\"))
                mkdir("$(pwd())\\plots\\"*experimentTag*"\\"*"classPerf\\")
            end
            valErrorSavePath = "$(pwd())\\plots\\"*experimentTag*"\\"*"classPerf\\"*"classPerf_"*visitDistTag*"_"*anisotropicityTag*".png"
            savefig(valErrorSavePath)

            figure("CompleteRateFig")
            plotCompleteRate(meanPerturbDistanceVec./numFolds, expTitle, dataSet.name, xmax, ymax)

            # Check if the desired director exists, if it doesn't, make it.
            if(!isdir("$(pwd())\\plots\\"*experimentTag*"\\"*"completeFrac\\"))
                mkdir("$(pwd())\\plots\\"*experimentTag*"\\"*"completeFrac\\")
            end
            valErrorSavePath = "$(pwd())\\plots\\"*experimentTag*"\\"*"completeFrac\\"*"completeFrac_"*visitDistTag*"_"*anisotropicityTag*".png"
            savefig(valErrorSavePath)
        end

    end


    # Close the saving-only figures.
    close("ValidationErrorFig")
    close("CompleteRateFig")

    # Turn interactive mode back on, as this is our default state.
    ion()

    # Show the remaining figures.
    show()

    return(annealingOutput)
end



###################################################################################################################################################

# experimentList = Any[]

# for experiment in experimentList

#   cutoffError = 0.0
#   (expTitle, visitDistTag, anisotropicityTag, experimentTag, annealingFunction) =
#   experimentResults = runSynapticAnnealingExperiment(expTitle, generate, saveData, loadData, view, savePlots, windowSize, xmax, ymax,
#                                                      visitDistTag, anisotropicityTag, experimentTag,
#                                                      numFolds, matrixConfig, annealingFunction,
#                                                      cutoffError, maxRuns, neighborhoodFunction, stateUpdate,
#                                                      trainErrorFunction, reportErrorFunction,
#                                                      initTemp, initLearnRate,
#                                                      synMatIn, actFun,
#                                                      dataSet, batchSize, reportFrequency)

# end


# ###################################################################################################################################################



outTuple_g_i  = @time runSynapticAnnealingExperiment("Gaussian - Isotropic", generate, saveData, loadData, savePlots, view, 3, maxRuns, 1,
                                                     "GaussianVisit", "IsotropicAnisotropicity", "comparitive_simulated_annealing",
                                                     numFolds, matrixConfig, synapticAnnealing,
                                                     0.0, maxRuns, gaussian_Isotropic_SynapticPerturbation, AnnealingState.updateState_csa,
                                                     getDataClassErr, getDataClassErr,
                                                     initTemp, 1,
                                                     synMatIn, tanh,
                                                     dataSet, batchSize, reportFrequency)

# putdata(outTuple_g_i, "outTuple_g_i")
# outTuple_g_i = getdata("outTuple_g_i")

(meanValErrorVec_g_i, meanTrainErrorVec_g_i, meanPerturbDistanceVec_g_i, minValErrorSynapseMatrix_g_i) = outTuple_g_i

plotCompleteRate(meanPerturbDistanceVec_g_i./numFolds, "Gaussian - Isotropic")

plotGammaDistPDFfromVector(meanPerturbDistanceVec_g_i, "Gaussian - Isotropic")

(toptExpVal_g_i, toptStd_g_i) = calcPerfectClassStats(meanPerturbDistanceVec_g_i)

plotAnnealResults(meanTrainErrorVec_g_i, meanValErrorVec_g_i, reportFrequency, "Gaussian - Isotropic")


###################################################################################################################################################



outTuple_g_ua = @time runSynapticAnnealingExperiment("Gaussian - Uniform Anisotropicity", generate, saveData, loadData, savePlots, view, 5, maxRuns, 1,
                                                     "GaussianVisit", "UniformAnisotropicity", "comparitive_simulated_annealing",
                                                     numFolds, matrixConfig, synapticAnnealing,
                                                     0.0, maxRuns, gaussian_uniformAnisotropic_SynapticPerturbation, AnnealingState.updateState_csa,
                                                     getDataClassErr, getDataClassErr,
                                                     initTemp, 1,
                                                     synMatIn, tanh,
                                                     dataSetList, batchSize, reportFrequency)

# putdata(outTuple_g_ua, "outTuple_g_ua")
# outTuple_g_ua = getdata("outTuple_g_ua")

(meanValErrorVec_g_ua, meanTrainErrorVec_g_ua, meanPerturbDistanceVec_g_ua, minValErrorSynapseMatrix_g_ua) = outTuple_g_ua

(meanValErrorVec, stdValErrorVec, meanTrainErrorVec, stdTrainErrorVec, meanPerturbDistanceVec, minValErrorSynapseMatrix) = outTuple_g_ua

(toptExpVal_g_ua, toptStd_g_ua) = calcPerfectClassStats(meanPerturbDistanceVec_g_ua)


subplot(2,2,1)

plotAnnealResults(meanTrainErrorVec_g_ua, meanValErrorVec_g_ua, reportFrequency, "Gaussian - Uniform Anisotropicity")

subplot(2,2,2)

plotCmaAnnealingResults(meanValErrorVec_g_ua, 9, reportFrequency, "Gaussian - Uniform Anisotropicity")

subplot(2,2,3)

plotCompleteRate(meanPerturbDistanceVec_g_ua./numFolds, "Gaussian - Uniform Anisotropicity")

subplot(2,2,4)

plotGammaDistPDFfromVector(meanPerturbDistanceVec_g_ua, "Gaussian - Uniform Anisotropicity")





###################################################################################################################################################

outTuple_g_va  = @time runSynapticAnnealingExperiment("Gaussian - Variable Anisotropicity", generate, saveData, loadData, savePlots, view, 3, maxRuns, 1,
                                                      "GaussianVisit", "VariableAnisotropicity", "comparitive_simulated_annealing",
                                                      numFolds, matrixConfig, synapticAnnealing,
                                                      0.0, maxRuns, gaussian_variableAnisotropic_SynapticPerturbation, AnnealingState.updateState_fsa_anisotropicity,
                                                      getDataClassErr, getDataClassErr,
                                                      initTemp, 1,
                                                      synMatIn, tanh,
                                                      dataSet, batchSize, reportFrequency)

# putdata(outTuple_g_va, "outTuple_g_va")
# outTuple_g_va = getdata("outTuple_g_va")

(meanValErrorVec_g_va, meanTrainErrorVec_g_va, meanPerturbDistanceVec_g_va, minValErrorSynapseMatrix_g_va) = outTuple_g_va

plotCompleteRate(meanPerturbDistanceVec_g_va./numFolds, "Gaussian - Variable Anisotropicity")

plotGammaDistPDFfromVector(meanPerturbDistanceVec_g_va, "Gaussian - Variable Anisotropicity")

(toptExpVal_g_va, toptStd_g_va) = calcPerfectClassStats(meanPerturbDistanceVec_g_va, numFolds)

plotAnnealResults(meanTrainErrorVec_g_va, meanValErrorVec_g_va, reportFrequency, "Gaussian - Variable Anisotropicity")



###################################################################################################################################################

# EXPONENTIAL

###################################################################################################################################################


outTuple_e_i = @time runSynapticAnnealingExperiment("Exponential - Isotropic", generate, saveData, loadData, savePlots, view, 3, maxRuns, 1,
                                                    "ExponentialVisit", "IsotropicAnisotropicity", "comparitive_simulated_annealing",
                                                    numFolds, matrixConfig, synapticAnnealing,
                                                    0.0, maxRuns, exponential_Isotropic_SynapticPerturbation, AnnealingState.updateState_fsa,
                                                    getDataClassErr, getDataClassErr,
                                                    initTemp, 1,
                                                    synMatIn, tanh,
                                                    dataSet, batchSize, reportFrequency)

putdata(outTuple_e_i, "outTuple_e_i")
# outTuple_e_i = getdata("outTuple_e_i")

(meanValErrorVec_e_i, meanTrainErrorVec_e_i, meanPerturbDistanceVec_e_i, minValErrorSynapseMatrix_e_i) = outTuple_e_i

plotCompleteRate(meanPerturbDistanceVec_e_i./numFolds, "Exponential - Isotropic")

plotGammaDistPDFfromVector(meanPerturbDistanceVec_e_i, "Exponential - Isotropic")

(toptExpVal_e_i, toptStd_e_i) = calcPerfectClassStats(meanPerturbDistanceVec_e_i, numFolds)


plotAnnealResults(meanTrainErrorVec_e_i, meanValErrorVec_e_i, reportFrequency, "Exponential - Isotropic")



###################################################################################################################################################


###################################################################################################################################################


outTuple_e_uo = @time runSynapticAnnealingExperiment("Exponential - Uniform Anisotropicity", generate, saveData, loadData, savePlots, view, 3, maxRuns, 1,
                                                     "ExponentialVisit", "UniformAnisotropicity", "comparitive_simulated_annealing",
                                                     (numFolds, matrixConfig, synapticAnnealing,
                                                      0.0, maxRuns,  exponential_UniformlyAnisotropic_SynapticPerturbation, AnnealingState.updateState_fsa,
                                                      getDataClassErr, getDataClassErr,
                                                      initTemp, 1,
                                                      synMatIn,tanh,
                                                      dataSet, batchSize, reportFrequency)

putdata(outTuple_e_uo, "outTuple_e_uo")
# outTuple_e_uo = getdata("outTuple_e_uo")

(meanValErrorVec_e_uo, meanTrainErrorVec_e_uo, meanPerturbDistanceVec_e_uo, minValErrorSynapseMatrix_e_uo) = outTuple_e_uo

plotCompleteRate(meanPerturbDistanceVec_e_uo./numFolds, "Exponential - Uniform Anisotropicity")

plotGammaDistPDFfromVector(meanPerturbDistanceVec_e_uo, "Exponential - Uniform Anisotropicity")

(toptExpVal_e_uo, toptStd_e_uo) = calcPerfectClassStats(meanPerturbDistanceVec_e_uo, numFolds)


plotAnnealResults(meanTrainErrorVec_e_uo, meanValErrorVec_e_uo, reportFrequency, "Exponential - Uniform Anisotropicity")



###################################################################################################################################################

###################################################################################################################################################


outTuple_e_va = @time runSynapticAnnealingExperiment("Exponential - Variable Anisotropicity", generate, saveData, loadData, savePlots, view, 3, maxRuns, 1,
                                                     "ExponentialVisit", "VariableAnisotropicity", "comparitive_simulated_annealing",
                                                     numFolds, matrixConfig, synapticAnnealing,
                                                     0.0, maxRuns,  exponential_VariablyAnisotropic_SynapticPerturbation, AnnealingState.updateState_fsa_anisotropicity,
                                                     getDataClassErr, getDataClassErr,
                                                     initTemp, 1,
                                                     synMatIn, tanh,
                                                     dataSet, batchSize, reportFrequency)

putdata(outTuple_e_va, "outTuple_e_va")
# outTuple_e_va = getdata("outTuple_e_va")

(meanValErrorVec_e_va, meanTrainErrorVec_e_va, meanPerturbDistanceVec_e_va, minValErrorSynapseMatrix_e_va) = outTuple_e_va

plotCompleteRate(meanPerturbDistanceVec_e_va./numFolds, "Exponential - Variable Anisotropicity")

plotGammaDistPDFfromVector(meanPerturbDistanceVec_e_va, "Exponential - Variable Anisotropicity")

(toptExpVal_e_va, toptStd_e_va) = calcPerfectClassStats(meanPerturbDistanceVec_e_va, numFolds)

plotAnnealResults(meanTrainErrorVec_e_va, meanValErrorVec_e_va, reportFrequency, "Exponential - Variable Anisotropicity")


###################################################################################################################################################



# UNIFORM


###################################################################################################################################################


outTuple_u_i = @time runSynapticAnnealingExperiment("Uniform - Isotropic", generate, saveData, loadData, savePlots, view, 3, maxRuns, 1,
                                                    "UniformVisit", "IsotropicAnisotropicity", "comparitive_simulated_annealing",
                                                    numFolds, matrixConfig, synapticAnnealing,
                                                    0.0, maxRuns, uniform_Isotropic_SynapticPerturbation, AnnealingState.updateState_fsa,
                                                    getDataClassErr, getDataClassErr,
                                                    initTemp, 1,
                                                    synMatIn, tanh,
                                                    dataSet, batchSize, reportFrequency)

putdata(outTuple_u_i, "outTuple_u_i")
# outTuple_u_i = getdata("outTuple_u_i")

(meanValErrorVec_u_i, meanTrainErrorVec_u_i, meanPerturbDistanceVec_u_i, minValErrorSynapseMatrix_u_i) = outTuple_u_i

plotCompleteRate(meanPerturbDistanceVec_u_i./numFolds, "Unifrom - Isotropic")

plotGammaDistPDFfromVector(meanPerturbDistanceVec_u_i, "Unifrom - Isotropic")

(toptExpVal_u_va, toptStd_u_va) = calcPerfectClassStats(meanPerturbDistanceVec_u_i, numFolds)


plotAnnealResults(meanTrainErrorVec_u_i, meanValErrorVec_u_i, reportFrequency, "Unifrom - Isotropic")



###################################################################################################################################################


###################################################################################################################################################


outTuple_u_ua = @time runSynapticAnnealingExperiment("Uniform - Uniform Anisotropicity", generate, saveData, loadData, savePlots, view, 3, maxRuns, 1,
                                                     "UniformVisit", "UniformAnisotropicity", "comparitive_simulated_annealing",
                                                     numFolds, matrixConfig, synapticAnnealing,
                                                     0.0, maxRuns,  uniform_UniformlyAnisotropic_SynapticPerturbation, AnnealingState.updateState_fsa,
                                                     getDataClassErr, getDataClassErr,
                                                     initTemp, 1,
                                                     synMatIn,tanh,
                                                     dataSet, batchSize, reportFrequency)

putdata(outTuple_u_ua, "outTuple_u_ua")
# outTuple_u_ua = getdata("outTuple_u_ua")

(meanValErrorVec_u_ua, meanTrainErrorVec_u_ua, meanPerturbDistanceVec_u_ua, minValErrorSynapseMatrix_u_ua) = outTuple_u_ua

plotCompleteRate(meanPerturbDistanceVec_u_ua./numFolds, "Uniform - Uniform Anisotropicity")

plotGammaDistPDFfromVector(meanPerturbDistanceVec_u_ua, "Uniform - Uniform Anisotropicity")

(toptExpVal_u_ua, toptStd_u_ua) = calcPerfectClassStats(meanPerturbDistanceVec_u_ua, numFolds)


plotAnnealResults(meanTrainErrorVec_u_ua, meanValErrorVec_u_ua, reportFrequency, "Uniform - Uniform Anisotropicity")



###################################################################################################################################################

###################################################################################################################################################


outTuple_u_va = @time runSynapticAnnealingExperiment("Uniform - VariableAnisotropicity", generate, saveData, loadData, savePlots, view, 3, maxRuns, 1,
                                                     "UniformVisit", "VariableAnisotropicity", "comparitive_simulated_annealing",
                                                     numFolds, matrixConfig, synapticAnnealing,
                                                     0.0, maxRuns,  uniform_VariablyAnisotropic_SynapticPerturbation, AnnealingState.updateState_fsa_anisotropicity,
                                                     getDataClassErr, getDataClassErr,
                                                     initTemp, 1,
                                                     synMatIn, tanh,
                                                     dataSet, batchSize, reportFrequency)

# putdata(outTuple_u_va, "outTuple_u_va")
outTuple_u_va = getdata("outTuple_u_va")

(meanValErrorVec_u_va, meanTrainErrorVec_u_va, meanPerturbDistanceVec_u_va, minValErrorSynapseMatrix_u_va) = outTuple_u_va

plotCompleteRate(meanPerturbDistanceVec_u_va./numFolds, "Uniform - Variable Anisotropicity")

plotGammaDistPDFfromVector(meanPerturbDistanceVec_u_va, "Uniform - Variable Anisotropicity")

(toptExpVal_u_va, toptStd_u_va) = calcPerfectClassStats(meanPerturbDistanceVec_u_va, numFolds

plotAnnealResults(meanTrainErrorVec_u_va, meanValErrorVec_u_va, reportFrequency, "Uniform - Variable Anisotropicity")

###################################################################################################################################################


# CAUCHY


###################################################################################################################################################


outTuple_c_i = @time runSynapticAnnealingExperiment("Cauchy - Isotropic", generate, saveData, loadData, savePlots, view, 3, maxRuns, 1,
                                                    "CauchyVisit", "IsotropicAnisotropicity", "comparitive_simulated_annealing",
                                                    numFolds, matrixConfig, synapticAnnealing,
                                                    0.0, maxRuns, cauchy_Isotropic_SynapticPerturbation, AnnealingState.updateState_csa,
                                                    getDataClassErr, getDataClassErr,
                                                    initTemp, 1,
                                                    synMatIn, tanh,
                                                    dataSet, batchSize, reportFrequency)

# putdata(outTuple_c_i, "outTuple_c_i")
outTuple_c_i = getdata("outTuple_c_i")

(meanValErrorVec_c_i, meanTrainErrorVec_c_i, meanPerturbDistanceVec_c_i, minValErrorSynapseMatrix_c_i) = outTuple_c_i

plotCompleteRate(meanPerturbDistanceVec_c_i./numFolds, "Cauchy - Isotropic")

plotGammaDistPDFfromVector(meanPerturbDistanceVec_c_i, "Cauchy - Isotropic")

(toptExpVal_c_i, toptStd_c_i) = calcPerfectClassStats(meanPerturbDistanceVec_c_i, numFolds)

plotAnnealResults(meanTrainErrorVec_c_i, meanValErrorVec_c_i, reportFrequency, "Cauchy - Isotropic")



###########################################################

###################################################################################################################################################


outTuple_c_ua = @time runSynapticAnnealingExperiment("Cauchy - Uniform Anisotropicity", generate, saveData, loadData, savePlots, view, 3, maxRuns, 1,
                                                     "CauchyVisit", "UniformAnisotropicity", "comparitive_simulated_annealing",
                                                     numFolds, matrixConfig, synapticAnnealing,
                                                     0.0, maxRuns,  cauchy_UniformlyAnisotropic_SynapticPerturbation, AnnealingState.updateState_fsa,
                                                     getDataClassErr, getDataClassErr,
                                                     initTemp, 1,
                                                     synMatIn,tanh,
                                                     dataSet, batchSize, reportFrequency)

# putdata(outTuple_c_ua, "outTuple_c_ua")
outTuple_c_ua = getdata("outTuple_c_ua")

(meanValErrorVec_c_ua, meanTrainErrorVec_c_ua, meanPerturbDistanceVec_c_ua, minValErrorSynapseMatrix_c_ua) = outTuple_c_ua

plotCompleteRate(meanPerturbDistanceVec_c_ua./numFolds, "Cauchy - Uniform Anisotropicity")

plotGammaDistPDFfromVector(meanPerturbDistanceVec_c_ua, "Cauchy - Uniform Anisotropicity")

(toptExpVal_c_ua, toptStd_c_ua) = calcPerfectClassStats(meanPerturbDistanceVec_c_ua, numFolds)

plotAnnealResults(meanTrainErrorVec_c_ua, meanValErrorVec_c_ua, reportFrequency, "Cauchy - Uniform Anisotropicity")



###################################################################################################################################################

###################################################################################################################################################


outTuple_c_va = @time runSynapticAnnealingExperiment("Cauchy - Variable Anisotropicity", generate, saveData, loadData, savePlots, view, 3, maxRuns, 1,
                                                     "CauchyVisit", "VariableAnisotropicity", "comparitive_simulated_annealing",
                                                     numFolds, matrixConfig, synapticAnnealing,
                                                     0.0, maxRuns,  cauchy_VariablyAnisotropic_SynapticPerturbation, AnnealingState.updateState_fsa_anisotropicity,
                                                     getDataClassErr, getDataClassErr,
                                                     initTemp, 1,
                                                     synMatIn, tanh,
                                                     dataSet, batchSize, reportFrequency)

# putdata(outTuple_c_va, "outTuple_c_va")
outTuple_c_va = getdata("outTuple_c_va")

(meanValErrorVec_c_va, meanTrainErrorVec_c_va, meanPerturbDistanceVec_c_va, minValErrorSynapseMatrix_c_va) = outTuple_c_va

plotCompleteRate(meanPerturbDistanceVec_c_va./numFolds, "Cauchy - Variable Anistropicity")

plotGammaDistPDFfromVector(meanPerturbDistanceVec_u_va, "Cauchy - Variable Anistropicity")

(toptExpVal_u_va, toptStd_u_va) = calcPerfectClassStats(meanPerturbDistanceVec_u_va, numFolds)


plotAnnealResults(meanTrainErrorVec_c_va, meanValErrorVec_c_va, reportFrequency, "Cauchy - Variable Anistropicity")

# Generate graphs of performance for each exp and save in folder
# Generate graphs of each exps completed-at-epoch  and save each in folder. Empircally observed cumulative distribution functions of simulation epoch required to achieve perfect classification
# Overlay all the graphs. Average
###################################################################################################################################################

function stepExpectationValue(v)

  sum(([1:length(v)])[[false,bool(diff(v))]] .* [0,diff(v)][[false,bool(diff(v))]])

end

function stepStd(v)
  sqrt(sum(([0,([1:length(v)])[[false,bool(diff(v))]]] .- stepExpectationValue(v)).^2)/length([0,diff(v)][[false,bool(diff(v))]]))

end

function stepVar(v)
  sum(([0,([1:length(v)])[[false,bool(diff(v))]]] .- stepExpectationValue(v)).^2)/length([0,diff(v)][[false,bool(diff(v))]])

end

function gaussianPDF(x,mu,std)
  (1/(std*2*pi))exp(-((x.-mu).^2)./(2*(std^2)))
end

function calcPerfectClassStats(v)
  toptExpVal = (stepExpectationValue((v./maximum(v))))
  toptStd = (stepStd(v./maximum(v)))
  return(toptExpVal, toptStd)
end

