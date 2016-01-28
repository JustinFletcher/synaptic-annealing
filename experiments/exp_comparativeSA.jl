# Pkg.init()
# Pkg.add("PyCall")
# Pkg.build("PyCall")
# Pkg.add("PyPlot")
# Pkg.build("PyPlot")
# # Pkg.add("StatsBase")
# Pkg.add("BackpropNeuralNet")
# Pkg.add("MNIST")


using PyPlot

rmprocs(workers())
addprocs(25)
# @everywhere using Devectorize

@everywhere using BackpropNeuralNet
@everywhere cd("\\fletcher-thesis")

pwd()

@everywhere include("$(pwd())\\src\\"*"ExperimentDataset.jl")
@everywhere include("$(pwd())\\src\\"*"AnnealingState.jl")

# Include utility libraries.|
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
@everywhere include("$(pwd())\\src\\"*"gsa.jl")

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


irisDatapath = "$(pwd())\\data\\iris.dat"
dataInputDimensions = [1:4]
dataOutputDimensions = [5]

irisDataset = ExperimentDataset.Dataset(irisDatapath, dataInputDimensions, dataOutputDimensions, "Iris")


###################################################################################################################################################



wineDatapath = "$(pwd())\\data\\wine.dat"
dataInputDimensions = [1:13]
dataOutputDimensions = [14]

wineDataset = ExperimentDataset.Dataset(wineDatapath, dataInputDimensions, dataOutputDimensions, "Wine")


###################################################################################################################################################

cancerDatapath = "$(pwd())\\data\\wisconsonBreastCancerData.dat"
dataInputDimensions = [1:30]
dataOutputDimensions = [31]

cancerDataset = ExperimentDataset.Dataset(cancerDatapath, dataInputDimensions, dataOutputDimensions, "Cancer")



# ###################################################################################################################################################
# using MNIST

# # Function to orthogonalize MNIST. Credit: github.com/yarlett
# function digits_to_indicators(digits)
# 	digit_indicators = zeros(Float64, (10, length(digits)))
# 	for j = 1:length(digits)
# 		digit_indicators[int(digits[j])+1, j] = 1.0
# 	end
# 	digit_indicators
# end

# # Load MNIST training and testing data.
# mnistTrainInput, mnistTrainClasses = traindata()
# mnistTrainInput ./= 255.0
# mnistTrainClasses = digits_to_indicators(mnistTrainClasses)

# # XTE, YTE = testdata()
# # XTE ./= 255.0
# # YTE = digits_to_indicators(YTE)

# # Make the classes antisemetric for consistency.
# mnistTrainClassesAntisymmetric = (mnistTrainClasses)

# # Transpose the MNIST training data for consistency.
# mnistTrainInput = transpose(mnistTrainInput)
# mnistTrainClassesAntisymmetric = transpose(mnistTrainClassesAntisymmetric)

# mnistTrainData = [mnistTrainInput mnistTrainClassesAntisymmetric]

# dataInputDimensions = [1:size(mnistTrainInput)[2]]
# dataOutputDimensions = size(mnistTrainInput)[2]+1

# mnistDataset = ExperimentDataset.Dataset(mnistTrainData[1:50,:], dataInputDimensions, dataOutputDimensions, "MNIST")

# mnistDataset.data[1,:]
# ###################################################################################################################################################

# lcvfDatapath = "$(pwd())\\data\\lcvfData.csv"
# dataInputDimensions = [1:194]
# dataOutputDimensions = [195]

# lcvfDataset = ExperimentDataset.Dataset(lcvfDatapath, dataInputDimensions, dataOutputDimensions, "LCVF")

# ###################################################################################################################################################


###################################################################################################################################################

dataSet = irisDataset

###################################################################################################################################################

#dataSet = lcvfDataset

###################################################################################################################################################

dataSet = wineDataset

###################################################################################################################################################

#dataSet = mnistDataset

###################################################################################################################################################

dataSetList = Any[wineDataset, irisDataset, cancerDataset]
#dataSetList = Any[mnistDataset]
#dataSetList = Any[lcvfDataset]
###################################################################################################################################################


numFolds = 25

maxRuns = 1000000

initTemp = 100

numHiddenLayers = 1

matrixConfig = [length(dataSet.inputCols), repmat([length(dataSet.inputCols)], numHiddenLayers), length(dataSet.outputCols)]

matrixConfig = [length(dataSet.inputCols),  length(dataSet.outputCols)]

synMatIn = null

batchSize = 150

batchSize = size(dataSet.data)[1]

reportFrequency = 10000

generate = true
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



# ###################################################################################################################################################

outTuple_g_i  = @time runSynapticAnnealingExperiment("Gaussian - Isotropic", generate, saveData, loadData, savePlots, view, 3, maxRuns, 1,
                                                     "GaussianVisit", "IsotropicAnisotropicity", "comparitive_simulated_annealing",
                                                     numFolds, matrixConfig, synapticAnnealing,
                                                     0.0, maxRuns, gaussian_Isotropic_SynapticPerturbation, AnnealingState.updateState_fsa,
                                                     getDataClassErr, getDataClassErr,
                                                     initTemp, 1,
                                                     synMatIn, tanh,
                                                     dataSet, batchSize, reportFrequency)


###################################################################################################################################################



outTuple_g_ua = @time runSynapticAnnealingExperiment("Gaussian - Uniform Anisotropicity", generate, saveData, loadData, savePlots, view, 3, maxRuns, 1,
                                                     "GaussianVisit", "UniformAnisotropicity", "comparitive_simulated_annealing",
                                                     numFolds, matrixConfig, synapticAnnealing,
                                                     0.0, maxRuns, gaussian_uniformAnisotropic_SynapticPerturbation, AnnealingState.updateState_csa,
                                                     getDataClassErr, getDataClassErr,
                                                     initTemp, 1,
                                                     synMatIn, tanh,
                                                     dataSetList, batchSize, reportFrequency)





###################################################################################################################################################

outTuple_g_na  = @time runSynapticAnnealingExperiment("Gaussian - Normative Anisotropicity", generate, saveData, loadData, savePlots, view, 3, maxRuns, 1,
                                                      "GaussianVisit", "NormativeAnisotropicity", "comparitive_simulated_annealing",
                                                      numFolds, matrixConfig, synapticAnnealing,
                                                      0.0, maxRuns, gaussian_NormativeAnisotropic_SynapticPerturbation, AnnealingState.updateState_csa,
                                                      getDataClassErr, getDataClassErr,
                                                      initTemp, 1,
                                                      synMatIn, tanh,
                                                      dataSet, batchSize, reportFrequency)


###################################################################################################################################################

# EXPONENTIAL

###################################################################################################################################################


outTuple_e_i = @time runSynapticAnnealingExperiment("Exponential - Isotropic", generate, saveData, loadData, savePlots, view, 3, maxRuns, 1,
                                                    "ExponentialVisit", "IsotropicAnisotropicity", "comparitive_simulated_annealing",
                                                    numFolds, matrixConfig, synapticAnnealing,
                                                    0.0, maxRuns, exponential_Isotropic_SynapticPerturbation, AnnealingState.updateState_csa,
                                                    getDataClassErr, getDataClassErr,
                                                    initTemp, 1,
                                                    synMatIn, tanh,
                                                    dataSet, batchSize, reportFrequency)


###################################################################################################################################################


###################################################################################################################################################


outTuple_e_uo = @time runSynapticAnnealingExperiment("Exponential - Uniform Anisotropicity", generate, saveData, loadData, savePlots, view, 3, maxRuns, 1,
                                                     "ExponentialVisit", "UniformAnisotropicity", "comparitive_simulated_annealing",
                                                     numFolds, matrixConfig, synapticAnnealing,
                                                      0.0, maxRuns,  exponential_UniformlyAnisotropic_SynapticPerturbation, AnnealingState.updateState_csa,
                                                      getDataClassErr, getDataClassErr,
                                                      initTemp, 1,
                                                      synMatIn,tanh,
                                                      dataSet, batchSize, reportFrequency)


###################################################################################################################################################

###################################################################################################################################################


outTuple_e_da = @time runSynapticAnnealingExperiment("Exponential - Normative Anisotropicity", generate, saveData, loadData, savePlots, view, 3, maxRuns, 1,
                                                     "ExponentialVisit", "NormativeAnisotropicity", "comparitive_simulated_annealing",
                                                     numFolds, matrixConfig, synapticAnnealing,
                                                     0.0, maxRuns,  exponential_NormativeAnisotropic_SynapticPerturbation, AnnealingState.updateState_csa,
                                                     getDataClassErr, getDataClassErr,
                                                     initTemp, 1,
                                                     synMatIn, tanh,
                                                     dataSet, batchSize, reportFrequency)


###################################################################################################################################################



# UNIFORM


###################################################################################################################################################


outTuple_u_i = @time runSynapticAnnealingExperiment("Uniform - Isotropic", generate, saveData, loadData, savePlots, view, 3, maxRuns, 1,
                                                    "UniformVisit", "IsotropicAnisotropicity", "comparitive_simulated_annealing",
                                                    numFolds, matrixConfig, synapticAnnealing,
                                                    0.0, maxRuns, uniform_Isotropic_SynapticPerturbation, AnnealingState.updateState_csa,
                                                    getDataClassErr, getDataClassErr,
                                                    initTemp, 1,
                                                    synMatIn, tanh,
                                                    dataSet, batchSize, reportFrequency)


###################################################################################################################################################


###################################################################################################################################################



outTuple_u_ua = @time runSynapticAnnealingExperiment("Uniform - Uniform Anisotropicity", generate, saveData, loadData, savePlots, view, 3, maxRuns, 1,
                                                     "UniformVisit", "UniformAnisotropicity", "comparitive_simulated_annealing",
                                                     numFolds, matrixConfig, synapticAnnealing,
                                                     0.0, maxRuns,  uniform_UniformlyAnisotropic_SynapticPerturbation, AnnealingState.updateState_csa,
                                                     getDataClassErr, getDataClassErr,
                                                     initTemp, 1,
                                                     synMatIn,tanh,
                                                     dataSet, batchSize, reportFrequency)


###################################################################################################################################################

###################################################################################################################################################


outTuple_u_da = @time runSynapticAnnealingExperiment("Uniform - Normative Anisotropicity", generate, saveData, loadData, savePlots, view, 3, maxRuns, 1,
                                                     "UniformVisit", "NormativeAnisotropicity", "comparitive_simulated_annealing",
                                                     numFolds, matrixConfig, synapticAnnealing,
                                                     0.0, maxRuns, uniform_NormativeAnisotropic_SynapticPerturbation, AnnealingState.updateState_csa,
                                                     getDataClassErr, getDataClassErr,
                                                     initTemp, 1,
                                                     synMatIn, tanh,

###################################################################################################################################################


# CAUCHY


###################################################################################################################################################


outTuple_c_i = @time runSynapticAnnealingExperiment("Cauchy - Isotropic", generate, saveData, loadData, savePlots, view, 3, maxRuns, 1,
                                                    "CauchyVisit", "IsotropicAnisotropicity", "comparitive_simulated_annealing",
                                                    numFolds, matrixConfig, synapticAnnealing,
                                                    0.0, maxRuns, cauchy_Isotropic_SynapticPerturbation, AnnealingState.updateState_fsa,
                                                    getDataClassErr, getDataClassErr,
                                                    initTemp, 0.1,
                                                    synMatIn, tanh,
                                                    dataSet, batchSize, reportFrequency)


###################################################################################################################################################
outTuple_c_i_ra = @time runSynapticAnnealingExperiment("Cauchy - Isotropic", generate, saveData, loadData, savePlots, view, 3, maxRuns, 1,
                                                    "CauchyVisit", "IsotropicAnisotropicity_ra", "comparitive_simulated_annealing",
                                                    numFolds, matrixConfig, synapticAnnealing,
                                                    0.0, maxRuns,cauchy_Isotropic_SynapticPerturbation, AnnealingState.updateState_fsa_ra,
                                                    getDataClassErr, getDataClassErr,
                                                    initTemp, 0.1,
                                                    synMatIn, tanh,
                                                    dataSet, batchSize, reportFrequency)


###################################################################################################################################################


# outTuple_c_ua = @time runSynapticAnnealingExperiment("Cauchy - Uniform Anisotropicity", generate, saveData, loadData, savePlots, view, 3, maxRuns, 1,
#                                                      "CauchyVisit", "UniformAnisotropicity", "comparitive_simulated_annealing",
#                                                      numFolds, matrixConfig, synapticAnnealing,
#                                                      0.0, maxRuns,  cauchy_UniformlyAnisotropic_SynapticPerturbation, AnnealingState.updateState_fsa,
#                                                      getDataClassErr, getDataClassErr,
#                                                      initTemp, 0.1,
#                                                      synMatIn,tanh,
#                                                      dataSet, batchSize, reportFrequency)


###################################################################################################################################################


outTuple_c_na = @time runSynapticAnnealingExperiment("Cauchy - Normative Anisotropicity", generate, saveData, loadData, savePlots, view, 3, maxRuns, 1,
                                                     "CauchyVisit", "NormativeAnisotropicity_fsa_t1000", "comparitive_simulated_annealing",
                                                     numFolds, matrixConfig, synapticAnnealing,
                                                     0.0, maxRuns,  cauchy_NormativeAnisotropic_SynapticPerturbation, AnnealingState.updateState_fsa,
                                                     getDataClassErr, getDataClassErr,
                                                     initTemp, 0.1,
                                                     synMatIn, tanh,
                                                     dataSet, batchSize, reportFrequency)


###################################################################################################################################################



outTuple_c_neurala = @time runSynapticAnnealingExperiment("Cauchy - Neural Anisotropicity", generate, saveData, loadData, savePlots, view, 3, maxRuns, 1,
                                                     "CauchyVisit", "NeuralAnisotropicity", "comparitive_simulated_annealing",
                                                     numFolds, matrixConfig, synapticAnnealing,
                                                     0.0, maxRuns,  cauchy_NeuralAnisotropic_SynapticPerturbation, AnnealingState.updateState_fsa,
                                                     getDataClassErr, getDataClassErr,
                                                     initTemp, 0.1,
                                                     synMatIn, tanh,
                                                     dataSet, batchSize, reportFrequency)


###################################################################################################################################################



outTuple_gsa_i = @time runSynapticAnnealingExperiment("GSA - Isotropic", generate, saveData, loadData, savePlots, view, 3, maxRuns, 1,
                                                     "GSAVisit", "IsotropicAnisotropicity_alpha0p1_qv2p5_t1_d1", "comparitive_simulated_annealing",
                                                     numFolds, matrixConfig, synapticAnnealing,
                                                     0.0, maxRuns,  gsa_Isotropic_SynapticPerturbation, AnnealingState.updateState_fsa,
                                                     getDataClassErr, getDataClassErr,
                                                     initTemp, 0.1,
                                                     synMatIn, tanh,
                                                     dataSet, batchSize, reportFrequency)

