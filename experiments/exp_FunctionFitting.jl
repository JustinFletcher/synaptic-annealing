using PyPlot

rmprocs(workers())
addprocs(5)
# @everywhere using Devectorize

@everywhere using BackpropNeuralNet
@everywhere cd("\\mscs-thesis")

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

# Include cross val annealing libraries.
@everywhere include("$(pwd())\\src\\"*"buildFolds.jl")
@everywhere include("$(pwd())\\src\\"*"nFoldCrossValidateSynapticAnnealing.jl")

# Include cross val annealing libraries.
@everywhere include("$(pwd())\\src\\"*"backpropTraining.jl")
@everywhere include("$(pwd())\\src\\"*"nFoldCrossValidateBackprop.jl")
@everywhere include("$(pwd())\\src\\"*"gsa.jl")

ion()


###################################################################################################################################################


function complicatedInteraction(a,b,c,d,f,g,h,i,j,k,l,x,y)
  return(a.*(b+((e.^(x+c)).*sin(d.*((f.*x)-g).^h)))*(e.^((i).*(y+j))).*(sin((k.*y)+l)))
end


function complicatedInteraction(x,y)
  return(1.9.*(1.35+((e.^(x+1)).*sin(13.*((0.5.*x)-0.1).^2)))*(e.^((-0.5).*(y+1))).*(sin((3.5.*y)+3.5)))
end

function build2dFunctionArray(f,xRange,yRange)

    outArray = zeros((length(xRange), length(yRange)))
    xIndex = 0
    for x in xRange
        xIndex += 1
        yIndex = 0
        for y in yRange
            yIndex += 1
            outArray[xIndex, yIndex] = complicatedInteraction(x,y)
        end
    end

    return(outArray)

end

function sampleFunction(f, nSamples, xSampleRange, ySampleRange)

    outMat = zeros(nSamples, 3)

    for t in 1:nSamples
        xSample = rand(xSampleRange)
        ySample = rand(ySampleRange)
        out = f(xSample, ySample)
        outMat[t, :] = [xSample, ySample, out]
    end

    return(outMat)

end

function writeSampleToFile(sample, filename)
    csvfile = open("$(pwd())\\data\\"*filename*".csv","w")
    for i = 1:size(functionSample)[1]

        y1,y2,y3 = functionSample[i, :]
        ysavetuple = y1, y2, y3
        write(csvfile, join(ysavetuple,","), "\n")
    end

    close(csvfile)
end


function viewRegressionResults(inSurf, outSurf, xRange, yRange)

    surf(xRange, yRange, outSurf, alpha=.5)

    surf(xRange, yRange, (inSurf./maximum(abs(inSurf))), alpha=.5, color="red")

end

function constructModelPrediction(annealingOutput, xRange, yRange)

    outArray = zeros((length(xRange), length(yRange)))
    xIndex = 0
    for x in xRange
        xIndex += 1
        yIndex = 0
        for y in yRange
            yIndex += 1
            outArray[xIndex, yIndex] = net_eval(annealingOutput[6][2], [x, y])[1]
        end
    end

  return(outArray)

end

function viewRegressionResultImages(inSurf, outSurf, xRange, yRange)

    colorLimits = (-1,1)
    figure()
    normInSurf = inSurf./maximum(abs(inSurf))
    subplot(1,3,1)
    imshow(normInSurf, clim=colorLimits)
#     colorbar()
    subplot(1,3,2)
    imshow(outSurf, clim=colorLimits)
#     colorbar()
    subplot(1,3,3)
    imshow(normInSurf.-outSurf, clim=colorLimits)
#     colorbar()


end

function constructModelPrediction(annealingOutput, xRange, yRange)

    outArray = zeros((length(xRange), length(yRange)))
    xIndex = 0
    for x in xRange
        xIndex += 1
        yIndex = 0
        for y in yRange
            yIndex += 1
            outArray[xIndex, yIndex] = net_eval(annealingOutput[6][2], [x, y])[1]
        end
    end

  return(outArray)

end

###################################################################################################################################################

xRange=-1:0.01:1
yRange=-1:0.01:1

complicatedInteractionFuncArray = build2dFunctionArray(complicatedInteraction,xRange,yRange)

mesh(complicatedInteractionFuncArray./maximum(abs(complicatedInteractionFuncArray)))

functionSample = sampleFunction(complicatedInteraction, 100, xRange, yRange)

# writeSampleToFile(functionSample, filename)



###################################################################################################################################################

compInteractionDataset = ExperimentDataset.Dataset(functionSample./maximum(abs(functionSample)), [1:2], [3], "Complicated Interaction")

dataSet = compInteractionDataset

###################################################################################################################################################


numFolds = 5

maxRuns = 10000

initTemp = 1

numHiddenLayers = 1

matrixConfig = [length(dataSet.inputCols), 10, length(dataSet.outputCols)]

synMatIn = null

batchSize = size(dataSet.data)[1]

reportFrequency = 1000


###################################################################################################################################################

initLearnRate = 0.001

actFun = tanh

###################################################################################################################################################


annealingOutput_gaussian = @time nFoldCrossValidateSynapticAnnealingPar(numFolds, matrixConfig,  synapticAnnealing,
                                                               0.0, maxRuns,
                                                               gaussian_Isotropic_SynapticPerturbation,
                                                               AnnealingState.updateState_csa,
                                                               getDataRegErr, getDataRegErr,
                                                               initTemp, initLearnRate,
                                                               synMatIn, actFun,
                                                               dataSet, batchSize, reportFrequency)


# scatter3D(functionSample[:,1],functionSample[:,2],functionSample[:,3]./maximum(abs(functionSample[:,3])), color="red")
# viewRegressionResults(complicatedInteractionFuncArray, constructModelPrediction(imshowannealingOutput_gaussianannealingOutput_gaussian, xRange, yRange), xRange, yRange)
viewRegressionResultImages(complicatedInteractionFuncArray, constructModelPrediction(annealingOutput_gaussian, xRange, yRange), xRange, yRange)

###################################################################################################################################################


annealingOutput_cauchy = @time nFoldCrossValidateSynapticAnnealingPar(numFolds, matrixConfig,  synapticAnnealing,
                                                               0.0, maxRuns,
                                                               cauchy_Isotropic_SynapticPerturbation,
                                                               AnnealingState.updateState_fsa,
                                                               getDataRegErr, getDataRegErr,
                                                               initTemp, initLearnRate,
                                                               synMatIn, actFun,
                                                               dataSet, batchSize, reportFrequency)



# scatter3D(functionSample[:,1],functionSample[:,2],functionSample[:,3]./maximum(abs(functionSample[:,3])), color="red")
# viewRegressionResults(complicatedInteractionFuncArray, constructModelPrediction(annealingOutput_cauchy, xRange, yRange), xRange, yRange)
viewRegressionResultImages(complicatedInteractionFuncArray, constructModelPrediction(annealingOutput_cauchy, xRange, yRange), xRange, yRange)


###################################################################################################################################################


annealingOutput_gsa = @time nFoldCrossValidateSynapticAnnealingPar(numFolds, matrixConfig,  synapticAnnealing,
                                                               0.0, maxRuns,
                                                               gsa_Isotropic_SynapticPerturbation,
                                                               AnnealingState.updateState_fsa,
                                                               getDataRegErr, getDataRegErr,
                                                               0.001, initLearnRate,
                                                               synMatIn, actFun,
                                                               dataSet, batchSize, reportFrequency)


# scatter3D(functionSample[:,1],functionSample[:,2],functionSample[:,3]./maximum(abs(functionSample[:,3])), color="red")
viewRegressionResults(complicatedInteractionFuncArray, constructModelPrediction(annealingOutput_gsa, xRange, yRange), xRange, yRange)
viewRegressionResultImages(complicatedInteractionFuncArray, constructModelPrediction(annealingOutput_gsa, xRange, yRange), xRange, yRange)

###################################################################################################################################################


regAnnealingOutputList = Any[[annealingOutput_gaussian, "Gaussian"],
                          [annealingOutput_cauchy, "Cauchy"],
                          [annealingOutput_gsa, "GSA"]]

###################################################################################################################################################

xmax = 10000
ymax = 1
titleStr = "Validation Set Mean Squared Error"
for annealingOutputVec in regAnnealingOutputList

  annealingOutput = annealingOutputVec[1]
  dataSetName = annealingOutputVec[2]

  plotCentralMovingAverageValError(annealingOutput[1], annealingOutput[2], 3, reportFrequency, titleStr, dataSetName, xmax, ymax, "blue")

end
plotCentralMovingAverageValError(annealingOutput_gaussian[1], annealingOutput_gaussian[2], 3,
                                 reportFrequency, "Gaussian", "Gaussian",
                                 10000, 1, "blue")

plot(annealingOutput_gaussian[1], label="Gaussian")
plot(annealingOutput_cauchy[1], label="Cauchy")
plot(annealingOutput_gsa[1], label="GSA")'
legend(loc=1)

