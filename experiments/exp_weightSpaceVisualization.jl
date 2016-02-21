using PyPlot

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
@everywhere include("$(pwd())\\src\\"*"synapticAnnealing_fullRecord.jl")
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





function plotAnnealingJumps2d(w1, w2, colorChoice, labelText)

    currentPoint = [w1[1], w2[1]]

    scatter(currentPoint[1], currentPoint[2], color=colorChoice, marker="  \$\ start \$\  ", s=700)

    for step = 2:length(w1)-1

        p1 = currentPoint
        p2 = [w1[step], w2[step]]

        if outputTuple[7][step]==1
          style="-"
          transparency=1
        else
          style=":"
          transparency=0.5
        end

        if step == length(w1)-1
            plot([p1[1],p2[1]], [p1[2],p2[2]], color=colorChoice, ls=style, label=labelText, alpha=transparency)
        else
            plot([p1[1],p2[1]], [p1[2],p2[2]], color=colorChoice, ls=style, alpha=transparency)
        end

        if outputTuple[7][step]==1
          currentPoint = [w1[step], w2[step]]
        end

    end

    scatter(currentPoint[1], currentPoint[2], color=colorChoice, marker="  \$\ end \$\  ", s=500)
    legend(loc=1)
    grid("on")

    xlabel(" \$\ w_1 \$\ ")

    ylabel(" \$\ w_2 \$\ ")

    title(labelText*" Weight Space Traversal")

end


function plotAnnealingJumpsError3d(w1, w2, err, colorChoice, labelText)

    currentPoint = [w1[1], w2[1], err[1]]
    for step = 2:length(w1)-1

        p1 = currentPoint
        p2 = [w1[step], w2[step], err[step]]

        if outputTuple[7][step]==1
          style="-"
          transparency=1
        else
          style=":"
          transparency=0.25
        end

        if step == length(w1)-1
            plot3D([p1[1],p2[1]], [p1[2],p2[2]], [p1[3],p2[3]], color=colorChoice, ls=style, label=labelText, alpha=transparency)
        else
            plot3D([p1[1],p2[1]], [p1[2],p2[2]], [p1[3],p2[3]], color=colorChoice, ls=style, alpha=transparency)
        end

        if outputTuple[7][step]==1
          currentPoint = [w1[step], w2[step], err[step]]
        end
    end
    legend(loc=1)
    grid("on")


end


function plotAnnealingResults(meanTrainErrorVec, meanValErrorVec, reportFrequency, titleStr, dataSetName)
   	plotHandle = plot(reportFrequency.*(0:length(meanTrainErrorVec)-1), meanTrainErrorVec, label="Training Error", alpha=0.7, color="b")
    plotHandle = plot(reportFrequency.*(0:length(meanValErrorVec)-1), meanValErrorVec, label="Validation Error", alpha=0.7, color="g")
#     ylim(0,1)
    xlim(0,length(meanValErrorVec))

#     if(minimum(meanValErrorVec)<0 || maximum(meanValErrorVec)>1)
#       ylim(minimum(meanValErrorVec), maximum(meanValErrorVec))
#     end

    ax = gca()
    ylabel("MSE")
    legend(loc=1)
    grid("on")
    title(dataSetName*" - "*titleStr)


#     finalValErr = string(meanValErrorVec[end])[1:6]
#     scatter(length(meanValErrorVec)-100, finalValErr, marker="  \$\ "*finalValErr*" \$\  ", s=300 ,color= "g")
#     finalTrainErr = string(meanTrainErrorVec[end])[1:6]
#     scatter(length(meanTrainErrorVec)-100, finalTrainErr, marker="  \$\ "*finalTrainErr*" \$\  ", s=300, color="b")


    return(plotHandle)
end

function showWeightSpaceTraversalFigures(outputTuple, expTitleString, plotColor, windowSize)
    w1 = [w[2,1,1] for w in outputTuple[6]]
    w2 = [w[3,1,1] for w in outputTuple[6]]

    figure(1)
    plotAnnealingJumps2d(w1, w2, plotColor, expTitleString)

    # plotAnnealingJumpsError3d(w1, w2, outputTuple[5], "r", "GSA")

    figure(2)
    subplot(2,1,1)
    meanTrainErrorVec = outputTuple[3]
    meanValErrorVec = outputTuple[2]
    plotAnnealingResults(movingWindowAverage(meanTrainErrorVec, windowSize), movingWindowAverage(meanValErrorVec, windowSize), 1, "Error During Training", expTitleString)
    ylim(0,0.3)

    println([meanTrainErrorVec[end] meanValErrorVec[end]])

    subplot(2,1,2)
    ssw = [sum(w.^2) for w in outputTuple[6]]
    plot(movingWindowAverage(ssw, windowSize))
    title("Sum of Squared Weights During Training")
    xlabel("Training Epoch")
    ylabel("Sum of Squared Weights")
    grid()
    ylim(0,maximum(ssw))
end
###################################################################################################################################################

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
            outArray[xIndex, yIndex] = f(x,y)
        end
    end

    return(outArray)

end

function build2dGenericFunctionArray(f,xRange,yRange)

    outArray = zeros((length(xRange), length(yRange)))
    xIndex = 0
    for x in xRange
        xIndex += 1
        yIndex = 0
        for y in yRange
            yIndex += 1
            outArray[xIndex, yIndex] = f(x,y)
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
xRange=-1:0.01:1
yRange=-1:0.01:1

complicatedInteractionFuncArray = build2dFunctionArray(complicatedInteraction,xRange,yRange)

functionSample = sampleFunction(complicatedInteraction, 100, xRange, yRange)

compInteractionDataset = ExperimentDataset.Dataset(functionSample./maximum(abs(functionSample)), [1:2], [3], "Complicated Interaction")

dataSet = compInteractionDataset


###################################################################################################################################################





###################################################################################################################################################

reportFrequency = 100
maxRuns = 5000
actFun = tanh
initTemperature = 10
initLearnRate = 0.001

matrixConfig = [length(dataSet.inputCols), 2, length(dataSet.outputCols)]

synMatIn = null
batchSize = size(dataSet.data)[1]
###################################################################################################################################################

synMatConfigVec = matrixConfig

###################################################################################################################################################

valData  = ExperimentDataset.Dataset(dataSet.data[1:90,:], dataSet.inputCols, dataSet.outputCols, dataSet.name)
trainData = ExperimentDataset.Dataset(dataSet.data[91:end,:], dataSet.inputCols, dataSet.outputCols, dataSet.name)


function newNet(synMatConfigVec)


  # Initialize a new synapse matrix.
  netIn = init_network(synMatConfigVec)
  netIn.learning_rate = 0
  netIn.propagation_function = actFun
  return(netIn)

end

###################################################################################################################################################

###################################################################################################################################################

###################################################################################################################################################

outputTuple_g = synapticAnnealing_fullRecord(0.0, maxRuns,
                             gaussian_Isotropic_SynapticPerturbation,
                             AnnealingState.updateState_fsa,
                             getDataRegErr, getDataRegErr,
                             initTemperature, 0.001,
                             newNet(synMatConfigVec), actFun,
                             trainData, valData,
                             batchSize, reportFrequency)


showWeightSpaceTraversalFigures(outputTuple_g,"Gaussian", "c", 21)

###################################################################################################################################################

@everywhere include("$(pwd())\\src\\"*"annealingTraversalFunctions.jl")
outputTuple_c = synapticAnnealing_fullRecord(0.0, maxRuns,
                             cauchy_Isotropic_SynapticPerturbation,
                             AnnealingState.updateState_fsa,
                             getDataRegErr, getDataRegErr,
                             initTemperature, initLearnRate,
                             newNet(synMatConfigVec), actFun,
                             trainData, valData,
                             batchSize, reportFrequency)


showWeightSpaceTraversalFigures(outputTuple_c,"Cauchy", "b", 21)


###################################################################################################################################################

@everywhere include("$(pwd())\\src\\"*"annealingTraversalFunctions.jl")
outputTuple_u = synapticAnnealing_fullRecord(0.0, maxRuns,
                             uniform_Isotropic_SynapticPerturbation,
                             AnnealingState.updateState_fsa,
                             getDataRegErr, getDataRegErr,
                             initTemperature, initLearnRate,
                             newNet(synMatConfigVec), actFun,
                             trainData, valData,
                             batchSize, reportFrequency)


showWeightSpaceTraversalFigures(outputTuple_u,"Uniform", "purple", 21)


###################################################################################################################################################

outputTuple_gsa_i = synapticAnnealing_fullRecord(0.0, maxRuns,
                             gsa_Isotropic_SynapticPerturbation,
                             AnnealingState.updateState_fsa,
                             getDataRegErr, getDataRegErr,
                             initTemperature, initLearnRate,
                             newNet(synMatConfigVec), actFun,
                             trainData, valData,
                             batchSize, reportFrequency)


showWeightSpaceTraversalFigures(outputTuple_gsa_i,"GSA - Isotropic", "r", 21)

###################################################################################################################################################

@everywhere include("$(pwd())\\src\\"*"annealingTraversalFunctions.jl")

outputTuple_gsa_wa = synapticAnnealing_fullRecord(0.0, maxRuns,
                             gsa_WeightAnisotropic_SynapticPerturbation,
                             AnnealingState.updateState_fsa,
                             getDataRegErr, getDataRegErr,
                             initTemperature, initLearnRate,
                             newNet(synMatConfigVec), actFun,
                             trainData, valData,
                             batchSize, reportFrequency)


showWeightSpaceTraversalFigures(outputTuple_gsa_wa, "GSA - Weight Anisotropic", "g", 21)

##############################################################
function plotTanhDerivative(xRange,yRange)

    set_cmap("gray")
    function tanhDerivative(x,y)
        xVal = 1 .- (tanh(x).^2)
        yVal = 1 .- (tanh(y).^2)
        return(xVal*yVal)
    end

    imshow(build2dFunctionArray(tanhDerivative,xRange,yRange),extent=[minimum(xRange),maximum(xRange),minimum(yRange),maximum(yRange)])

end

gradientDisp = 100
gradientRes = 0.1
gradientRange = -gradientDisp:gradientRes:gradientDisp
plotTanhDerivative(gradientRange, gradientRange)

