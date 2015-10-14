









###################################################################################################################################################################################################
# ---- Experiment Development Area ----

# Pull the physics training data.
tauThreeMuTrainData = readdlm("C:\\afit\\CSCE823\\project\\tauThreeMuTrainData.dat", ',' , Any)
# Clip the last two columns, and the first row.
tauThreeMuTrainData = tauThreeMuTrainData[2:end, 1:end-2]
# Clip the Second from the last column, and the first column
tauThreeMuTrainData = [tauThreeMuTrainData[:, 2:end-2] tauThreeMuTrainData[:, end]]

tauThreeMuTrainData = orthogonalizeDataClasses(tauThreeMuTrainData, [47])
tauThreeMuTrainData = normalizeData(tauThreeMuTrainData)
tauThreeMuTrainData = shuffleData(tauThreeMuTrainData)

subsetSize = 600
tauThreeMuTrainData_subsetTrain = tauThreeMuTrainData[1:subsetSize, :]
tauThreeMuTrainData_subsetTest = tauThreeMuTrainData[(subsetSize+1):end, :]

meanValErrorVec, meanTrainErrorVec, minValErrorSynapseMatrix = @time nFoldCrossValidateSynapticAnnealing(4, 25000, [47,25,2], annealSynapses_allWeight, getDataRegErr, getDataClassErr, linearTemperatureSchedule_eightTenths, sigmoidDecay_Tuned, tanh, tauThreeMuTrainData_subsetTrain, [1:46], [47:48])

putdata(meanTrainErrorVec, "meanTrainErrorVec_tauThreeMuTrain_epochs2000_samples6000")
putdata(meanValErrorVec, "meanValErrorVec_tauThreeMuTrain_epochs2000_samples6000")
putdata(minValErrorSynapseMatrix, "synapseMat_tauThreeMuTrain_epochs2000_samples6000")

meanTrainErrorVec = getdata("meanTrainErrorVec_tauThreeMuTrain_epochs2000_samples6000")
meanValErrorVec = getdata("meanValErrorVec_tauThreeMuTrain_epochs2000_samples6000")
minValErrorSynapseMatrix = getdata("synapseMat_tauThreeMuTrain_epochs2000_samples6000")

finalMeanValError = meanValErrorVec[end]

plotAnnealResults( meanTrainErrorVec, meanValErrorVec, "Training and Validation Classification Error of a Synaptic\n Annealing Neural Network using a Linear Temperature\n Schedule, Decaying Learn Rate After Annealing")



# Pull the physics training data.
tauThreeMuTestData = readdlm("C:\\afit\\CSCE823\\project\\tauThreeMuTestData.csv", ',' , Any)

tauThreeMuTestData = tauThreeMuTestData[2:end, :]

tauThreeMuTestData[:, 2:end] = normalizeData(tauThreeMuTestData[:, 2:end])
tauThreeMuTestDataIdVec = tauThreeMuTestData[:, 1]
tauThreeMuTestData = tauThreeMuTestData[:, 2:end]
tauThreeMuTrainData_predictions = getDataPreditions(minValErrorSynapseMatrix[:,:,:,1], tanh, float64(tauThreeMuTestData), [1:46],  [1:2])

tauThreeMuTrainData_predVec = (tauThreeMuTrainData_predictions[:,2]+1)./2

tauThreeMuTestOutputSet = [["id",int32(tauThreeMuTestDataIdVec)] ["prediction",tauThreeMuTrainData_predVec]]

outfile = open("C:\\afit\\CSCE823\\project\\testClassOutput_epochs2000_samples6000.csv", "w")
writecsv(outfile, tauThreeMuTestOutputSet)
close(outfile)
###################################################################################################################################################################################################




# ---- Resource Analysis ----

sampleRes = 10
maxEpochs = 100
epochComplexityVec = zeros(length(1:sampleRes:maxEpochs),2)

iter = 0
for epochTrial in 1:sampleRes:maxEpochs
    iter+=1
    tic()
    nFoldCrossValidateSynapticAnnealing(5, int(epochTrial), [5,10,3], annealSynapses_allWeight, getDataRegErr, getDataClassErr, linearTemperatureSchedule_eightTenths, function(x) return(1) end, tanh, irisDataClassed, [1:4], [5:7])

    time=toq()
    println([int(epochTrial), time[1]])
    println(iter)
    epochComplexityVec[iter,:] = [int(epochTrial) time[1]/(5*int(epochTrial))]
end

plot(epochComplexityVec[:,1],epochComplexityVec[:,2])




sampleRes = 10
maxSamples = 100
samplesComplexityVec = zeros(length(10:sampleRes:maxSamples+10),2)
irisDataClassed[1:10,:]
iter = 0
for numSamples in 10:sampleRes:maxSamples+10
    iter+=1
    dataSubset = irisDataClassed[(1:(numSamples)), :]

    tic()
    nFoldCrossValidateSynapticAnnealing(5, 100, [5,10,3], annealSynapses_allWeight, getDataRegErr, getDataClassErr, linearTemperatureSchedule_eightTenths, function(x) return(1) end, tanh, dataSubset, [1:4], [5:7])
    time=toq()

    samplesComplexityVec[iter, :] = [numSamples, time[1]]
end
samplesComplexityVec

plot(samplesComplexityVec[:,1],samplesComplexityVec[:,2]/(5*100))
title("Variation in CPU Time per Epoch with Increasing Sample Size")
ylabel("Average CPU Seconds Per Epoch")
xlabel("Number of Samples in the Dataset")




sampleRes = 1
maxNeurons = 75
neuronsComplexityVec = zeros(length(1:sampleRes:maxNeurons),2)
iter = 0
for numNeurons in 1:sampleRes:maxNeurons
    iter+=1

    tic()
    nFoldCrossValidateSynapticAnnealing(5, 10, [5,int(numNeurons),3], annealSynapses_allWeight, getDataRegErr, getDataClassErr, linearTemperatureSchedule_eightTenths, function(x) return(1) end, tanh, irisDataClassed, [1:4], [5:7])
    time=toq()

    neuronsComplexityVec[iter, :] = [numNeurons, time[1]]
end
neuronsComplexityVec

plot(neuronsComplexityVec[:,1],neuronsComplexityVec[:,2]/(10*5))
title("Variation in CPU Time per Epoch with Increasing Hidden Layer Size")
ylabel("Average CPU Seconds Per Epoch")
xlabel("Number of Neurons in the Hidden Layer")



sampleRes = 1
maxLayers= 20
layersComplexityVec = zeros(length(1:sampleRes:maxLayers),2)
iter = 0

for numLayers in 1:sampleRes:maxLayers
    iter+=1
    println(iter)
    tic()
    nFoldCrossValidateSynapticAnnealing(5, 10, [5,int([5*ones(int(numLayers))]),3], annealSynapses_allWeight, getDataRegErr, getDataClassErr, linearTemperatureSchedule_eightTenths, function(x) return(1) end, tanh, irisDataClassed, [1:4], [5:7])
    time=toq()

    layersComplexityVec[iter, :] = [numLayers, time[1]]
end
layersComplexityVec

plot(layersComplexityVec[:,1],layersComplexityVec[:,2]/(10*5))
title("Variation in CPU Time per Epoch with Increasing Hidden Layers")
ylabel("Average CPU Seconds Per Epoch")
xlabel("Number of Hidden Layers (5 Neurons Each)")



plot(layersComplexityVec[:,1]*5,layersComplexityVec[:,2]/(10*5), label="Mulitiple Hidden Layers")
plot(neuronsComplexityVec[:,1],neuronsComplexityVec[:,2]/(10*5), label="Single Hidden Layer")
title("Variation in CPU Time per Epoch with\n Increasing Number of Hidden Neurons")
ylabel("Average CPU Seconds Per Epoch")
xlabel("Number of Neurons in the Hidden Layer")
legend(loc=2)


# Calculate individual epoch time.
# Construct the iris dataset
irisData = readdlm(".\\iris.dat", ',' , Any)
irisData = orthogonalizeDataClasses(irisData, [5])
irisData = normalizeData(irisData)

function calcEpochTime_annealing(trainingFunction, numEpochs, synMatConfigVec, errorFunction, reportErrorFunction, temperatureScheduleFcn, learnRateDecay, actFun, trainData, valData, inputCols, outputCols)

    synapseMatrixIn = createSynapseMatrix(synMatConfigVec)
    tic()
    trainingFunction(numEpochs, errorFunction, reportErrorFunction, temperatureScheduleFcn,  function(x) return(1) end, synapseMatrixIn, actFun, trainData, valData, inputCols, outputCols)
    time = toq()[1]
    return(time/numEpochs)
end

epochTimeAvg_anneal = calcEpochTime_annealing(annealSynapses_allWeight_fixedStep, 100, [5,10,3], getDataRegErr, getDataClassErr, linearTemperatureSchedule_eightTenths, learnRateDecay, tanh, irisData[31:end, :], irisData[1:30, :], [1:4], [5:7])
epochTimeAvg_anneal


function annealSynapses_allWeight_fixedStep(numEpochs, errorFunction, reportErrorFunction, temperatureScheduleFcn, learnRateDecay, synapseMatrixIn, actFun, trainData, valData, inputCols, outputCols)







