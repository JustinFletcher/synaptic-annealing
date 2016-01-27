using PyPlot
@everywhere using BackpropNeuralNet
@everywhere cd("\\fletcher-thesis")

@everywhere include("$(pwd())\\src\\"*"nativeNetsToSynMats.jl")

netIn = init_network([15,50,50,50,50,10])
netIn.learning_rate = 1
netIn.propagation_function = tanh
synapseMatrix = getNetworkSynapseMatrix(netIn)

function plotAnisotropicityMatrix_SameScale(anisotropicityMatrixList)

    subplotCount = 0

    numMatrixTypes = size(anisotropicityMatrixList)[1]

    figure()

    biggestWeight = 0
    for anisotropicityMatrix in anisotropicityMatrixList
        maxVal = maximum(maximum(anisotropicityMatrix))
        if maxVal > biggestWeight
            biggestWeight = maxVal
        end
    end


    for matrixTypeNum in 1:numMatrixTypes

        anisotropicityMatrix = anisotropicityMatrixList[matrixTypeNum]

        numLayers = size(anisotropicityMatrix)[3]


        for layerNum in 1:numLayers

            subplotCount+=1

            subplot(numMatrixTypes, numLayers, subplotCount)

            imshow(anisotropicityMatrix[:,:,layerNum], interpolation="nearest", cmap="RdBu", vmin=-biggestWeight, vmax=biggestWeight)

            colorbar()

        end

    end

end

function plotAnisotropicityMatrix_RowScale(anisotropicityMatrixList, rowLableList)

    subplotCount = 0

    numMatrixTypes = size(anisotropicityMatrixList)[1]

    figure()

    for matrixTypeNum in 1:numMatrixTypes

        anisotropicityMatrix = anisotropicityMatrixList[matrixTypeNum]

        biggestWeight = maximum(maximum(anisotropicityMatrix))
        numLayers = size(anisotropicityMatrix)[3]

        printRowLable = true
        for layerNum in 1:numLayers

            subplotCount+=1

            subplot(numMatrixTypes, numLayers, subplotCount)

            imshow(anisotropicityMatrix[:,:,layerNum], interpolation="nearest", cmap="RdBu", vmin=-biggestWeight, vmax=biggestWeight)

            colorbar()


            if printRowLable
                printRowLable = false
                ylabel(rowLableList[matrixTypeNum])
            end

        end

    end

end

function runAnisotropicExperiment(synapseMatrix)

    println("--------------------")

    anisotropicityMatrixList = Array[]

    # ----- Isotrpic
    anisotropicityMatrix = ones(size(synapseMatrix)).*int(bool(synapseMatrix))
    anisotropicityMatrix = (anisotropicityMatrix./(sum(anisotropicityMatrix))).*((2*int(rand(size(synapseMatrix)).>0.5))-1)
    push!(anisotropicityMatrixList, anisotropicityMatrix)

    println((sum(anisotropicityMatrix)))

    # ----- Uniform
    anisotropicityMatrix = rand(size(synapseMatrix)).*int(bool(synapseMatrix))
    anisotropicityMatrix =(anisotropicityMatrix./(sum(anisotropicityMatrix))).*((2*int(rand(size(synapseMatrix)).>0.5))-1)
    push!(anisotropicityMatrixList, anisotropicityMatrix)

    println((sum(anisotropicityMatrix)))

    # ----- Normal
    anisotropicityMatrix = abs(randn(size(synapseMatrix)).*int(bool(synapseMatrix)))
    anisotropicityMatrix =(anisotropicityMatrix./(sum(anisotropicityMatrix))).*((2*int(rand(size(synapseMatrix)).>0.5))-1)
    push!(anisotropicityMatrixList, anisotropicityMatrix)

    println((sum(anisotropicityMatrix)))


    # ----- L1 Distance Weighted Uniform

    (numRows, numCols, numLayers) = size(synapseMatrix)
    centerX = rand(1:numRows)
    centerY = rand(1:numCols)
    centerZ = rand(1:numLayers)

    centerX = 10
    centerY = 25
    centerZ = 3

    anisotropicityMatrix = rand(size(synapseMatrix)).*[ 1/((abs(x-centerX)+abs(y-centerY)+abs(z-centerZ+1/0.2))) for x=1:numRows, y=1:numCols, z=1:numLayers].*int(bool(synapseMatrix))
    anisotropicityMatrix =(anisotropicityMatrix./(sum(anisotropicityMatrix))).*((2*int(rand(size(synapseMatrix)).>0.5))-1)
    #L1DistanceMatrix = [ (abs(x-centerX),abs(y-centerY),abs(z-centerZ)) for x=1:numRows, y=1:numCols, z=1:numLayers].*int(bool(synapseMatrix))
    push!(anisotropicityMatrixList, anisotropicityMatrix)
    println((sum(anisotropicityMatrix)))

      # ----- Row Weighted Uniform

    (numRows, numCols, numLayers) = size(synapseMatrix)
    centerX = rand(1:numRows)
    centerZ = rand(1:numLayers)

    centerY = 10
    centerZ = 3

    anisotropicityMatrix = [int((y==centerY)&&(z==centerZ)) for x=1:numRows, y=1:numCols, z=1:numLayers].*int(bool(synapseMatrix))
    #anisotropicityMatrix =(anisotropicityMatrix./(sum(anisotropicityMatrix))).*((2*int(rand(size(synapseMatrix)).>0.5))-1)
    #L1DistanceMatrix = [ (abs(x-centerX),abs(y-centerY),abs(z-centerZ)) for x=1:numRows, y=1:numCols, z=1:numLayers].*int(bool(synapseMatrix))
    push!(anisotropicityMatrixList, anisotropicityMatrix)
    println((sum(anisotropicityMatrix)))

    plotAnisotropicityMatrix_SameScale(anisotropicityMatrixList)


end

runAnisotropicExperiment(synapseMatrix)

function runNormativeAnisotorpicityTunableExperiment(synapseMatrix)
    println("--------------------")

    anisotropicityMatrixList = Array[]

    (numRows, numCols, numLayers) = size(synapseMatrix)
    centerX = 10
    centerY = 25
    centerZ = 3


    rowLableList = [0.01, 0.1, 0.2, 0.4, 0.8, 1.1, 5]

    for controlParam in rowLableList

        anisotropicityMatrix = rand(size(synapseMatrix)).*[ 1/(((abs(x-centerX)+abs(y-centerY)+abs(z-centerZ)))+(1/controlParam)) for x=1:numRows, y=1:numCols, z=1:numLayers].*int(bool(synapseMatrix))
        anisotropicityMatrix =(anisotropicityMatrix./(sum(anisotropicityMatrix))).*((2*int(rand(size(synapseMatrix)).>0.5))-1)
        #L1DistanceMatrix = [ (abs(x-centerX),abs(y-centerY),abs(z-centerZ)) for x=1:numRows, y=1:numCols, z=1:numLayers].*int(bool(synapseMatrix))
        push!(anisotropicityMatrixList, anisotropicityMatrix)
    end
    plotAnisotropicityMatrix_RowScale(anisotropicityMatrixList, rowLableList)


end

runNormativeAnisotorpicityTunableExperiment(synapseMatrix)

