using PyPlot
@everywhere using BackpropNeuralNet
@everywhere cd("\\mscs-thesis")

@everywhere include("$(pwd())\\src\\"*"nativeNetsToSynMats.jl")

netIn = init_network([3,5,10,5,2])
netIn.learning_rate = 1
netIn.propagation_function = tanh
synapseMatrix = getNetworkSynapseMatrix(netIn)

function plotAnisotropicityMatrix(anisotropicityMatrixList)

    subplotCount = 0

    numMatrixTypes = size(anisotropicityMatrixList)[1]

    figure()

    biggestWeight = maximum(maximum(maximum(abs(anisotropicityMatrix))))

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

function runAnisotropicExperiment()

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

    # ----- Variable
    anisotropicField = 0.25
    anisotropicityMatrix = rand(size(synapseMatrix)).*int(bool(synapseMatrix))
    anisotropicityMatrix = (anisotropicityMatrix./(sum(anisotropicityMatrix)))
    anisotropicityMatrix = (anisotropicityMatrix.^(1/(1-(0.9*anisotropicField)))).*((2*int(rand(size(synapseMatrix)).>0.5))-1)
    push!(anisotropicityMatrixList, anisotropicityMatrix)

    println((sum(anisotropicityMatrix)))

    # ----- Directional
    anisotropicField = 0.05
    anisotropicityMatrix = rand(size(synapseMatrix)).*int(bool(synapseMatrix))
    anisotropicityMatrix = (anisotropicityMatrix./(sum(anisotropicityMatrix)))
    anisotropicityMatrix = (anisotropicityMatrix.^(1/(1-(0.9*anisotropicField)))).*((2*int(rand(size(synapseMatrix)).>0.5))-1)
    push!(anisotropicityMatrixList, anisotropicityMatrix)

    println((sum(anisotropicityMatrix)))

    anisotropicityMatrixList

    plotAnisotropicityMatrix(anisotropicityMatrixList)


end

runAnisotropicExperiment()

