# -------------- Nieghborhood Functions


@everywhere include("$(pwd())\\src\\"*"gsa.jl")



function gaussian_Isotropic_SynapticPerturbation(synapseMatrix, state)

    # With probability of the value of the tunneling field, take a large, random step.
#     stepSize =  learnRate*(1+(maxConfigDist*exp(-rand())*int(rand()<(tunnelingField))))

    visitingDistibutionSample = randn().*sum(int(bool(synapseMatrix)))


    stepSize = state.learnRate.*visitingDistibutionSample

    anisotropicityMatrix = ones(size(synapseMatrix)).*int(bool(synapseMatrix))
    anisotropicityMatrix = (anisotropicityMatrix./(sum(anisotropicityMatrix))).*((2*int(rand(size(synapseMatrix)).>0.5))-1)
#     anisotropicityMatrix = anisotropicityMatrix.*sum(int(bool(synapseMatrix)))

    synapsePerturbation = stepSize.*anisotropicityMatrix
#    synapsePerturbation = stepSize.*((randMat./(sum(randMat))).*((2*int(rand(size(synapseMatrix)).>0.5))-1))

    synapsePerturbation =  state.learnRate.*randn(size(synapseMatrix)).*int(bool(synapseMatrix))

    return(Any[synapsePerturbation, stepSize])
end


function gaussian_uniformAnisotropic_SynapticPerturbation(synapseMatrix, state)

    # With probability of the value of the tunneling field, take a large, random step.
#     stepSize =  learnRate*(1+(maxConfigDist*exp(-rand())*int(rand()<(tunnelingField))))

    visitingDistibutionSample = randn().*sum(int(bool(synapseMatrix)))


    stepSize = state.learnRate.*visitingDistibutionSample

    anisotropicityMatrix = rand(size(synapseMatrix)).*int(bool(synapseMatrix))
    anisotropicityMatrix = (anisotropicityMatrix./(sum(anisotropicityMatrix))).*((2*int(rand(size(synapseMatrix)).>0.5))-1)
#     anisotropicityMatrix = anisotropicityMatrix.*sum(int(bool(synapseMatrix)))

    synapsePerturbation = stepSize.*anisotropicityMatrix
#    synapsePerturbation = stepSize.*((randMat./(sum(randMat))).*((2*int(rand(size(synapseMatrix)).>0.5))-1))

    return(Any[synapsePerturbation, stepSize])
end

function gaussian_variableAnisotropic_SynapticPerturbation(synapseMatrix, state)

    # With probability of the value of the tunneling field, take a large, random step.
#     stepSize =  learnRate*(1+(maxConfigDist*exp(-rand())*int(rand()<(tunnelingField))))

    visitingDistibutionSample = randn().*sum(int(bool(synapseMatrix)))


    stepSize = state.learnRate.*visitingDistibutionSample

    anisotropicityMatrix = rand(size(synapseMatrix)).*int(bool(synapseMatrix))
    anisotropicityMatrix = (anisotropicityMatrix./(sum(anisotropicityMatrix)))
    anisotropicityMatrix = (anisotropicityMatrix.^(1/(1-(0.9*state.anisotropicField)))).*((2*int(rand(size(synapseMatrix)).>0.5))-1)

    synapsePerturbation = stepSize.*anisotropicityMatrix
#    synapsePerturbation = stepSize.*((randMat./(sum(randMat))).*((2*int(rand(size(synapseMatrix)).>0.5))-1))

    return(Any[synapsePerturbation, stepSize])
end

function gaussian_NormativeAnisotropic_SynapticPerturbation(synapseMatrix, state)

    # With probability of the value of the tunneling field, take a large, random step.
#     stepSize =  learnRate*(1+(maxConfigDist*exp(-rand())*int(rand()<(tunnelingField))))

    visitingDistibutionSample = randn().*sum(int(bool(synapseMatrix)))


    stepSize = state.learnRate.*visitingDistibutionSample

    (numRows, numCols, numLayers) = size(synapseMatrix)
    centerX = rand(1:numRows)
    centerY = rand(1:numCols)
    centerZ = rand(1:numLayers)
    anisotropicityMatrix = rand(size(synapseMatrix)).*[ 1/((abs(x-centerX)+abs(y-centerY)+abs(z-centerZ+1/0.1))) for x=1:numRows, y=1:numCols, z=1:numLayers].*int(bool(synapseMatrix))
    anisotropicityMatrix =(anisotropicityMatrix./(sum(anisotropicityMatrix))).*((2*int(rand(size(synapseMatrix)).>0.5))-1)


    synapsePerturbation = stepSize.*anisotropicityMatrix
#    synapsePerturbation = stepSize.*((randMat./(sum(randMat))).*((2*int(rand(size(synapseMatrix)).>0.5))-1))

    return(Any[synapsePerturbation, stepSize])
end




function capValues(v, cap)
  return(v-(v.*(abs(v).>cap))+(sign(v).*cap.*(abs(v).>cap)))
end


# -------------- Cauchy

function cauchy_Isotropic_SynapticPerturbation(synapseMatrix, state)


    ## OLD ^
    synapsePerturbation = state.learnRate.*tan(pi.*(rand(size(synapseMatrix)).-0.5)).*int(bool(synapseMatrix))

    #synapsePerturbation = capValues(synapsePerturbation, 50)
     return(Any[synapsePerturbation, sum(synapsePerturbation)])
end


function cauchy_UniformlyAnisotropic_SynapticPerturbation(synapseMatrix, state)

#     visitingDistibutionSample = minimum([state.maxConfigDist,1/state.normTunnelingField])*tan(pi*(rand()-0.5))
#     visitingDistibutionSample = minimum([state.maxConfigDist,abs((1/(1-(state.normTunnelingField)))*tan(pi*(rand()-0.5)))])

#     visitingDistibutionSample = minimum([(1/state.normTunnelingField)*tan(pi*(rand()-0.5))*sum(int(bool(synapseMatrix))), state.maxConfigDist])

    visitingDistibutionSample = 1*tan(pi*(rand()-0.5)).*sum(int(bool(synapseMatrix)))
    stepSize =  state.learnRate*visitingDistibutionSample

    anisotropicityMatrix = rand(size(synapseMatrix)).*int(bool(synapseMatrix))
    anisotropicityMatrix =(anisotropicityMatrix./(sum(anisotropicityMatrix))).*((2*int(rand(size(synapseMatrix)).>0.5))-1)

    synapsePerturbation = stepSize.*anisotropicityMatrix

    return(Any[synapsePerturbation, stepSize])
end



function cauchy_MaxAnisotropic_SynapticPerturbation(synapseMatrix, state)



#     visitingDistibutionSample = minimum([state.maxConfigDist,1/state.normTunnelingField])*tan(pi*(rand()-0.5))

#     visitingDistibutionSample = minimum([state.maxConfigDist,abs((1/(1-(state.normTunnelingField)))*tan(pi*(rand()-0.5)))]).*sum(int(bool(synapseMatrix)))

    visitingDistibutionSample = 1*tan(pi*(rand()-0.5))*sum(int(bool(synapseMatrix)))

    stepSize =  state.learnRate*visitingDistibutionSample


    anisotropicityMatrix = rand(size(synapseMatrix)).*int(bool(synapseMatrix))

    anisotropicityMatrix =int(anisotropicityMatrix.==maximum(anisotropicityMatrix))
    anisotropicityMatrix =(anisotropicityMatrix./(sum(anisotropicityMatrix))).*((2*int(rand(size(synapseMatrix)).>0.5))-1)

    synapsePerturbation = stepSize.*anisotropicityMatrix



    return(Any[synapsePerturbation, stepSize])
end

function cauchy_NormativeAnisotropic_SynapticPerturbation(synapseMatrix, state)


    (numRows, numCols, numLayers) = size(synapseMatrix)
    centerX = rand(1:numRows)
    centerY = rand(1:numCols)
    centerZ = rand(1:numLayers)


    ## OLD ^
    visitingDistributionSampleMatrix = stepSize.*tan(pi.*(rand(size(synapseMatrix)).-0.5)).*int(bool(synapseMatrix))


    anisotropicityMatrix = rand(size(synapseMatrix)).*[ 1/((abs(x-centerX)+abs(y-centerY)+abs(z-centerZ+1/100))) for x=1:numRows, y=1:numCols, z=1:numLayers].*int(bool(synapseMatrix))
    anisotropicityMatrix =(anisotropicityMatrix./(sum(anisotropicityMatrix)))

    synapsePerturbation = visitingDistributionSampleMatrix .* anisotropicityMatrix

    return(Any[synapsePerturbation, stepSize])
end

function cauchy_NeuralAnisotropic_SynapticPerturbation(synapseMatrix, state)


    (numRows, numCols, numLayers) = size(synapseMatrix)
    centerX = rand(1:numRows)
    centerY = rand(1:numCols)
    centerZ = rand(1:numLayers)


    visitingDistributionSampleMatrix = state.learnRate*tan(pi.*(rand(size(synapseMatrix)).-0.5)).*int(bool(synapseMatrix))

    anisotropicityMatrix = [int((y==centerY)&&(z==centerZ)) for x=1:numRows, y=1:numCols, z=1:numLayers].*int(bool(synapseMatrix))

    synapsePerturbation = visitingDistributionSampleMatrix .* anisotropicityMatrix

    return(Any[synapsePerturbation, sum(synapsePerturbation)])
end


# -------------- Uniform




function uniform_Isotropic_SynapticPerturbation(synapseMatrix, state)


#     visitingDistibutionSample = ((state.maxConfigDist*(rand()))).*sum(int(bool(synapseMatrix)))

#     stepSize = state.learnRate.*visitingDistibutionSample

#     anisotropicityMatrix = ones(size(synapseMatrix)).*int(bool(synapseMatrix))
#     anisotropicityMatrix = (anisotropicityMatrix./(sum(anisotropicityMatrix))).*((2*int(rand(size(synapseMatrix)).>0.5))-1)

    synapsePerturbation = state.learnRate.*(rand(size(synapseMatrix))-0.5).*int(bool(synapseMatrix))
    return(Any[synapsePerturbation, sum(synapsePerturbation)])

end

function uniform_UniformlyAnisotropic_SynapticPerturbation(synapseMatrix, state)

    visitingDistibutionSample = ((state.maxConfigDist*(rand()))).*sum(int(bool(synapseMatrix)))

    stepSize =  state.learnRate*visitingDistibutionSample

    anisotropicityMatrix = rand(size(synapseMatrix)).*int(bool(synapseMatrix))
    anisotropicityMatrix =(anisotropicityMatrix./(sum(anisotropicityMatrix))).*((2*int(rand(size(synapseMatrix)).>0.5))-1)

    synapsePerturbation = stepSize.*anisotropicityMatrix

    return(Any[synapsePerturbation, stepSize])
end

function uniform_VariablyAnisotropic_SynapticPerturbation(synapseMatrix, state)



    visitingDistibutionSample = ((state.maxConfigDist*(rand()))).*sum(int(bool(synapseMatrix)))


    stepSize =  state.learnRate*visitingDistibutionSample

    anisotropicityMatrix = rand(size(synapseMatrix)).*int(bool(synapseMatrix))
    anisotropicityMatrix = (anisotropicityMatrix./(sum(anisotropicityMatrix)))
    anisotropicityMatrix = (anisotropicityMatrix.^(1/(1-(0.9*state.anisotropicField)))).*((2*int(rand(size(synapseMatrix)).>0.5))-1)

    synapsePerturbation = stepSize.*anisotropicityMatrix


    return(Any[synapsePerturbation, stepSize])
end

function uniform_NormativeAnisotropic_SynapticPerturbation(synapseMatrix, state)



    visitingDistibutionSample = ((state.maxConfigDist*(rand()))).*sum(int(bool(synapseMatrix)))


    stepSize =  state.learnRate*visitingDistibutionSample

    (numRows, numCols, numLayers) = size(synapseMatrix)
    centerX = rand(1:numRows)
    centerY = rand(1:numCols)
    centerZ = rand(1:numLayers)

    anisotropicityMatrix = rand(size(synapseMatrix)).*[ 1/((abs(x-centerX)+abs(y-centerY)+abs(z-centerZ+1/0.1))) for x=1:numRows, y=1:numCols, z=1:numLayers].*int(bool(synapseMatrix))
    anisotropicityMatrix =(anisotropicityMatrix./(sum(anisotropicityMatrix))).*((2*int(rand(size(synapseMatrix)).>0.5))-1)

    synapsePerturbation = stepSize.*anisotropicityMatrix


    return(Any[synapsePerturbation, stepSize])
end


# -------------- GSA

function gsa_Isotropic_SynapticPerturbation(synapseMatrix, state)


    synapsePerturbation = state.learnRate.*randgsa(2.6, 1, 1, size(synapseMatrix)).*int(bool(synapseMatrix))

    return(Any[synapsePerturbation, sum(synapsePerturbation)])
end


function gsa_WeightAnisotropic_SynapticPerturbation(synapseMatrix, state)

    anisotropicity = int(abs(synapseMatrix).>2) .* (rand(size(synapseMatrix)).*synapseMatrix)

    #anisotropicity = int(abs(synapseMatrix).>5) .* (0.5.*synapseMatrix)
    anisotropicity = abs(tanh(synapseMatrix)) .* 0.5.*(synapseMatrix)
    synapsePerturbation = state.learnRate.*(randgsa(2.6, 1, 1, size(synapseMatrix)).*int(bool(synapseMatrix)))-anisotropicity

    return(Any[synapsePerturbation, sum(synapsePerturbation)])
end

# -------------- Placeholders

function cauchy_GeneralizedAnisotropic_SynapticPerturbation(synapseMatrix, state)



    visitingDistibutionSample = 1*tan(pi*(rand()-0.5))*sum(int(bool(synapseMatrix)))

    stepSize =  state.learnRate*visitingDistibutionSample



    anisotropicityMatrix = rand(size(synapseMatrix)).*int(bool(synapseMatrix))
    anisotropicityMatrix = (anisotropicityMatrix./(sum(anisotropicityMatrix)))
    anisotropicityMatrix = (anisotropicityMatrix.^(1/(1-(0.9*state.anisotropicField)))).*((2*int(rand(size(synapseMatrix)).>0.5))-1)



    synapsePerturbation = stepSize.*anisotropicityMatrix



    return(Any[synapsePerturbation, stepSize])
end






function gsa_NormativeAnisotropic_SynapticPerturbation(synapseMatrix, state)



    visitingDistibutionSample = ((state.maxConfigDist*(rand()))).*sum(int(bool(synapseMatrix)))

    stepSize =  state.learnRate*visitingDistibutionSample

    (numRows, numCols, numLayers) = size(synapseMatrix)
    centerX = rand(1:numRows)
    centerY = rand(1:numCols)
    centerZ = rand(1:numLayers)

    anisotropicityMatrix = rand(size(synapseMatrix)).*[ 1/((abs(x-centerX)+abs(y-centerY)+abs(z-centerZ+1/0.1))) for x=1:numRows, y=1:numCols, z=1:numLayers].*int(bool(synapseMatrix))
    anisotropicityMatrix =(anisotropicityMatrix./(sum(anisotropicityMatrix))).*((2*int(rand(size(synapseMatrix)).>0.5))-1)

    synapsePerturbation = stepSize.*anisotropicityMatrix


    return(Any[synapsePerturbation, stepSize])
end
