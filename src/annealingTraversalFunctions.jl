# -------------- Nieghborhood Functions





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

# -------------- Exponential



function exponential_Isotropic_SynapticPerturbation(synapseMatrix, state)

    visitingDistibutionSample = (-log(rand())/(0.1)).*sum(int(bool(synapseMatrix)))


    stepSize = state.learnRate.*visitingDistibutionSample

    anisotropicityMatrix = ones(size(synapseMatrix)).*int(bool(synapseMatrix))
    anisotropicityMatrix = (anisotropicityMatrix./(sum(anisotropicityMatrix))).*((2*int(rand(size(synapseMatrix)).>0.5))-1)

    synapsePerturbation = stepSize.*anisotropicityMatrix
    return(Any[synapsePerturbation, stepSize])
end

function exponential_UniformlyAnisotropic_SynapticPerturbation(synapseMatrix, state)


    visitingDistibutionSample = (-log(rand())/(0.1)).*sum(int(bool(synapseMatrix)))


    stepSize =  state.learnRate*visitingDistibutionSample

    anisotropicityMatrix = rand(size(synapseMatrix)).*int(bool(synapseMatrix))
    anisotropicityMatrix =(anisotropicityMatrix./(sum(anisotropicityMatrix))).*((2*int(rand(size(synapseMatrix)).>0.5))-1)

    synapsePerturbation = stepSize.*anisotropicityMatrix

    return(Any[synapsePerturbation, stepSize])
end

function exponential_VariablyAnisotropic_SynapticPerturbation(synapseMatrix, state)


    visitingDistibutionSample = (-log(rand())/(0.1)).*sum(int(bool(synapseMatrix)))

    stepSize =  state.learnRate*visitingDistibutionSample

    anisotropicityMatrix = rand(size(synapseMatrix)).*int(bool(synapseMatrix))
    anisotropicityMatrix = (anisotropicityMatrix./(sum(anisotropicityMatrix)))
    anisotropicityMatrix = (anisotropicityMatrix.^(1/(1-(0.9*state.anisotropicField)))).*((2*int(rand(size(synapseMatrix)).>0.5))-1)

    synapsePerturbation = stepSize.*anisotropicityMatrix



    return(Any[synapsePerturbation, stepSize])

end

function exponential_NormativeAnisotropic_SynapticPerturbation(synapseMatrix, state)


    visitingDistibutionSample = (-log(rand())/(0.1)).*sum(int(bool(synapseMatrix)))

    stepSize =  state.learnRate*visitingDistibutionSample

    anisotropicityMatrix = rand(size(synapseMatrix)).*int(bool(synapseMatrix))
    anisotropicityMatrix = (anisotropicityMatrix./(sum(anisotropicityMatrix)))
    anisotropicityMatrix = (anisotropicityMatrix.^(1/(1-(0.9*state.anisotropicField)))).*((2*int(rand(size(synapseMatrix)).>0.5))-1)

    synapsePerturbation = stepSize.*anisotropicityMatrix



    return(Any[synapsePerturbation, stepSize])

end

function exponential_NormativeAnisotropic_SynapticPerturbation(synapseMatrix, state)


    visitingDistibutionSample = (-log(rand())/(0.1)).*sum(int(bool(synapseMatrix)))

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

# -------------- Cauchy

function cauchy_Isotropic_SynapticPerturbation(synapseMatrix, state)


#     visitingDistibutionSample = minimum([state.maxConfigDist,abs((1/(1-(state.normTunnelingField)))*tan(pi*(rand()-0.5)))])
#     visitingDistibutionSample = state.normTunnelingField * tan(pi*(rand()-0.5))

    visitingDistibutionSample = 1*tan(pi*(rand()-0.5)).*sum(int(bool(synapseMatrix)))

#     visitingDistibutionSample = minimum([(1/state.normTunnelingField)*tan(pi*(rand()-0.5))*sum(int(bool(synapseMatrix))), state.maxConfigDist])


    stepSize = state.learnRate.*visitingDistibutionSample

    anisotropicityMatrix = ones(size(synapseMatrix)).*int(bool(synapseMatrix))
    anisotropicityMatrix = (anisotropicityMatrix./(sum(anisotropicityMatrix))).*((2*int(rand(size(synapseMatrix)).>0.5))-1)

    synapsePerturbation = stepSize.*anisotropicityMatrix

    ## OLD ^
    synapsePerturbation = stepSize.*tan(pi.*(rand(size(synapseMatrix)).-0.5))

    return(Any[synapsePerturbation, stepSize])
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



#     visitingDistibutionSample = minimum([state.maxConfigDist,1/state.normTunnelingField])*tan(pi*(rand()-0.5))

#     visitingDistibutionSample = minimum([state.maxConfigDist,abs((1/(1-(state.normTunnelingField)))*tan(pi*(rand()-0.5)))]).*sum(int(bool(synapseMatrix)))

    visitingDistibutionSample = 1*tan(pi*(rand()-0.5))*sum(int(bool(synapseMatrix)))

    stepSize =  state.learnRate*visitingDistibutionSample


    (numRows, numCols, numLayers) = size(synapseMatrix)
    centerX = rand(1:numRows)
    centerY = rand(1:numCols)
    centerZ = rand(1:numLayers)

    anisotropicityMatrix = rand(size(synapseMatrix)).*[ 1/((abs(x-centerX)+abs(y-centerY)+abs(z-centerZ+1/100))) for x=1:numRows, y=1:numCols, z=1:numLayers].*int(bool(synapseMatrix))
    anisotropicityMatrix =(anisotropicityMatrix./(sum(anisotropicityMatrix))).*((2*int(rand(size(synapseMatrix)).>0.5))-1)
    synapsePerturbation = stepSize.*anisotropicityMatrix



    ## OLD ^
    visitingDistributionSampleMatrix = stepSize.*tan(pi.*(rand(size(synapseMatrix)).-0.5))


    anisotropicityMatrix = rand(size(synapseMatrix)).*[ 1/((abs(x-centerX)+abs(y-centerY)+abs(z-centerZ+1/100))) for x=1:numRows, y=1:numCols, z=1:numLayers].*int(bool(synapseMatrix))
    anisotropicityMatrix =(anisotropicityMatrix./(sum(anisotropicityMatrix))).*((2*int(rand(size(synapseMatrix)).>0.5))-1)

    synapsePerturbation = visitingDistributionSampleMatrix .* anisotropicityMatrix

    return(Any[synapsePerturbation, stepSize])
end


# -------------- Uniform




function uniform_Isotropic_SynapticPerturbation(synapseMatrix, state)


    visitingDistibutionSample = ((state.maxConfigDist*(rand()))).*sum(int(bool(synapseMatrix)))


    stepSize = state.learnRate.*visitingDistibutionSample

    anisotropicityMatrix = ones(size(synapseMatrix)).*int(bool(synapseMatrix))
    anisotropicityMatrix = (anisotropicityMatrix./(sum(anisotropicityMatrix))).*((2*int(rand(size(synapseMatrix)).>0.5))-1)

    synapsePerturbation = stepSize.*anisotropicityMatrix
    return(Any[synapsePerturbation, stepSize])

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
