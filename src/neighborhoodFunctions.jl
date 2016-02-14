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


# -------------- Cauchy

function cauchy_Isotropic_SynapticPerturbation(synapseMatrix, state)


#     visitingDistibutionSample = minimum([state.maxConfigDist,abs((1/(1-(state.normTunnelingField)))*tan(pi*(rand()-0.5)))])
#     visitingDistibutionSample = state.normTunnelingField * tan(pi*(rand()-0.5))

    visitingDistibutionSample = 1*tan(pi*(rand()-0.5)).*sum(int(bool(synapseMatrix)))

#     visitingDistibutionSample = minimum([(1/state.normTunnelingField)*tan(pi*(rand()-0.5))*sum(int(bool(synapseMatrix))), state.maxConfigDist])


    stepSize = state.learnRate.*visitingDistibutionSample

    anisotropicityMatrix = ones(size(synapseMatrix)).*int(bool(synapseMatrix))
    anisotropicityMatrix = (anisotropicityMatrix./(sum(anisotropicityMatrix))).*((2*int(rand(size(synapseMatrix)).>0.5))-1)

    tan(pi*(rand(size(synapseMatrix))-0.5)).*int(bool(synapseMatrix))


    synapsePerturbation = stepSize.*tan(pi*(rand(size(synapseMatrix))-0.5)).*int(bool(synapseMatrix))
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


function cauchy_VariablyAnisotropic_SynapticPerturbation(synapseMatrix, state)



#     visitingDistibutionSample = minimum([state.maxConfigDist,1/state.normTunnelingField])*tan(pi*(rand()-0.5))

#     visitingDistibutionSample = minimum([state.maxConfigDist,abs((1/(1-(state.normTunnelingField)))*tan(pi*(rand()-0.5)))]).*sum(int(bool(synapseMatrix)))

    visitingDistibutionSample = 1*tan(pi*(rand()-0.5))*sum(int(bool(synapseMatrix)))

    stepSize =  state.learnRate*visitingDistibutionSample

    anisotropicityMatrix = rand(size(synapseMatrix)).*int(bool(synapseMatrix))
    anisotropicityMatrix = (anisotropicityMatrix./(sum(anisotropicityMatrix)))
    anisotropicityMatrix = (anisotropicityMatrix.^(1/(1-(0.9*state.anisotropicField)))).*((2*int(rand(size(synapseMatrix)).>0.5))-1)

    synapsePerturbation = stepSize.*anisotropicityMatrix



    return(Any[synapsePerturbation, stepSize])
end


# -------------- Uniform




function uniform_Isotropic_SynapticPerturbation(synapseMatrix, state)



    synapsePerturbation = state.learnRate.*(2.*rand(size(synapseMatrix)).1).*int(bool(synapseMatrix))

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



# -------------- Placeholders





function gsa_SynapticPerturbation(synapseMatrix, state)


    # With probability of the value of the tunneling field, take a large, random step.
#     stepSize =  learnRate*(1+(maxConfigDist*exp(-rand())*int(rand()<(tunnelingField))))
    stepSize =  state.learnRate

    # Construct a rand matrix of equal size to synapseMatrix and mask it by existing nuerons.
    randMat = rand(size(synapseMatrix)).*int(bool(synapseMatrix))


    synapsePerturbation = stepSize.*(randMat./(sum(randMat))).*sum(int(bool(synapseMatrix))).*((2*int(rand(size(synapseMatrix)).>0.5))-1)
#    synapsePerturbation = stepSize.*((randMat./(sum(randMat))).*((2*int(rand(size(synapseMatrix)).>0.5))-1))

    return(Any[synapsePerturbation, stepSize])
end
