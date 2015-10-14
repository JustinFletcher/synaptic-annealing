# ---- Annealing Training Functions ----

function quantumAnisotropicSynapticPerturbation(synapseMatrix, stateTupleIn)

    (temperature, initTemperature,  learnRate, tunnelingField, maxConfigDist, epochsCool, numEpochs, anisotropicField) = stateTupleIn

    # With probability of the value of the tunneling field, take a large, random step.
#     stepSize =  learnRate*(1+(maxConfigDist*exp(-rand())*int(rand()<(tunnelingField))))
    stepSize =  learnRate*(1+(maxConfigDist*rand()*int(rand()<(tunnelingField))))

    # Construct a rand matrix of equal size to synapseMatrix and mask it by existing nuerons.
    randMat = rand(size(synapseMatrix)).*int(bool(synapseMatrix))


	normMat = randMat./sum(randMat)


	anisotropicMat = normMat.^(1/(1-(0.9*anisotropicField)))
	anisotropicNormMat = anisotropicMat./sum(anisotropicMat)

    perturbMat = stepSize.*(anisotropicNormMat).*((2*int(rand(size(synapseMatrix)).>0.5))-1)

    return(Any[perturbMat, stepSize])
end




function quantumSynapticChange(synapseMatrix, stateTupleIn)

    (temperature, initTemperature,  learnRate, tunnelingField, maxConfigDist, epochsCool, numEpochs, anisotropicField) = stateTupleIn

    # With probability of the value of the tunneling field, take a large, random step.
#     stepSize =  learnRate*(1+(maxConfigDist*exp(-rand())*int(rand()<(tunnelingField))))
    stepSize =  learnRate*(1+(maxConfigDist*rand()*int(rand()<(tunnelingField))))

    # Construct a rand matrix of equal size to synapseMatrix and mask it by existing nuerons.
    randMat = rand(size(synapseMatrix)).*int(bool(synapseMatrix))

    synapseChange = stepSize.*((randMat./(sum(randMat))).*((2*int(rand(size(synapseMatrix)).>0.5))-1))

    return(Any[synapseChange, stepSize])
end



function fixedStepSizeOmniDimSynapticChange(synapseMatrix, stateTupleIn)

    (temperature, initTemperature,  learnRate, tunnelingField, maxConfigDist, epochsCool, numEpochs, anisotropicField) = stateTupleIn

#         randMat = rand(size(synapseMatrix)).*int(bool(synapseMatrix))
#         synapseChange = stepSize.*((randMat./(sum(randMat))).*((2*int(rand(size(synapseMatrix)).>0.5))-1))
    # Set the step size to the learn rate.
    stepSize = learnRate

    # Construct a matrix of values which will modify the weights of the synapses.
    # Scalar Multiple .* Random Matrix Intersect Exists Synapse Which Adds To One .* Random Negator
    randMat = rand(size(synapseMatrix)).*int(bool(synapseMatrix))

    synapseChange = stepSize.*((randMat./(sum(randMat))).*((2*int(rand(size(synapseMatrix)).>0.5))-1))

    return(Any[synapseChange, stepSize])
end

function omniDimSynapticChange(synapseMatrix, stateTupleIn)

    (temperature, initTemperature,  learnRate, tunnelingField, maxConfigDist, epochsCool, numEpochs, anisotropicField) = stateTupleIn

    stepSize = learnRate

    # Construct a matrix of values which will modify the weights of the synapses.
    synapseChange = stepSize.*(2*(rand(size(synapseMatrix))-0.5)).*int(bool(synapseMatrix))

    return(Any[synapseChange, stepSize])
end

function singleDimSynapticChange(synapseMatrix, stateTupleIn)

    (temperature, initTemperature,  learnRate, tunnelingField, maxConfigDist, epochsCool, numEpochs, anisotropicField) = stateTupleIn

    stepSize = learnRate

    # Construct a matrix of values which will modify the weights of the synapses.
    randMat = rand(size(synapseMatrix)).*int(bool(synapseMatrix))

    synapseChange = stepSize.*int(randMat.==(maximum(randMat))).*(2*(rand()-0.5))

    return(Any[synapseChange, stepSize])
end


