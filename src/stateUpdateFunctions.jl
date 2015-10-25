
function updateState(stateTupleIn)

    #  Unpack the input tuple. For readability.
    (temperature, initTemperature,  learnRate, tunnelingField, maxConfigDist, epochsCool, numEpochs, anisotropicField) = stateTupleIn

	# Iterate epoch count.
	numEpochs += 1

    # Decay the temperature by 1, stopping at 0.
    temperature = maximum([temperature-1, 0])

    # Add to the count of cool epochs, if cool.
    epochsCool += int(temperature==0)

    # Set the strength of the tunneling field.
# 	learnRate = (sin(((1/100)*numEpochs)+3*rand())+1)/2

    # Pack and return the state tuple.
    return([temperature, initTemperature,  learnRate, tunnelingField, maxConfigDist, epochsCool, numEpochs, anisotropicField])

end




@everywhere function updateState_q(stateTupleIn)

    #  Unpack the input tuple. For readability.
    (temperature, initTemperature,  learnRate, tunnelingField, maxConfigDist, epochsCool, numEpochs, anisotropicField) = stateTupleIn

	# Iterate epoch count.
	numEpochs += 1

    # Decay the temperature by 1, stopping at 0.
    temperature = maximum([temperature-1, 0])

    # Add to the count of cool epochs, if cool.
    epochsCool += int(temperature==0)

    # Set the strength of the tunneling field.
	if (epochsCool>(3*initTemperature))
		tunnelingField = 1
	end

    # Pack and return the state tuple.
    return([temperature, initTemperature,  learnRate, tunnelingField, maxConfigDist, epochsCool, numEpochs, anisotropicField])

end

@everywhere function updateState_oscillatory(stateTupleIn)

    #  Unpack the input tuple. For readability.
    (temperature, initTemperature,  learnRate, tunnelingField, maxConfigDist, epochsCool, numEpochs, anisotropicField) = stateTupleIn

	# Iterate epoch count.
	numEpochs += 1

    # Decay the temperature by 1, stopping at 0.
    temperature = maximum([temperature-1, 0])

    # Set the strength of the tunneling field.
	tunnelingField = (sin(((1/900)*numEpochs)+3*rand())+1)/2
	temperature = initTemperature*(sin(((1/1000)*numEpochs)+3*rand())+1)/2
	anisotropicField = 1-(sin(((1/500)*numEpochs)+3*rand())+1)/2


    # Pack and return the state tuple.
    return([temperature, initTemperature,  learnRate, tunnelingField, maxConfigDist, epochsCool, numEpochs, anisotropicField])

end

# temp = zeros(15000)
# for numEpochs = 1:15000
# 	temp[numEpochs] = (sin(((1/1000)*numEpochs)+3*rand())+1)/2
# end
# plot(temp)



@everywhere function updateState_q_reheat(stateTupleIn)

    #  Unpack the input tuple. For readability.
    (temperature, initTemperature,  learnRate, tunnelingField, maxConfigDist, epochsCool, numEpochs, anisotropicField) = stateTupleIn
	# Iterate epoch count.
	numEpochs += 1

    # Decay the temperature by 1, stopping at 0.
    temperature = maximum([temperature-1, 0])

    # Add to the count of cool epochs, if cool.
    epochsCool += int(temperature==0)

	period = int(1000/initTemperature)

	convergenceCoef = 0.2
    # Set the strength of the tunneling field.
	if (epochsCool>(((period-1)*convergenceCoef)*initTemperature))
		tunnelingField = 0.1
	end

# 	if (epochsCool>(4*initTemperature))
# 		tunnelingField = 0
# 		learnRate = learnRate
# 	end

	if (epochsCool>((period-1)*initTemperature))
		tunnelingField = 0
		epochsCool = 0
		temperature = initTemperature # (10*(freq+1)*initTemperature/numEpochs)
	end


    # Pack and return the state tuple.
    return([temperature, initTemperature,  learnRate, tunnelingField, maxConfigDist, epochsCool, numEpochs, anisotropicField])

end

@everywhere function updateState_q_only(stateTupleIn)

    #  Unpack the input tuple. For readability.
    (temperature, initTemperature,  learnRate, tunnelingField, maxConfigDist, epochsCool, numEpochs, anisotropicField) = stateTupleIn

	# Iterate epoch count.
	numEpochs += 1


	tunnelingField = 1
	temperature=0

    # Pack and return the state tuple.
    return([temperature, initTemperature,  learnRate, tunnelingField, maxConfigDist, epochsCool, numEpochs, anisotropicField])

end
