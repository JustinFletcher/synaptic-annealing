module AnnealingState


	export State

	type State

		# Constants
		maxConfigDist

		# Initializations
		initTunnelingField
		initTemperature
		initLearnRate

		# Counting variables
		epochsCool
		epochsComplete

		# Field variables
		learnRate
		temperature
		anisotropicField
		tunnelingField

		# Reporting variables
		trainError
		valError
		minTrainError
		minValError
		normTunnelingField
		normTemperature

		# Constructor for type State.
		function State(initTemperature, initLearnRate, initTunnelingField, maxConfigDist)

			temperature = initTemperature
			learnRate = initLearnRate
			tunnelingField = initTunnelingField
			epochsCool = 0
			epochsComplete = 0
			anisotropicField = 0
			minTrainError = 1
			minValError = 1
			trainError = 1
			valError = 1

			if (initTunnelingField == 0)
				normTunnelingField = 0
			else
				normTunnelingField = tunnelingField/initTunnelingField
			end

			if (initTemperature == 0)
				normTemperature = 0
			else
				normTemperature = temperature/initTemperature
			end


			new (maxConfigDist, initTunnelingField, initTemperature, initLearnRate,
				 epochsCool, epochsComplete, learnRate, temperature,
				 anisotropicField, tunnelingField, minTrainError, minValError,
				 trainError, valError,  normTunnelingField, normTemperature)
		end


	end

	function updateState_csa(state::State)

		# Iterate epoch count.
		state.epochsComplete += 1

		# Decay the temperature by 1, stopping at 0.
		state.temperature = state.initTemperature * (1/log(state.epochsComplete))



		state.tunnelingField = 0

		# Compute normalized value.
		state.normTemperature = state.temperature/state.initTemperature

	end

	function updateState_csa_ra(state::State)

		# Iterate epoch count.
		state.epochsComplete += 1

		# Decay the temperature by 1, stopping at 0.
		state.temperature = state.initTemperature * (1/log(state.epochsComplete))

    if ((state.epochsComplete % 200000) == 0)
        state.temperature = state.initTemperature
    end

		state.tunnelingField = 0

		# Compute normalized value.
		state.normTemperature = state.temperature/state.initTemperature

	end



	function updateState_fsa(state::State)

		# Iterate epoch count.
		state.epochsComplete += 1

		# Decay the temperature by 1, stopping at 0.
		state.temperature = state.initTemperature * (1/state.epochsComplete)

		state.tunnelingField = state.initTunnelingField * (1/(state.epochsComplete))

		# Compute normalized value.
		state.normTunnelingField = (state.tunnelingField/state.initTunnelingField)

		# Compute normalized value.
		state.normTemperature = state.temperature/state.initTemperature

	end

	function updateState_fsa_ra(state::State)

		# Iterate epoch count.
		state.epochsComplete += 1

		# Decay the temperature by 1, stopping at 0.
		state.temperature = state.initTemperature * (1/state.epochsComplete)

    if ((state.epochsComplete % 200000) == 0)
        state.temperature = state.initTemperature
    end

		state.tunnelingField = state.initTunnelingField * (1/(state.epochsComplete))

		# Compute normalized value.
		state.normTunnelingField = (state.tunnelingField/state.initTunnelingField)

		# Compute normalized value.
		state.normTemperature = state.temperature/state.initTemperature

	end

	function updateState_fsa_anisotropicity(state::State)

		# Iterate epoch count.
		state.epochsComplete += 1

		# Decay the temperature by 1, stopping at 0.
		state.temperature = state.initTemperature * (1/state.epochsComplete)

		state.tunnelingField = state.initTunnelingField * (1/(state.epochsComplete))

		# Compute normalized value.
		state.normTunnelingField = (state.tunnelingField/state.initTunnelingField)

		# Compute normalized value.
		state.normTemperature = state.temperature/state.initTemperature

		state.anisotropicField = 1-(sin(((1/10)*state.epochsComplete )+2*rand())+1)/2

	end

	function updateState_classicalOnly_unit_cooling(state::State)

		# Iterate epoch count.
		state.epochsComplete += 1

		# Decay the temperature by 1, stopping at 0.
		state.temperature = maximum([state.temperature-1, 0])

		state.tunnelingField = 0

		# Compute normalized value.
		state.normTemperature = state.temperature/state.initTemperature

	end

	function updateState_classicalOnly_inverse_cooling(state::State)

		# Iterate epoch count.
		state.epochsComplete += 1

		# Decay the temperature by 1, stopping at 0.
		state.temperature = state.initTemperature*(1/state.epochsComplete)

		state.tunnelingField = 0

		# Compute normalized value.
		state.normTemperature = (1/state.epochsComplete)

	end

	function updateState_gsa(state::State)

		# Iterate epoch count.
		state.epochsComplete += 1


    q_v = 3

		# Decay the temperature by 1, stopping at 0.
		state.temperature = state.initTemperature*(((2^(q_v-1))-1)/(((1+state.epochsComplete)^(q_v-1))-1))

		state.tunnelingField = 0

		# Compute normalized value.
		state.normTemperature = state.temperature/state.initTemperature

	end


	function updateState_quantumOnly_unit_cooling(state::State)

		# Iterate epoch count.
		state.epochsComplete += 1

		# Set the temperature to 0.
		state.temperature = 0

		# Decay the temperature by 1, stopping at 0.
		state.tunnelingField = maximum([state.tunnelingField-1, 0])

		# Compute normalized value.
		state.normTunnelingField = state.tunnelingField/state.initTunnelingField

	end

	function updateState_oscillatory(state::State)

		# Iterate epoch count.
		state.epochsComplete += 1

		# Decay the temperature by 1, stopping at 0.
		state.temperature = maximum([state.temperature-1, 0])

		# Set the strength of the tunneling field.
		state.tunnelingField = state.initTunnelingField*(sin(((1/1000)*state.epochsComplete )+3*rand())+1)/2
		state.temperature = state.initTemperature*(sin(((1/1000)*state.epochsComplete )+3*rand())+1)/2
		state.anisotropicField = 1-(sin(((1/100)*state.epochsComplete )+3*rand())+1)/2


		# Compute normalized value.
		state.normTunnelingField = (state.tunnelingField/state.initTunnelingField)

		# Compute normalized value.
		state.normTemperature = (state.temperature/state.initTemperature)

	end

	function updateState_quantum_only(state::State)

		# Iterate epoch count.
		state.epochsComplete += 1


		state.tunnelingField = 1
		state.temperature = 0


	end

end






# @everywhere function updateState_quantum_custom_reheat(stateTupleIn)

#     #  Unpack the input tuple. For readability.
#     (temperature, initTemperature,  learnRate, tunnelingField, maxConfigDist, epochsCool, numEpochs, anisotropicField) = stateTupleIn
# 	# Iterate epoch count.
# 	numEpochs += 1

#     # Decay the temperature by 1, stopping at 0.
#     temperature = maximum([temperature-1, 0])

#     # Add to the count of cool epochs, if cool.
#     epochsCool += int(temperature==0)

# 	period = int(1000/initTemperature)

# 	convergenceCoef = 0.2
#     # Set the strength of the tunneling field.
# 	if (epochsCool>(((period-1)*convergenceCoef)*initTemperature))
# 		tunnelingField = 0.1
# 	end

# # 	if (epochsCool>(4*initTemperature))
# # 		tunnelingField = 0
# # 		learnRate = learnRate
# # 	end

# 	if (epochsCool>((period-1)*initTemperature))
# 		tunnelingField = 0
# 		epochsCool = 0
# 		temperature = initTemperature # (10*(freq+1)*initTemperature/numEpochs)
# 	end


#     # Pack and return the state tuple.
#     return([temperature, initTemperature,  learnRate, tunnelingField, maxConfigDist, epochsCool, numEpochs, anisotropicField])

# end

# @everywhere function updateState_quantum_only(stateTupleIn)

#     #  Unpack the input tuple. For readability.
#     (temperature, initTemperature,  learnRate, tunnelingField, maxConfigDist, epochsCool, numEpochs, anisotropicField) = stateTupleIn

# 	# Iterate epoch count.
# 	numEpochs += 1


# 	tunnelingField = 1
# 	temperature=0

#     # Pack and return the state tuple.
#     return([temperature, initTemperature,  learnRate, tunnelingField, maxConfigDist, epochsCool, numEpochs, anisotropicField])

# end

