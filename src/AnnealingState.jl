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
	numEpochs

	# Field variables
	learnRate
	temperature
	anisotropicField
	tunnelingField

	# Reporting variables
	normTunnelingField
	normTemperature




	function State(initTemperature, initLearnRate, initTunnelingField, maxConfigDist)

		temperature = initTemperature
		learnRate = initLearnRate
		tunnelingField = initTunnelingField
		epochsCool = 0
		numEpochs = 0
		anisotropicField = 0

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
			 epochsCool, numEpochs, learnRate, temperature,
			 anisotropicField, tunnelingField, normTunnelingField, normTemperature)
	end


end
end
