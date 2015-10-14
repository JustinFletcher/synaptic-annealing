
function sigmoidTemperatureSchedule(numEpochs)

    k=0.9
    x0=12.5
    return(1-(1./(1+exp(-k .* (20*((1:numEpochs)./numEpochs).-x0)    ))))
end

function linearTemperatureSchedule_eightTenths(numEpochs)
    lineValues = (1-(1:numEpochs)./(0.8*numEpochs))
    return( lineValues .- (lineValues .* int(lineValues.<0)) )
end

function linearTemperatureSchedule_sixTenths(numEpochs)
    lineValues = (1-(1:numEpochs)./(0.6*numEpochs))
    return( lineValues .- (lineValues .* int(lineValues.<0)) )
end

#plot(sigmoidTemperatureSchedule(1000))
#plot(linearTemperatureSchedule_sixTenths(1000))
