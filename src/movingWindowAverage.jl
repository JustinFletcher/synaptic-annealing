function movingWindowAverage(inputVector, windowSize)

    movingAverageWindowVector = Float64[]

    for (windowMidIndex = 1:length(inputVector))

        windowStartIndex = maximum([minimum([windowMidIndex-floor(windowSize/2), (length(inputVector)-windowSize)]), 1])

        windowEndIndex = minimum([length(inputVector),(windowStartIndex+windowSize)])


#         println("---------")
#         println(windowStartIndex)
#         println(windowEndIndex)
#         println(windowEndIndex-windowStartIndex)



        push!(movingAverageWindowVector, mean(inputVector[windowStartIndex:windowEndIndex]))

    end

  return(movingAverageWindowVector)

end

# trialdata = sin(0.1:0.1:50)+rand(length(0.1:0.1:50)).^2

# plot(trialdata)
# plot(movingWindowAverage(trialdata, 15))

