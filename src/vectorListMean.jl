
function vectorListMean(vectorList)

    numVectors = length(vectorList)

    maxLength = maximum([length(vector) for vector in vectorList])

    accumVec = zeros(maxLength )

    for vector in vectorList
        temp = (ones(maxLength)*vector[end])
        temp[1:length(vector)] = vector
        accumVec+=temp
    end

    return(accumVec/numVectors)
end
