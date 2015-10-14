

function vectorListToMatrix(vectorList)

    maxLength = maximum([length(vector) for vector in vectorList])

    numVectors = length(vectorList)

    outMat = zeros(numVectors, maxLength)

    matRow = 0

    for vector in vectorList
        temp = (ones(maxLength)*vector[end])
        temp[1:length(vector)] = vector

        matRow += 1
        outMat[matRow, :] = temp
    end

    return(outMat)
end