function inverse(x)
  return(1./(x))
end

function constructInverseDataset(inverseDataRange, trainToTestRatio)

    # Construct the data.
    inverseData = [inverseDataRange inverse(inverseDataRange)]

    # Randomly partition the data set into test and train subsets. Start by shuffling the data.
    randIndVec = sample(1:size(inverseData)[1], size(inverseData)[1])
    inverseData = inverseData[randIndVec, :]

    # Now select a subset of the data to be train and test.
    inverseDataTrain = inverseData[1:int64(floor(size(inverseData)[1]*trainToTestRatio)), :]
    inverseDataTest = inverseData[int64(floor(size(inverseData)[1]*trainToTestRatio))+1:end, :]

    return(inverseDataTrain,inverseDataTest)

end