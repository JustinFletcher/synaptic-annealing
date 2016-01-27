
@everywhere cd("\\fletcher-thesis")

@everywhere include("$(pwd())\\src\\"*"getput.jl")


function sgsa(qv, t, r)

    # This function returns the SGSA visiting distribution over the specified range, r.
    return(1./((1+(((qv-1).*(r.^2))./(t^(2/(qv-3))))).^((1/(qv-1))-(0.5))))

end


function gsa(qv, t, d, x)

    # This function returns the SGSA visiting distribution over the specified range, x.

end

function computeGSACDF(displacmentLimit, resolution, qv, tqv, d)

    # Construct the range of inputs that will be considered
    inputRange = -displacmentLimit:resolution:displacmentLimit

    # Invoke Tsallis Eq. 9 to generate a GSA distribution.
#     gsaUnnormalizedDist = gsa(qv, t, d, inputRange)
    gsaUnnormalizedDist = sgsa(qv, tqv, inputRange)

    # Numerically compute the integral of the GSA distribution.
    gsaUnnormalizedArea = ((sum(gsaUnnormalizedDist)*resolution))

    # Normalize the distribution.
    gsaNormalizedDistribution = gsaUnnormalizedDist./gsaUnnormalizedArea

    # Compute the CDF of the GSA visiting distribution through sequential summing.
    sumHolder = 0
    gsaCFD = [sumHolder+=gsaNormalizedDistribution[index] for index in 1:length(gsaNormalizedDistribution)]


    scaledGSACDF = gsaCFD.*resolution

    return(Any[inputRange, scaledGSACDF])
end


function randomSampleFromCDF(cdfObject)

    # Pull out the CDF values and the associated domain values. For code clarity.
    cdfVector = cdfObject[2]
    domainVector = cdfObject[1]

    # Return the domain value which is closest to a randomly-choosen CDF value.
    # This is numeric inverse transform sampling.
    return(domainVector[findmin(abs(cdfVector-rand()))[2]])
end

function buildSampleLibrary(displacmentLimit, resolution, qv, tqv, d, nDesiredSamples)

    cdfObject = computeGSACDF(displacmentLimit, resolution, qv, tqv, d)

    newSample = [[qv, tqv, d], [randomSampleFromCDF(cdfObject) for temp in 1:nDesiredSamples]]

    if isfile(".\\cache\\gsasamples\\gsasamples")

        sampleLib = getdata("gsasamples\\gsasamples")

    else

        sampleLib = Any[]
    end

    push!(sampleLib, newSample)

    putdata(sampleLib, "gsasamples\\gsasamples")

    return(sampleLib)
end


# rm(".\\cache\\gsasamples\\gsasamples")

function buildCompleteSampleLibrary (displacmentLimit, resolution, qvRange, tqvRange, dRange, nDesiredSamples)


    if isfile(".\\cache\\gsasamples\\gsasamples")

        rm(".\\cache\\gsasamples\\gsasamples")
    end

    for qv in qvRange
        for tqv in tqvRange
            for d in dRange
                buildSampleLibrary(displacmentLimit, resolution, qv, tqv, d, nDesiredSamples)
            end
        end
    end
end

# displacmentLimit = 50
# resolution = 0.001
# qvRange = [2.25]
# #tqvRange = [t for t in [1,2.5,5,10]]
# tqvRange = [t for t in [1]]
# dRange = [0]
# nDesiredSamples = 10000

# @time buildCompleteSampleLibrary(displacmentLimit, resolution, qvRange, tqvRange, dRange, nDesiredSamples)


# This code runs when this file is included.
gsaSampleLib = getdata("gsasamples\\gsasamples")

gsdSampleMatrix = gsaSampleLib[1]
for sampleRow in gsaSampleLib[2:end]
    gsdSampleMatrix = hcat(gsdSampleMatrix, sampleRow)
end

gsdSampleMatrix  = transpose(gsdSampleMatrix)


function randgsa(qv, tqv, d, sizeTuple=(1))

    # This function assumes that the matrix is sort in priority order by the first three columns.

    # Find the numeric sample set which most-closely matches the choose one.
    #closestIndex = @time findmin(sum((gsdSampleMatrix[:,1:3].-repmat([2.2 1 0], size(gsdSampleMatrix[:,1:3])[1]))))[2]

    i=1
    while (qv<gsdSampleMatrix[i,1])
        i+=1
    end

    while (tqv<gsdSampleMatrix[i,2])
        i+=1
    end

    while (d<gsdSampleMatrix[i,3])
        i+=1
    end

    #gsaSample = @time gsdSampleMatrix[2, 4:end]
#   println(i)
    # Compute the desried number of samples.
    nSamples =  prod(sizeTuple)
    #sampleWindowStart = rand(1:(length(gsdSampleMatrix[i, 4:end])-nSamples))
    sampleWindowStart = int(rand()*(length(gsdSampleMatrix[i, 4:end])-nSamples))
    populationSample = gsdSampleMatrix[i, [4+sampleWindowStart:3+sampleWindowStart+nSamples]]

    # Reshape the sample to the desired size.
    return(reshape(populationSample, sizeTuple))

end

# @time randgsa(2.25, 1, 1, (50,50,2))

# @time(rand((50,50,2)))

