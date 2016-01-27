
using PyPlot

@everywhere cd("\\fletcher-thesis")

pwd()
@everywhere include("$(pwd())\\src\\"*"getput.jl")

function sgsa(qv, t, r)

    return(1./((1+(((qv-1).*(r.^2))./(t^(2/(qv-3))))).^((1/(qv-1))-(0.5))))

end
inputRangeRes = 0.01

inputLimit = 1000

inputRange = -inputLimit:inputRangeRes:inputLimit

temperatureRange = [t for t in [1,2.5,5,10]]

qvRange = [1.25 ,1.75, 2, 2.25]

qvTempSurface = zeros((length(temperatureRange), length(inputRange)))

function generateSGSAData (inputRangeRes, inputLimit, inputRange, temperatureRange)
  retvaltemp = []
  qvTempSurface = zeros((length(temperatureRange), length(inputRange)))
  qvIndex = 0
  for qv in qvRange

    qvIndex+=1

    tIndex = 0
    for t in temperatureRange
      qvTempSurface[tIndex+=1,:] = sgsa(qv, t, inputRange)
    end


    subplot(2, length(qvRange), qvIndex)

    title("Qv = " *string(qv))
    #imshow(qvTempSurface,origin="lower")
    #contour(qvTempSurface)

    normalizedCauchyDist = (1./(pi.*(1.+inputRange.^2)))
    plot(inputRange, normalizedCauchyDist, label="Cauchy", ls="-.")


    tIndex = 0
    for t in temperatureRange

      normConstant = (1/(sum(vec(qvTempSurface[tIndex+=1,:]))*inputRangeRes))
      normalizedIsotempDist = vec(qvTempSurface[tIndex,:]).*normConstant
      plot(inputRange, normalizedIsotempDist, label="Tq= "*string(t))

    end

    xlim(-10,10)
    legend(loc=1)


    subplot(2, length(qvRange), length(qvRange)+qvIndex)

    plot(inputRange, normalizedCauchyDist, label="Cauchy", ls="-.")
    tIndex = 0
    for t in temperatureRange
      # Numerically compute the integral approximation and scale the integral to 1.
      normConstant = (1/(sum(vec(qvTempSurface[tIndex+=1,:]))*inputRangeRes))
      retvaltemp = normalizedIsotempDist = vec(qvTempSurface[tIndex,:]).*normConstant
      plot(inputRange, normalizedIsotempDist, label="Temperature = "*string(t))
    end

    xlim(-inputLimit,inputLimit)
    ylim(0,0.001)

  end

  return(retvaltemp)
end

normalizedIsotempDist =  generateSGSAData (inputRangeRes, inputLimit, inputRange, temperatureRange)


@time randSGSA(100, 0.1, 2, 1, 0).*[normalizedIsotempDist[rand(1:end)] for x=1:50, y=1:50, z=1:2]

qvIndex = 0
for qv in qvRange

  qvIndex+=1

  # Generate the SGSA data.
  tIndex = 0
  for t in temperatureRange
    qvTempSurface[tIndex+=1,:] = sgsa(qv, t, inputRange)
  end

  # Invoke a subplot, top row.
  subplot(2, length(qvRange), qvIndex)

  #imshow(qvTempSurface,origin="lower")
  #contour(qvTempSurface)

  normalizedCauchyDist = (1./(pi.*(1.+inputRange.^2)))

  plot(inputRange, normalizedCauchyDist, label="Cauchy", ls="-.")

  tIndex = 0
  for t in temperatureRange

    normConstant = (1/(sum(vec(qvTempSurface[tIndex+=1,:]))*inputRangeRes))

    normalizedIsotempDist = vec(qvTempSurface[tIndex,:]).*normConstant

    plot(inputRange, normalizedIsotempDist, label="Tq= "*string(t))

  end

  xlim(-10,10)
  legend(loc=1)
  title("Qv = " *string(qv))

  # Begin the second row of plots.
  subplot(2, length(qvRange), length(qvRange)+qvIndex)

  tIndex = 0
  for t in temperatureRange
    tIndex+=1

    # Numerically compute the integral approximation and scale the integral to 1.
    normConstant = (1/(sum(vec(qvTempSurface[tIndex,:]))*inputRangeRes))

    normalizedIsotempDist = vec(qvTempSurface[tIndex,:]).*normConstant

    println(tIndex)

    sumHolder = 0

    gsaCFD = [sumHolder+=normalizedIsotempDist[index] for index in 1:length(normalizedIsotempDist)]

    plot(inputRange, gsaCFD, label="Temperature = "*string(t))

  end

  xlim(-inputLimit,inputLimit)
  #ylim(0,1)

end


function gsa(qv, t, d, inputRange)

end

function computeSGSACDF(displacmentLimit, resolution, qv, tqv, d)

    inputRange = -displacmentLimit:resolution:displacmentLimit
    # Invoke Tsallis Eq. 9 to generate a GSA distribution.
#     gsaUnnormalizedDist = gsa(qv, t, d, inputRange)
    gsaUnnormalizedDist = sgsa(qv, tqv, inputRange)

    # Numerically compute the integral of the GSA distribution.
    normConstant = (1/(sum(gsaUnnormalizedDist)*resolution))

    gsaNormalizedDistribution = gsaUnnormalizedDist.*normConstant

    sumHolder = 0

    gsaCFD = [sumHolder+=gsaNormalizedDistribution[index] for index in 1:length(gsaNormalizedDistribution)]

    scaledGSACDF = gsaCFD.*resolution

    return(Any[inputRange, scaledGSACDF])
end

function computeGSACDF(displacmentLimit, resolution, qv, tqv, d)

    inputRange = -displacmentLimit:resolution:displacmentLimit
    # Invoke Tsallis Eq. 9 to generate a GSA distribution.
#     gsaUnnormalizedDist = gsa(qv, t, d, inputRange)
    gsaUnnormalizedDist = gsa(qv, tqv, inputRange)

    # Numerically compute the integral of the GSA distribution.
    normConstant = (1/(sum(gsaUnnormalizedDist)*resolution))

    gsaNormalizedDistribution = gsaUnnormalizedDist.*normConstant

    sumHolder = 0

    gsaCFD = [sumHolder+=gsaNormalizedDistribution[index] for index in 1:length(gsaNormalizedDistribution)]

    scaledGSACDF = gsaCFD.*resolution

    return(Any[inputRange, scaledGSACDF])
end

function randomSampleFromCDF(cdfObject)
    cdfVector = cdfObject[2]
    domainVector = cdfObject[1]
    return(domainVector[findmin(abs(cdfVector-rand()))[2]])
end

function randSGSA(displacmentLimit, resolution, qv, tqv, d)
    return(randomSampleFromCDF(computeSGSACDF(displacmentLimit, resolution, qv, tqv, d)))
end

function randGSA(displacmentLimit, resolution, qv, tqv, d)
    return(randomSampleFromCDF(computeGSACDF(displacmentLimit, resolution, qv, tqv, d)))
end

function randMatGSA(xSize,ySize,zSize, displacmentLimit, resolution, qv, tqv, d)
    return([randSGSA(displacmentLimit, resolution, qv, tqv, d) for x=1:xSize, y=1:ySize, z=1:zSize])
end

gsaCDF = @time computeSGSACDF(100, 0.01, 2, 1, 0)

plot(gsaCDF[1], gsaCDF[2])

@time randSGSA(100, 0.1, 2, 1, 0)

@time plt[:hist]([randSGSA(10, 0.1, 1.1, 1, 0) for foo in 1:1000], 100)
# If I generate a billion samples from the population, won't the underlying distribution be statistically likely to be represented in this new subpopulation?
@time randn()

gsaMat = @time randMatGSA(50, 50, 2, 5, 0.1, 2, 1, 0)


displacmentLimit = 100
resolution = 0.01
qv = 2.5
tqv = 1
d = 0

## proto function to gnerate sample from stored data.
nsamples = 50*50*3
nPopulationSamples = 10000
sgsaPopulationSample = @time [randSGSA(displacmentLimit, resolution, qv, tqv, d) for x=1:nPopulationSamples]
putdata(sgsaPopulationSample, "sgsaSample_dl100_res0p01_qv2p5_tqv1p0_d0")

sgsaPopulationSample = @time float64(getdata("sgsaSample_dl100_res0p01_qv2p5_tqv1p0_d0"))
sampleWindowStart = @time rand(1:(length(sgsaPopulationSample)-nsamples))
populationSample = @time sgsaPopulationSample[sampleWindowStart:sampleWindowStart+nsamples-1]


constructedSample = @time [randSGSA(displacmentLimit, resolution, qv, tqv, d) for foo in 1:nsamples]



populationSample = @time [sgsaPopulationSample[rand(1:end)] for foo in 1:nsamples]
tempRands = @time randn(nsamples)

# ~100 times slower..
# over a million runs...
(0.006082*1000000)/60/60
((7.9551e-5)*1000000)/60/60

@time plt[:hist](constructedSample, 100)
@time plt[:hist](populationSample, 100)


@everywhere include("$(pwd())\\src\\"*"testinclude.jl")

getMyVar()

