using PyPlot

function generateErrorSurface(f, sampleRange, alphaRange, betaRange,correctParamAlpha,correctParamBeta)

  errorSurface = ones(length(alphaRange), length(betaRange))


  alphaCount = 0
  for alpha in alphaRange
    alphaCount+=1
    betaCount = 0

    for beta in betaRange
      betaCount+=1
      errorSurface[alphaCount, betaCount] = mean((f(sampleRange,alpha,beta) .- f(sampleRange,correctParamAlpha,correctParamBeta)).^2)
    end
  end

  return(errorSurface)
end

start = 0.1
res = 0.05
stop = 10
sampleRange = [0.01:0.01:2.5]
alphaRange = [start:res:stop]
betaRange = [start:res:stop]

correctParamK = stop*rand()
correctParamL = stop*rand()

correctParamK = 5
correctParamL = 5

function customFun(x, a, b)
  a = 2.*(-exp(-(sin(a)+sin(b))).*x)
  b = (-exp(-(sin(a+5)+sin(b+5))).*x)./4
  c = (-exp(-(cos(a+5)+cos(b+5))).*x)./b
  d = (-exp(-(cos(a+5)+sin(b+5))).*x).^2
#   e = 10*rand(length(x)).*((cos(2*pi*(a/1000))+1)/2 + (cos(2*pi*(b/1000))+1)/2)/2
  std = 2.5
#   e = (1-(exp(-((((a-5).^2)/((2.*std)))+(((b-5).^2)/(2.*std))))))

  e = rand().*1.*(((tanh(a+5).+1)./2)+((tanh(b+5).+1)./2))
  return((a./3)+(b./3)+(c./3)+(d./3).*e)
end

errorSurfaceArray = @time generateErrorSurface(customFun, sampleRange, alphaRange, betaRange, correctParamK, correctParamL)
mesh(alphaRange, betaRange, errorSurfaceArray,cmap=ColorMap("gray"))


ion()
show()


sampleRange = [start:res:stop]
transformedVec = 5.^(sampleRange)
plot(sampleRange, transformedVec)

