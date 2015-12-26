
function plotCompleteRate(v, titleStr)


  plot(v)
#   xlim(0,500000)
  ylim(0,1)

	grid("on")

  xlabel("Training Epoch")
  ylabel("Fraction of Simulations with Perfect Classification")
  title(titleStr)
end



function plotGammaDistPDF(range, shape, scale, labelStr)

  x = range
  plot(x, (x.^(shape-1)).*(e.^(-x./scale))./((scale^shape).*(gamma(shape))), label=labelStr, alpha=0.7)

  ticklabel_format(style="sci", axis="y", scilimits=(0,0))
	legend(loc=1)
#   plt[:show]()

  xlabel("Training Epoch")
  ylabel("Probability of Achieving Perfect Classification")
  title("Gamma Distribution Approximiation of \nthe Probability of Achieving Perfect Classfication")

end

function plotGammaDistPDFfromVector(v, labelStr)


  (toptExpVal,toptStd) = calcPerfectClassStats(v)

  (shape,scale) =calculateGammaDistShapeScale(toptExpVal, toptStd^2)

  plotGammaDistPDF(0:length(v), shape, scale, labelStr)

end


function plotGaussian(v, normTo=1)

  evalRange = 0:length(v)
  toptExpVal = int(stepExpectationValue((v./normTo)))
  toptStd =int(stepStd(v./normTo))

  plot(evalRange, gaussianPDF(evalRange ,toptExpVal,toptStd))

end

function plotAnnealResults(meanTrainErrorVec, meanValErrorVec, reportFrequency ,titleStr)
   	plot(reportFrequency.*(0:length(meanTrainErrorVec)-1), meanTrainErrorVec, label="Training Classification Error", alpha=0.7)
    plot(reportFrequency.*(0:length(meanValErrorVec)-1), meanValErrorVec, label="Validation Classification Error", alpha=0.7)
    ylim(0, 0.1)
#     xlim(0, 500000)

	if(minimum(meanValErrorVec)<0 || maximum(meanValErrorVec)>1)

		ylim(minimum(meanValErrorVec), maximum(meanValErrorVec))
	end

	ax = gca()
  xlabel("Training Epoch")
  ylabel("25-Fold Cross-Validated Mean Classification Error")
	legend(loc=1)
	grid("on")
  title(titleStr)

#   plt[:show]()
end


function plotCmaAnnealingResults(meanValErrorVec, windowSize, reportFrequency, titleStr)
    plot(reportFrequency.*(0:length(meanValErrorVec)-1), movingWindowAverage(meanValErrorVec,  windowSize), label="Validation Classification Error", alpha=0.7)
    ylim(0, 0.1)
#     xlim(0, 500000)

	if(minimum(meanValErrorVec)<0 || maximum(meanValErrorVec)>1)

		ylim(minimum(meanValErrorVec), maximum(meanValErrorVec))
	end

	ax = gca()
  xlabel("Training Epoch")
  ylabel("25-Fold Cross-Validated Mean Classification Error")
	legend(loc=1)
	grid("on")
  title(titleStr)

#   plt[:show]()
end

