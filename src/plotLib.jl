
function plotCompleteRate(v, titleStr, dataSetName, xmax, ymax)


    plotHandle = plot(v)
    xlim(0,xmax)
    ylim(0,ymax)

    grid("on")

    xlabel("Training Epoch")
    ylabel("Fraction of Simulations with Perfect Classification")
    title(titleStr)


    return(plotHandle)
end



function plotGammaDistPDF(range, shape, scale, labelStr, xmax)

    x = range
    plotHandle = plot(x, (x.^(shape-1)).*(e.^(-x./scale))./((scale^shape).*(gamma(shape))), label=labelStr, alpha=0.7)

    ticklabel_format(style="sci", axis="y", scilimits=(0,0))
    legend(loc=1)
  #   plt[:show]()

    xlim(0, xmax)
    xlabel("Training Epoch")
    ylabel("Probability of Achieving Perfect Classification")
    title("Gamma Distribution Approximiation of \nthe Probability of Achieving Perfect Classfication")
    grid("on")


    return(plotHandle)

end

function plotGammaDistPDFfromVector(v, labelStr, xmax)


    (toptExpVal,toptStd) = calcPerfectClassStats(v)

    (shape,scale) =calculateGammaDistShapeScale(toptExpVal, toptStd^2)

    plotHandle = plotGammaDistPDF(0:length(v), shape, scale, labelStr, xmax)

    return(plotHandle)
end


function plotGaussian(v, normTo=1)

    evalRange = 0:length(v)
    toptExpVal = int(stepExpectationValue((v./normTo)))
    toptStd =int(stepStd(v./normTo))

    plot(evalRange, gaussianPDF(evalRange ,toptExpVal,toptStd))

end

function plotAnnealResults(meanTrainErrorVec, meanValErrorVec, reportFrequency, titleStr, dataSetName, xmax, ymax)
   	plotHandle = plot(reportFrequency.*(0:length(meanTrainErrorVec)-1), meanTrainErrorVec, label="Training Error - "*dataSetName, alpha=0.7)
    plotHandle = plot(reportFrequency.*(0:length(meanValErrorVec)-1), meanValErrorVec, label="Validation Error- "*dataSetName, alpha=0.7)
    xlim(0,xmax)
    ylim(0,ymax)

    if(minimum(meanValErrorVec)<0 || maximum(meanValErrorVec)>1)

      ylim(minimum(meanValErrorVec), maximum(meanValErrorVec))
    end

    ax = gca()
    xlabel("Training Epoch")
    ylabel("25-Fold Cross-Validated Mean Classification Error")
    legend(loc=1)
    grid("on")
    title(titleStr)


    return(plotHandle)
end


function plotCentralMovingAverageValError(meanValErrorVec, stdValErrorVec, windowSize, reportFrequency, titleStr, dataSetName, xmax, ymax, linecolor)

    x = reportFrequency.*(0:length(meanValErrorVec)-1)
    y = movingWindowAverage(meanValErrorVec,  windowSize)
    ystd = movingWindowAverage(stdValErrorVec,  windowSize)/2


    plotHandle = plot(x, y,label="Validation Error - "*dataSetName, alpha=0.7, color=linecolor)
    plotHandle = errorbar(x,y,yerr=ystd,fmt=".", alpha=0.7, color=linecolor)
    xlim(0,xmax)
    ylim(0,ymax)

    if(minimum(meanValErrorVec)<0 || maximum(meanValErrorVec)>1)
      ylim(minimum(meanValErrorVec), maximum(meanValErrorVec))
    end

    ax = gca()
    xlabel("Training Epoch")
    ylabel("25-Fold Cross-Validated Mean Classification Error")
    legend(loc=1)
    grid("on")
    title(titleStr)

    return(plotHandle)
end

