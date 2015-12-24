function plotAnnealResults(meanTrainErrorVec, meanValErrorVec, reportFrequency ,titleStr,)
   	plot(reportFrequency.*(0:length(meanTrainErrorVec)-1), meanTrainErrorVec, label="Training Classification Error", alpha=0.7)
    plot(reportFrequency.*(0:length(meanValErrorVec)-1), meanValErrorVec, label="Validation Classification Error", alpha=0.7)
    ylim(0, 0.1)
    xlim(0, 500000)

	if(minimum(meanValErrorVec)<0 || maximum(meanValErrorVec)>1)

		ylim(minimum(meanValErrorVec), maximum(meanValErrorVec))
	end

	ax = gca()
	reportFrequency
    xlabel("Training Epoch")
    ylabel("25-Fold Cross-Validated Mean Classification Error")
	legend(loc=1)
	grid("on")
    title(titleStr)

    plt[:show]()
end

