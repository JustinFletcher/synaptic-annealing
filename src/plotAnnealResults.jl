function plotAnnealResults(meanTrainErrorVec, meanValErrorVec, titleStr)
    plot(meanTrainErrorVec, label="Training Classification Error", alpha=0.7)
    plot(meanValErrorVec, label="Validation Classification Error", alpha=0.7)
    ylim(0, 1)

	if(minimum(meanValErrorVec)<0 || maximum(meanValErrorVec)>1)

		ylim(minimum(meanTrainErrorVec), maximum(meanValErrorVec))
	end
    xlabel("Training Epoch")
    ylabel("5-Fold Cross-Validated Mean Classification Error")
	legend(loc=1)
    title(titleStr)
    plt[:show]()
end

