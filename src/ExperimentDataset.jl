
module ExperimentDataset


	export Dataset, shuffle

	include("$(pwd())\\src\\"*"normalizeData.jl")
	include("$(pwd())\\src\\"*"orthogonalizeDataClasses.jl")
	include("$(pwd())\\src\\"*"shuffleData.jl")
	include("$(pwd())\\src\\"*"removeDataMean.jl")

	type Dataset
		data
		inputCols
		outputCols
    name

		function Dataset(datapath::String, inputCols, outputCols, name)

			data = readdlm(datapath, ',' , Any)
			data = orthogonalizeDataClasses(data, outputCols)
			data = normalizeData(data)
			data = removeDataMean(data, inputCols)
			data = shuffleData(data)



			outputCols = [outputCols[1]:((outputCols[1]-1)+(size(data, 2)-length(inputCols)))]
			new (data, inputCols, outputCols, name)
		end

		function Dataset(data, inputCols, outputCols, name)

			data = normalizeData(data)
			data = removeDataMean(data, inputCols)
			data = shuffleData(data)


			outputCols = [outputCols[1]:((outputCols[1]-1)+(size(data, 2)-length(inputCols)))]
			new (data, inputCols, outputCols, name)

		end
	end


	function shuffle(dataset::Dataset)
    	dataset.data = (dataset.data[randperm(size(dataset.data)[1]), :])
	end

end
