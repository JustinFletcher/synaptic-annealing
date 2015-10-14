
addprocs(int(CPU_CORES*0.85))
rmprocs()
nworkers()
workers()
test = rand(200)
nheads = @time @sync @parallel (+) for i=1:2000000
    randbool()
end

nheads = @schedule @time for i=1:10
    print("hi")
end

nheads = @schedule @time for i=1:10
    print("hi")
end

nheads = @schedule @time for i=1:10
    print("hi")
end

nheads = @schedule @time for i=1:10
    print("hi")
end

nheads = 0

temp = 0
nheads = @schedule @time for i=1:20000000
    temp += randbool()
end

@parallel (+) for t = 1:1000
	getDataClassErr(createSynapseMatrix([5,10,3]), tanh, irisDataClassed, [1:4], [5:7])
end



ref = fetch(@spawn(for i=1:20000000 randbool() end))

fetch(ref)

@everywhere function someVec(n)
    return(rand(n^2))
end

storageList = Any[]
@time for i = 1:10
    vecTemp = someVec(i)
    push!(storageList, vecTemp)
end
storageList

addprocs(10)

ref = @spawn someVec(1)
fetch(ref)

storageList = Any[]
@time for i = 1:10
    vecRef = @spawn someVec(i)
    push!(storageList, vecRef)
end

storageList

