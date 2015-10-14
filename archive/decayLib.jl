function decayOnePercent(input)
    return(0.99*input)
end

function decayUnityZeroBound(input)
    return(maximum([input-1, 0]))
end

function expDecay(x)
    return(0.1+(0.5*exp(-(0.1*x))))
end

function constVal(x)
    return(10)
end

function sigmoidDecay_Tuned(epoch)

    k=3
    x0=8
    decayVal = (1- 0.99*(1./(1+exp(-k .* (10.*(epoch)-x0)))) )
    return(decayVal)
end

#plot(sigmoidDecay_Tuned(0:0.001:1))
