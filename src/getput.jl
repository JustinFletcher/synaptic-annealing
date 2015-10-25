function putdata(data, idStr)
    # This function is a convienience interface to the serialize package.
    out = open(homedir()*"\\OneDrive\\afit\\rs\\synaptic-annealing\\cache\\"*idStr,"w")
    serialize(out,data);
    close(out)
end

function getdata(idStr)
    # This function is a convienience interface to the serialize package.
    in = open(homedir()*"\\OneDrive\\afit\\rs\\synaptic-annealing\\cache\\"*idStr,"r")
    data = deserialize(in)
    close(in)
    return(data)
end
