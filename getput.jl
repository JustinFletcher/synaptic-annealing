function putdata(data, idStr)
    # This function is a convienience interface to the serialize package.
    out = open("data\\"*idStr,"w")
    serialize(out,data);
    close(out)
end

function getdata(idStr)
    # This function is a convienience interface to the serialize package.
    in = open("data\\"*idStr,"r")
    data = deserialize(in)
    close(in)
    return(data)
end
