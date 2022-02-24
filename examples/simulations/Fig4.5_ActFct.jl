
x = collect(-3:3)


function relu(x)
    y = max(0,x)
    return y
end


function leaky_relu(x,β)
    y = max(β * x,x)
    return y
end

function softmax(x)
    y = 1.0 / (1.0 + exp(-x))
    return y
end


# to save Data (possible solution -  may exist better ones)
# using DelimitedFiles
# open("soft.txt", "w") do io
# 	writedlm(io, [x softmax(x)])
# end