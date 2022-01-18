using Plots

x = collect(-3:3)


function relu(x)
    y = max(0,x)
    return y
end


function leaky_relu(x,β)
    y = max(β * x,x)
    return y
end

function sigmoid(x)
    y = 1.0 / (1.0 + exp(-x))
    return y
end
 a = plot(x,[relu.(x),leaky_relu.(x,0.1), sigmoid.(x)], label = ["ReLU" "Leaky ReLU" "Softmax"], linestyle=:auto, framestyle = :origin ,lw = 3, legend=:bottomright ) # label = ["ReLU"])
 xlabel!("z")
 ylabel!("σ(z)")
# plot!(x,leaky_relu.(x,0.2), label = ["Line 2"])