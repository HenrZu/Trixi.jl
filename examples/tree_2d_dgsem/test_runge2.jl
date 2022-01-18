using Plots

function LagrangeInterp1D( fvals, xnodes, barw, t )
    numt = 0
    denomt = 0

    for j = 1 : length( xnodes )
        tdiff = t - xnodes[j]
        numt = numt + barw[j] / tdiff * fvals[j]
        denomt = denomt + barw[j] / tdiff

        if ( abs(tdiff) < 1e-15 )
            numt = fvals[j]
            denomt = 1.0
            break
        end
    end

    return numt / denomt

end

f(x) = 1/(1 + 16*x^2)

# Sampling
n = 15;
xnodes = EquispacedNodes(n);
w = EquispacedBarWeights(n);
f_sample = f.(xnodes);

# Interpolation
t = LinRange(-1,1,100)
f_interp = [LagrangeInterp1D( f_sample, xnodes, w, t[j] ) for j=1:length(t)]
plot(t,f.(t), label="Exact", marker = 3)
plot!(t,f_interp, label="Interp", marker = 3)