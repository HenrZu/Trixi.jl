using Plots, Colors
using Trixi

#Berechne die Gewichte fÃ¼r gegebene StÃ¼tzstellen x_j (1b)
function gewichte(x_j)
    N = length(x_j)
    w = zeros(N)
    for i = 1:N
        temp = 1
        # Berechne bis Punkt i-1
        for j = 1:i-1
            temp *= x_j[i] - x_j[j]
        end
        # Ab Punkt i+1 bis n
        for j = i+1:N
            temp *= x_j[i] - x_j[j]
        end
        w[i] = 1 / temp
    end
    return w
end

#Berechne den interpolierten  Funktionsswert an Stelle x für Fkt-Werte f
# und StÃ¼tzstelle x_j
function lagrange(x_j,f,x)
    N = length(f);
    # Platzhalter für Summen im Nenner und Zähler
    sum1=sum2=0;
    w = gewichte(x_j); # Berechne Gewichte

    for j =1:N
        xdif = x - x_j[j]
        # teste, ob x=x_j
        if (xdif==0)
            return f[j]
        end

        temp = w[j] / xdif

        sum1 += f[j]*temp
        sum2 += temp
    end
    return sum1/sum2
end


function gauss_lobatto(N; tol=1e-15)
    x = -cos.(LinRange(0, pi, N))
    P = zeros(N, N)  # Vandermonde Matrix
    x_old = 2
    while maximum(abs.(x - x_old)) > tol
        x_old = x
        P[:, 1] = 1
        P[:, 2] = x
        for k = 3:N
            P[:, k] = ((2 * k - 1) * x .* P[:, k - 1] -
                       (k - 1) * P[:, k - 2]) / k
        end
        x = x_old - (x .* P[:, N] - P[:, N - 1]) ./ (N* P[:, N])
    end
    return x
end


runge(x) =  1 ./(1 .+ 25*x.^2)
x = LinRange(-1, 1, 100)
x_int = LinRange(-1, 1, 11)
x_int2 = LobattoLegendreBasis(11)
x_int2 = x_int2.nodes
x_new = LinRange(-1, 1, 1000)

y_new = zeros(length(x_new))
for i in 1:length(x_new)
    y_new[i] = lagrange(x_int, runge(x_int), x_new[i])
end

y_new2 = zeros(length(x_new))
for i in 1:length(x_new)
    y_new2[i] = lagrange(x_int2, runge(x_int2), x_new[i])
end

b = plot(x, runge(x), label = "Exact Solution", color=colorant"red", lw = 3, line=(:dot,6), legendfontsize= 13, legend = :outertop)
plot!(x_new, y_new, label = "Using Equidistant nodes", color=colorant"blue", lw = 2) #, legendfontsize= 12, legend = :outertop)
plot!(x_new, y_new2, label = "Using LGL nodes", color=colorant"#3F383C", lw = 3) #, legendfontsize= 12, legend = :outertop)
#scatter!(x_int, runge(x_int), color=colorant"black", label ="Equidistant nodes", markersize=6)
#scatter!(x_int2, runge(x_int2), color=colorant"black", label ="LGL nodes" , markersize=6)
xlabel!("x")
ylabel!("y")


savefig(b,"C:\\Users\\Henri\\Pictures\\ausgabe_julia\\runge_sum.pdf")