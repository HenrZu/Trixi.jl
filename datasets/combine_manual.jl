using HDF5: h5open


X_t = h5open(joinpath("datasets","limiter_data.h5"), "r") do file
    read(file, "X")
end

Y_t = h5open(joinpath("datasets","limiter_data.h5"), "r") do file
    read(file, "Y")
end

X_sf = h5open(joinpath("datasets","traindata1dNNPP_noscale.h5"), "r") do file
    read(file, "X")
end

Y_sf = zeros(size(X_sf,2))

# scale Trixi Data

for i = 1:size(X_t,2)
    norm = max(maximum(abs.([X_t[1,i],X_t[1,i]])),1)
    X_t[1,index_output] = x1 /norm
    X_t[2,index_output] = x2 /norm
end

X = hcat(X_sf,X_t)
Y = vcat(Y_sf,Y_t)


# save data
h5open("datasets//limiter_unscale.h5", "w") do file
    write(file, "X", X)
    write(file, "Y", Y)
end