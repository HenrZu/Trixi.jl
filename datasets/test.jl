using HDF5: h5open



n_dims = 1
datatyp = "NNPP"

Y_train = h5open("datasets\\valid_sedov.h5", "r") do file
    read(file, "Y")
end

X_train = h5open("datasets\\valid_sedov.h5", "r") do file
    read(file, "X")
end

Y_train2 = h5open("datasets//valid_sedov_lax.h5", "r") do file
    read(file, "Y")
end

X_train2 = h5open("datasets//valid_sedov_lax.h5", "r") do file
    read(file, "X")
end

Y_trixi = h5open("datasets//valid_weak.h5", "r") do file
    read(file, "Y")
end

X_trixi = h5open("datasets//valid_weak.h5", "r") do file
    read(file, "X")
end

train_length = Int(length(X_train) / 2)
train2_length = Int(length(X_train2) / 2)
trixi_length = Int(length(X_trixi) / 2)

X_train = reshape(X_train, 2, train_length)
X_train2 = reshape(X_train2, 2, train2_length)
X_trixi = reshape(X_trixi, 2, trixi_length)
  
train_combine_x = hcat(X_train, X_train2, X_trixi)
train_combine_y = vcat(Y_train, Y_train2, Y_trixi)

lengthx = length(train_combine_x)
lengthy = length(train_combine_y)
println("SIze x = $lengthx")
println("SIze y = $lengthy")

h5open("datasets/validdata1d_Limiter_NNPP.h5", "w") do file
    write(file, "X", train_combine_x)
    write(file, "Y", train_combine_y)
end


