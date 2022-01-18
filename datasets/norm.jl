
using HDF5: h5open

using Random
using DelimitedFiles

     # Load data
    x_train = h5open("datasets/limiter_data.h5", "r") do file
        read(file, "X")
    end
    y_train = h5open("datasets/limiter_data.h5", "r") do file
        read(file, "Y")
    end

    for i = 1:length(dataset_x1)
        norm = max(maximum(abs.([x1,x2])),1)
        # X[1,index_output] = x1 /norm
        # X[2,index_output] = x2 /norm
    end

    X = zeros(3, size(x_train,2))
    X









    # directly normalize
    norm = max(maximum(abs.([x1,x2,nnodes])),1)
    X[1,index_output] = x1 / norm
    X[2,index_output] = x2 / norm
    X[3,index_output] = nnodes / norm

    # x_valid = h5open("datasets/validdata1d_Limiter_NNPP.h5", "r") do file
    #     read(file, "X")
    # end
    # y_valid = h5open("datasets/validdata1d_Limiter_NNPP.h5", "r") do file
    #     read(file, "Y")
    # end

    # # Vector to Array
    # y_train = reshape(y_train, 1, length(y_train))
    # y_valid = reshape(y_valid, 1, length(y_valid))

    #  # scale data
    #  for i in 1:size(x_train,2)
    #     x_train[:,i]=x_train[:,i]./max(maximum(abs.(x_train[:,i])),1)
    # end

    # for i in 1:size(x_valid,2)
    #     x_valid[:,i]=x_valid[:,i]./max(maximum(abs.(x_valid[:,i])),1)
    # end

    # writedlm("x_train.csv", x_train, ", ")
    # writedlm("y_train.csv", y_train, ", ")

    # writedlm("x_valid.csv", x_valid, ", ")
    # writedlm("y_valid.csv", y_valid, ", ")

    # h5open("datasets/data1d_Limiter_NNPP.h5", "w") do file
    #     write(file, "X", x_train)
    #     write(file, "Y", y_train)
    # end

    # h5open("datasets/validdata1d_Limiter_NNPP.h5", "w") do file
    #     write(file, "X", x_valid)
    #     write(file, "Y", y_valid)
    # end