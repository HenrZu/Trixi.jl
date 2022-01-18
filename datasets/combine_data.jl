using HDF5: h5open

function get_label(X1,X2,nnodes)

    alpha_max=0.5
    alpha_min=0.001

    # magic parameters
    threshold = 0.5 * 10^(-1.8 * (nnodes)^0.25)
    parameter_s = log((1 - 0.0001)/0.0001)

    # Calculate energy in higher modes
    energy = max(X1,X2)

    alpha_element = 1 / (1 + exp(-parameter_s / threshold * (energy - threshold)))

    # Take care of the case close to pure DG
    if alpha_element < alpha_min
      alpha_element = zero(alpha_element)
    end

    # Take care of the case close to pure FV
    if alpha_element > 1 - alpha_min
      alpha_element = one(alpha_element)
    end

    return min(alpha_max , alpha_element)
end


list_data = readdir("datasets//1d_files")

dataset_x1 = Float64[]
dataset_x2 = Float64[]
dataset_x3 = Float64[]
labels = Float64[]
X1 = Float64[]
X2 = Float64[]



for data in list_data
    
    println("Für folgendes Data set hinzu $data")

    Y = h5open(joinpath("datasets","1d_files",data), "r") do file
        read(file, "Y")
    end
    
    X = h5open(joinpath("datasets","1d_files",data), "r") do file
        read(file, "X")
    end

    data_x1 = X[1:3:end]
    data_x2 = X[2:3:end]
    data_x3 = X[3:3:end]

    for i = 1:length(data_x1)
        push!(dataset_x1,  data_x1[i])
        push!(dataset_x2, data_x2[i])
        push!(dataset_x3, data_x3[i])
        push!(labels,  Y[i])
    end
end

# Einzelne Vektoren in Array packen.
X_new = zeros(3, length(dataset_x1))
for i = 1:length(dataset_x1)
    X_new[1,i] = dataset_x1[i]
    X_new[2,i] = dataset_x2[i]
    X_new[3,i] = dataset_x3[i]
end



println(size(X_new))
println(size(labels))


# save data
h5open("datasets//1d_indicator_final.h5", "w") do file
    write(file, "X", X_new)
    write(file, "Y", labels)
end




  