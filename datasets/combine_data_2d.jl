using HDF5: h5open

list_data = readdir("datasets//2d_final")

dataset_x1 = Float64[]
dataset_x2 = Float64[]
dataset_x3 = Float64[]
dataset_x4 = Float64[]
labels = Float64[]
X1 = Float64[]
X2 = Float64[]



for data in list_data
    
    println("Füge folgendes Data set hinzu $data")

    Y = h5open(joinpath("datasets","2d_final",data), "r") do file
        read(file, "Y")
    end
    
    X = h5open(joinpath("datasets","2d_final",data), "r") do file
        read(file, "X")
    end

    datasetx1 = X[1:4:end]
    datasetx2 = X[2:4:end]
    datasetx3 = X[3:4:end]
    datasetx4 = X[4:4:end]

    
    for i = 1:Int(length(datasetx2))
        push!(dataset_x1,  datasetx1[i])
        push!(dataset_x2,  datasetx2[i])
        push!(dataset_x3,  datasetx3[i])
        push!(dataset_x4,  datasetx4[i])
        push!(labels,  Y[i])
    end
end

# Einzelne Vektoren in Array packen.
X_new = zeros(4, length(dataset_x1))
for i = 1:length(dataset_x1)
    X_new[1,i] = dataset_x1[i]
    X_new[2,i] = dataset_x2[i]
    X_new[3,i] = dataset_x3[i]
    X_new[4,i] = dataset_x4[i]
end



println(size(X_new))
println(size(labels))


# save data
h5open("datasets//2d_unlimited_final2.h5", "w") do file
    write(file, "X", X_new)
    write(file, "Y", labels)
end




  