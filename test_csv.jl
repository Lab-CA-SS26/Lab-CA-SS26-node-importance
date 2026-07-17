using DataFrames
using CSV

# Write CSV
open("test.csv", "w") do f
    write(f, "Dataset,Nodes,Edges,Runtime_s,Status\n")
    write(f, "p2p-Gnutella31.txt,62586,147892,2222.5444231033325,Success\n")
end

timing_df = CSV.read("test.csv", DataFrame)
dataset_name = basename("TestInstances/p2p-Gnutella31.txt")

t_row = filter(row -> row.Dataset == dataset_name, timing_df)
println("t_row size: ", size(t_row))
if !isempty(t_row)
    println("t_exact: ", t_row.Runtime_s[1])
end
