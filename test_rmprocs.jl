using Distributed
addprocs(1)
w = workers()[1]
future = @spawnat w begin
    # tight loop in C: we can use sleep in C
    ccall(:sleep, Cuint, (Cuint,), 100)
end
sleep(2)
println("calling rmprocs...")
@time rmprocs(w)
println("done!")
