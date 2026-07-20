using Distributed
addprocs(1)
w = workers()[1]
future = @spawnat w begin
    ccall(:sleep, Cuint, (Cuint,), 100)
end
sleep(2)
try
    proc = Distributed.worker_from_id(w).config.process
    println("Proc: ", proc)
    kill(proc, Base.SIGKILL)
catch e
    println("Error killing proc: ", e)
end
println("calling rmprocs...")
@time rmprocs(w; waitfor=0.0)
println("done!")
