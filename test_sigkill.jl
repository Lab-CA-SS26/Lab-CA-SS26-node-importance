using Distributed
addprocs(1)
w = workers()[1]
future = @spawnat w begin
    ccall(:sleep, Cuint, (Cuint,), 100)
end
sleep(2)
println("killing worker process manually...")
proc = Distributed.worker_from_id(w).config.process
kill(proc, Base.SIGKILL)
println("calling rmprocs...")
@time rmprocs(w; waitfor=0.0) # Waitfor=0.0 just removes it from the local registry without hanging
println("done!")
