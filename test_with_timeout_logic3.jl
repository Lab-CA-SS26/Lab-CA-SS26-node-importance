using Distributed

addprocs(1)
worker = workers()[1]

future = @spawnat worker begin
    ccall(:sleep, Cuint, (Cuint,), 100)
end

done = Channel{Bool}(1)

@async begin
    try
        fetch(future)
        put!(done, true)
    catch
        # Ignore errors if killed
    end
end

@async begin
    sleep(2)
    put!(done, false)
end

println("Waiting...")
success = take!(done)
println("Success: ", success)
if !success
    println("Timeout hit! Killing proc...")
    try
        proc = Distributed.worker_from_id(worker).config.process
        kill(proc, Base.SIGKILL)
        println("Sent SIGKILL")
    catch e
        println("Error in kill: ", e)
    end
    rmprocs(worker; waitfor=0.0)
end
println("done!")
