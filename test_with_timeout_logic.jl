using Distributed
using Graphs

addprocs(1)
worker = workers()[1]

@everywhere [worker] Base.eval(Main, quote
    using Graphs
end)

println("Generating graph...")
g = erdos_renyi(20000, 0.05)
println("Spawnat...")
future = @spawnat worker betweenness_centrality(g)

start_time = time()
while !isready(future)
    if (time() - start_time) > 5
        println("Timeout hit! Killing proc...")
        try
            proc = Distributed.worker_from_id(worker).config.process
            println("Got proc: ", proc)
            kill(proc, Base.SIGKILL)
            println("Sent SIGKILL")
        catch e
            println("Error in kill: ", e)
        end
        println("Calling rmprocs...")
        rmprocs(worker; waitfor=0.0)
        println("rmprocs done!")
        break
    end
    sleep(1.0)
end
println("Finished with_timeout logic.")
