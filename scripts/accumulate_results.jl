using JSON
using CSV
using DataFrames

function parse_results(output_dir)
    results = DataFrame(
        graph = String[],
        k = Int[],
        threads = Int[],
        version = String[],
        execution_time = Float64[],
        io_time = Float64[],
        num_samples = Int[]
    )
    
    for exp_dir in readdir(output_dir, join=true)
        if !isdir(exp_dir)
            continue
        end
        basename_exp = basename(exp_dir)
        if startswith(basename_exp, "kadabra-")
            # process all .stats.json files
            for file in readdir(exp_dir, join=true)
                if endswith(file, ".stats.json")
                    try
                        data = JSON.parsefile(file)
                        params = get(data, "parameters", Dict())
                        
                        graph_path = get(params, "input_file", "")
                        graph_name = splitext(basename(graph_path))[1]
                        
                        push!(results, (
                            graph_name,
                            get(params, "k", -1),
                            get(params, "threads", -1),
                            get(params, "version", "unknown"),
                            get(data, "execution_time_seconds", NaN),
                            get(data, "io_time_seconds", NaN),
                            get(data, "num_samples", -1)
                        ))
                    catch e
                        println("Error processing $file: $e")
                    end
                end
            end
        end
    end
    return results
end

output_dir = joinpath(@__DIR__, "..", "benchmark", "output")
results = parse_results(output_dir)

for lang in ["cpp", "julia"]
    lang_results = filter(row -> row.version == lang, results)
    out_dir = joinpath(@__DIR__, "..", "benchmark", "results", "kadabra", lang)
    mkpath(out_dir)
    out_file = joinpath(out_dir, "results.csv")
    CSV.write(out_file, lang_results)
    println("Saved $(nrow(lang_results)) $lang results to $out_file")
end
