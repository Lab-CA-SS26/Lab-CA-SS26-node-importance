using Flux
using CUDA

model = Dense(10 => 1) |> gpu
x = rand(Float32, 10, 5) |> gpu

loss, grads = Flux.withgradient(model) do m
    preds = m(x)
    preds_cpu = cpu(preds)
    sum(preds_cpu.^2)
end

println("Grads: ", grads[1] !== nothing)
