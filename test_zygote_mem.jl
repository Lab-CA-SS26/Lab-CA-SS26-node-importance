using Flux
using Zygote
using LinearAlgebra
using SparseArrays

N = 10_000
F = 12

X = randn(Float32, F, N)
A = sprand(Float32, N, N, 0.01)
A_t = copy(A')

layer = Dense(F, F)

function norm2_features(x)
    n = sqrt.(sum(x.^2, dims=1) .+ 1e-8)
    return x ./ n
end

function (l::typeof(layer))(X::AbstractMatrix, A_transposed)
    Z = l.weight * X
    out = (Zygote.dropgrad(A_transposed) * Z')'
    return norm2_features(relu.(out))
end

loss, grads = Flux.withgradient(layer) do l
    preds = l(X, A_t)
    sum(preds)
end
println("Success!")
