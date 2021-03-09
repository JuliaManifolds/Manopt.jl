#
#   Thomson Problem on Oblique(n,m)
#
using Manopt, Manifolds, ManifoldsBase, LinearAlgebra, Random
Random.seed!(42)
n = 50
m = 20
M = Oblique(n, m)

function F(::Oblique, X::Array{Float64,2})
    f = 0
    for i in 1:m
        for j in 1:m
            if i != j
                f = f + 1 / (norm(X[:, i] - X[:, j])^2)
            end
        end
    end
    return f
end

function gradF(::Oblique, X::Array{Float64,2})
    g = zeros(n, m)
    Id = Matrix(I, n, n)
    for i in 1:m
        f = zeros(n, 1)
        for j in 1:m
            if i != j
                f = f + 1 / (1.0 - X[:, i]' * X[:, j]) * X[:, j]
            end
        end
        g[:, i] = (Id - X[:, i] * X[:, i]') * f
    end
    return g
end

x = random_point(M)

@time quasi_Newton(
    M,
    F,
    gradF,
    x;
    memory_size=100,
    vector_transport_method=PowerVectorTransport(ParallelTransport()),
    stopping_criterion=StopWhenGradientNormLess(norm(M, x, gradF(M, x)) * 10^(-6)),
    debug=[:Iteration, " ", :Cost, "\n", 1, :Stop],
)
