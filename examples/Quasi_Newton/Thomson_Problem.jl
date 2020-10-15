#
#   Thomson Problem on Oblique(n,m)
#
using Manopt, Manifolds, ManifoldsBase, LinearAlgebra, Random
Random.seed!(42)
n = 50
m = 20
M = Oblique(n,m)

function F(X::Array{Float64,2})
    f = 0
    for i in 1:m
        for j in 1:m
            if i != j
                f = f + 1/(norm(X[:,i]-X[:,j])^2)
            end
        end
    end
    return f
end

function ∇F(X::Array{Float64,2})
    g = zeros(n,m)
    I = one(zeros(n,n))
    for i in 1:m
        f = zeros(n,1)
        for j in 1:m
            if i != j
                f = f + 1 / (1. - X[:,i]'*X[:,j]) * X[:,j]
            end
        end
        g[:,i] = (I - X[:,i]*X[:,i]')*f
    end
    return g
end

x = random_point(M)

quasi_Newton(M,F,∇F,x;memory_size = -1,vector_transport_method = PowerVectorTransport(ParallelTransport()), debug = [:Iteration, " ", :Cost, "\n", 1, :Stop])
