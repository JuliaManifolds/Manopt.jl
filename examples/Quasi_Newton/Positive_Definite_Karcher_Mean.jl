#
#   Posituve Definite Karcher Mean (Matlab Manopt)
#
using Manopt, Manifolds, ManifoldsBase, LinearAlgebra, Random
Random.seed!(42)
n = 5
m = 50
M = SymmetricPositiveDefinite(n)
x = random_point(M)
A = [random_point(M) for _ ∈ 1:m]
F(X::Array{Float64,2}) = sum([ distance(M,X,B)^2 for B ∈ A]) / (2*m)
∇F(X::Array{Float64,2}) = - sum([  log(M, X, B) for B ∈ A]) / m

@time quasi_Newton(M,F,∇F,x; memory_size = -100, stopping_criterion = StopWhenGradientNormLess(norm(M,x,∇F(x))*10^(-6)),debug = [:Iteration, " ", :Cost, "\n", 1, :Stop])

# B1 = quasi_Newton(M,F,∇F,x; memory_size = 100, debug = [:Iteration, " ", :Cost, "\n", 1, :Stop])
# B2 = mean(M,A)
