#
#   Posituve Definite Karcher Mean (Matlab Manopt)
#
using Manopt, Manifolds, ManifoldsBase, LinearAlgebra, Random
Random.seed!(42)
n = 5
m = 50
M = SymmetricPositiveDefinite(n)
x = diagm(ones(n))
A = [exp(M, x, random_tangent(M,x, Val(:Rician), 0.05)) for _ ∈ 1:m]
F(X::Array{Float64,2}) = sum([ distance(M,X,B)^2 for B ∈ A]) / (2*m)
∇F(X::Array{Float64,2}) = - sum([  log(M, X, B) for B ∈ A]) / m

B1 = quasi_Newton(M,F,∇F,x)
B2 = mean(M,A)
