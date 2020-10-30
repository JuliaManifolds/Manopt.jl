using Manopt, Manifolds, ManifoldsBase, Random, LinearAlgebra, BenchmarkTools
Random.seed!(42)
result1 = include("C:/Users/rieme/Manopt/examples/Quasi_Newton/Rayleigh_Quotient_n_100.jl")
result2 = include("C:/Users/rieme/Manopt/examples/Quasi_Newton/Rayleigh_Quotient_n_300.jl")
print("$result1 and $result2")