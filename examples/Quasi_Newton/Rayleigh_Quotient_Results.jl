using Manopt, Manifolds, ManifoldsBase, Random, LinearAlgebra, BenchmarkTools
Random.seed!(42)
result1 = include("Rayleigh_Quotient_n_100.jl")
result2 = include("Rayleigh_Quotient_n_300.jl")
print("Rayleigh Quotient Minimization \n Experiment 1, n=100: $result1 seconds. \n Experiment 2, n=300: $result2 seconds.")

# n=100: 0.15275131555500002 seconds 
# n=300: 0.9617019966510068 seconds 
