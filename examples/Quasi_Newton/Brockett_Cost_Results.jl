using Manopt, Manifolds, ManifoldsBase, LinearAlgebra, Random, BenchmarkTools
Random.seed!(42)
result1 = include("Brockett_Cost_n_32_k_32_m_1.jl")
result2 = include("Brockett_Cost_n_32_k_32_m_2.jl")
result3 = include("Brockett_Cost_n_32_k_32_m_4.jl")
result4 = include("Brockett_Cost_n_32_k_32_m_8.jl")
result5 = include("Brockett_Cost_n_32_k_32_m_16.jl")
result6 = include("Brockett_Cost_n_32_k_32_m_32.jl")
result7 = include("Brockett_Cost_n_1000_k_2_m_4.jl")
result8 = include("Brockett_Cost_n_1000_k_3_m_4.jl")
result9 = include("Brockett_Cost_n_1000_k_4_m_4.jl")
result10 = include("Brockett_Cost_n_1000_k_5_m_4.jl")

print("Brockett Cost Function \n Experiment 1, n = 32, k = 32, memory = 1: $result1 seconds. \n Experiment 2, n = 32, k = 32, memory = 2: $result2 seconds. \n Experiment 3, n = 32, k = 32, memory = 4: $result3 seconds. \n Experiment 4, n = 32, k = 32, memory = 8: $result4 seconds. \n Experiment 5, n = 32, k = 32, memory = 16: $result5 seconds. \n Experiment 6, n = 32, k = 32, memory = 32: $result6 seconds. \n Experiment 7, n = 1000, k = 2, memory = 4: $result7 seconds. \n Experiment 8, n = 1000, k = 3, memory = 4: $result8 seconds. \n Experiment 9, n = 1000, k = 4, memory = 4: $result9 seconds. \n Experiment 10, n = 1000, k = 5, memory = 4: $result10 seconds. \n")
