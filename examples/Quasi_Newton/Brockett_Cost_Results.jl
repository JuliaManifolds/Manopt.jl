using Manopt, Manifolds, ManifoldsBase, LinearAlgebra, Random, BenchmarkTools
import Manifolds: vector_transport_to!
vector_transport_to!(::Stiefel,Y,p,X,q,::ProjectionTransport) = (Y .= project(M, q, X))
Random.seed!(42)
result1 = include("C:/Users/rieme/Manopt/examples/Quasi_Newton/Brockett_Cost_n_32_k_32_m_1.jl")
result2 = include("C:/Users/rieme/Manopt/examples/Quasi_Newton/Brockett_Cost_n_32_k_32_m_2.jl")
result3 = include("C:/Users/rieme/Manopt/examples/Quasi_Newton/Brockett_Cost_n_32_k_32_m_3.jl")
result4 = include("C:/Users/rieme/Manopt/examples/Quasi_Newton/Brockett_Cost_n_32_k_32_m_4.jl")
result5 = include("C:/Users/rieme/Manopt/examples/Quasi_Newton/Brockett_Cost_n_32_k_32_m_5.jl")
result6 = include("C:/Users/rieme/Manopt/examples/Quasi_Newton/Brockett_Cost_n_32_k_32_m_6.jl")
result7 = include("C:/Users/rieme/Manopt/examples/Quasi_Newton/Brockett_Cost_n_1000_k_2_m_4.jl")
result8 = include("C:/Users/rieme/Manopt/examples/Quasi_Newton/Brockett_Cost_n_1000_k_3_m_4.jl")
result9 = include("C:/Users/rieme/Manopt/examples/Quasi_Newton/Brockett_Cost_n_1000_k_4_m_4.jl")
result10 = include("C:/Users/rieme/Manopt/examples/Quasi_Newton/Brockett_Cost_n_1000_k_5_m_4.jl")

print("Brockett Cost Function \n Experiment 1, n = 32, k = 32, memory = 1: $result1 seconds. \n Experiment 2, n = 32, k = 32, memory = 2: $result2 seconds. \n Experiment 3, n = 32, k = 32, memory = 3: $result3 seconds. \n Experiment 4, n = 32, k = 32, memory = 4: $result4 seconds. \n Experiment 5, n = 32, k = 32, memory = 5: $result5 seconds. \n Experiment 6, n = 32, k = 32, memory = 6: $result6 seconds. \n Experiment 7, n = 1000, k = 2, memory = 4: $result7 seconds. \n Experiment 8, n = 1000, k = 3, memory = 4: $result8 seconds. \n Experiment 9, n = 1000, k = 4, memory = 4: $result9 seconds. \n Experiment 10, n = 1000, k = 5, memory = 4: $result10 seconds. \n")