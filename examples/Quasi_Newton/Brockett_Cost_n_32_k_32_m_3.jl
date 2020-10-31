M_n_32_k_32_m_3 = Stiefel(32,32)
A_n_32_k_32_m_3 = randn(32,32)
A_n_32_k_32_m_3 = (A_n_32_k_32_m_3 + A_n_32_k_32_m_3')
N_n_32_k_32_m_3 = diagm(32:-1:1)
F_n_32_k_32_m_3(X::Array{Float64,2}) = tr(X' * A_n_32_k_32_m_3 * X * N_n_32_k_32_m_3)
∇F_n_32_k_32_m_3(X::Array{Float64,2}) = 2 * A_n_32_k_32_m_3 * X * N_n_32_k_32_m_3 - X * X' * A_n_32_k_32_m_3 * X * N_n_32_k_32_m_3 - X * N_n_32_k_32_m_3 * X' * A_n_32_k_32_m_3 * X
sample_times_n_32_k_32_m_3 = []

for i in 1:10
    x = random_point(M_n_32_k_32_m_3)
    bench_n_32_k_32_m_3 = @benchmark quasi_Newton(M_n_32_k_32_m_3, F_n_32_k_32_m_3, ∇F_n_32_k_32_m_3, $x; memory_size = 3, vector_transport_method = ProjectionTransport(), stopping_criterion = StopWhenGradientNormLess(norm(M_n_32_k_32_m_3,$x,∇F_n_32_k_32_m_3($x))*10^(-6))) seconds = 600 samples = 10 evals = 1
    append!(sample_times_n_32_k_32_m_3, bench_n_32_k_32_m_3.times)
end

result_n_32_k_32_m_3 = sum(sample_times_n_32_k_32_m_3) / length(sample_times_n_32_k_32_m_3) / 1000000000