M_n_32_k_32_m_6 = Stiefel(32,32)
A_n_32_k_32_m_6 = randn(32,32)
A_n_32_k_32_m_6 = (A_n_32_k_32_m_6 + A_n_32_k_32_m_6')
N_n_32_k_32_m_6 = diagm(32:-1:1)
F_n_32_k_32_m_6(X::Array{Float64,2}) = tr(X' * A_n_32_k_32_m_6 * X * N_n_32_k_32_m_6)
∇F_n_32_k_32_m_6(X::Array{Float64,2}) = 2 * A_n_32_k_32_m_6 * X * N_n_32_k_32_m_6 - X * X' * A_n_32_k_32_m_6 * X * N_n_32_k_32_m_6 - X * N_n_32_k_32_m_6 * X' * A_n_32_k_32_m_6 * X
sample_times_n_32_k_32_m_6 = []

for i in 1:10
    x = random_point(M_n_32_k_32_m_6)
    bench_n_32_k_32_m_6 = @benchmark quasi_Newton(M_n_32_k_32_m_6, F_n_32_k_32_m_6, ∇F_n_32_k_32_m_6, $x; memory_size = 6, vector_transport_method = ProjectionTransport(), stopping_criterion = StopWhenGradientNormLess(norm(M_n_32_k_32_m_6,$x,∇F_n_32_k_32_m_6($x))*10^(-6))) seconds = 600 samples = 10 evals = 1
    append!(sample_times_n_32_k_32_m_6, bench_n_32_k_32_m_6.times)
end

result_n_32_k_32_m_6 = sum(sample_times_n_32_k_32_m_6) / length(sample_times_n_32_k_32_m_6) / 1000000000