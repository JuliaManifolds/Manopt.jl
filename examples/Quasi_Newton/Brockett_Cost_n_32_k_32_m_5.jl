M_n_32_k_32_m_5 = Stiefel(32,32)
A_n_32_k_32_m_5 = randn(32,32)
A_n_32_k_32_m_5 = (A_n_32_k_32_m_5 + A_n_32_k_32_m_5')
N_n_32_k_32_m_5 = diagm(32:-1:1)
F_n_32_k_32_m_5(X::Array{Float64,2}) = tr(X' * A_n_32_k_32_m_5 * X * N_n_32_k_32_m_5)
∇F_n_32_k_32_m_5(X::Array{Float64,2}) = 2 * A_n_32_k_32_m_5 * X * N_n_32_k_32_m_5 - X * X' * A_n_32_k_32_m_5 * X * N_n_32_k_32_m_5 - X * N_n_32_k_32_m_5 * X' * A_n_32_k_32_m_5 * X
sample_times_n_32_k_32_m_5 = []

for i in 1:10
    x = random_point(M_n_32_k_32_m_5)
    bench_n_32_k_32_m_5 = @benchmark quasi_Newton(M_n_32_k_32_m_5, F_n_32_k_32_m_5, ∇F_n_32_k_32_m_5, $x; memory_size = 5, vector_transport_method = ProjectionTransport(), stopping_criterion = StopWhenGradientNormLess(norm(M_n_32_k_32_m_5,$x,∇F_n_32_k_32_m_5($x))*10^(-6))) seconds = 600 samples = 10 evals = 1
    append!(sample_times_n_32_k_32_m_5, bench_n_32_k_32_m_5.times)
end

result_n_32_k_32_m_5 = sum(sample_times_n_32_k_32_m_5) / length(sample_times_n_32_k_32_m_5) / 1000000000