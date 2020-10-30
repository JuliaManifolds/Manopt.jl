M_n_32_k_32_m_2 = Stiefel(32,32)
A_n_32_k_32_m_2 = randn(32,32)
A_n_32_k_32_m_2 = (A_n_32_k_32_m_2 + A_n_32_k_32_m_2')
N_n_32_k_32_m_2 = diagm(32:-1:1)
F_n_32_k_32_m_2(X::Array{Float64,2}) = tr(X' * A_n_32_k_32_m_2 * X * N_n_32_k_32_m_2)
∇F_n_32_k_32_m_2(X::Array{Float64,2}) = 2 * A_n_32_k_32_m_2 * X * N_n_32_k_32_m_2 - X * X' * A_n_32_k_32_m_2 * X * N_n_32_k_32_m_2 - X * N_n_32_k_32_m_2 * X' * A_n_32_k_32_m_2 * X
sample_times_n_32_k_32_m_2 = []

for i in 1:2
    x = random_point(M_n_32_k_32_m_2)
    norm_gradient_stopping = norm(M_n_32_k_32_m_2,x,∇F_n_32_k_32_m_2(x))*10^(-6)
    bench_n_32_k_32_m_2 = @benchmark quasi_Newton(M_n_32_k_32_m_2, F_n_32_k_32_m_2, ∇F_n_32_k_32_m_2, $x; memory_size = 2, vector_transport_method = ProjectionTransport(), stopping_criterion = StopWhenGradientNormLess($norm_gradient_stopping)) seconds = 20 samples = 3
    append!(sample_times_n_32_k_32_m_2, bench_n_32_k_32_m_2.times)
end

result_n_32_k_32_m_2 = sum(sample_times_n_32_k_32_m_2) / length(sample_times_n_32_k_32_m_2) / 1000000000