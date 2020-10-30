M_n_32_k_32_m_1 = Stiefel(32,32)
A_n_32_k_32_m_1 = randn(32,32)
A_n_32_k_32_m_1 = (A_n_32_k_32_m_1 + A_n_32_k_32_m_1')
N_n_32_k_32_m_1 = diagm(32:-1:1)
F_n_32_k_32_m_1(X::Array{Float64,2}) = tr(X' * A_n_32_k_32_m_1 * X * N_n_32_k_32_m_1)
∇F_n_32_k_32_m_1(X::Array{Float64,2}) = 2 * A_n_32_k_32_m_1 * X * N_n_32_k_32_m_1 - X * X' * A_n_32_k_32_m_1 * X * N_n_32_k_32_m_1 - X * N_n_32_k_32_m_1 * X' * A_n_32_k_32_m_1 * X
sample_times_n_32_k_32_m_1 = []

for i in 1:2
    x = random_point(M_n_32_k_32_m_1)
    norm_gradient_stopping = norm(M_n_32_k_32_m_1,x,∇F_n_32_k_32_m_1(x))*10^(-6)
    bench_n_32_k_32_m_1 = @benchmark quasi_Newton(M_n_32_k_32_m_1, F_n_32_k_32_m_1, ∇F_n_32_k_32_m_1, $x; memory_size = 1, vector_transport_method = ProjectionTransport(), stopping_criterion = StopWhenGradientNormLess($norm_gradient_stopping)) seconds = 20 samples = 3
    append!(sample_times_n_32_k_32_m_1, bench_n_32_k_32_m_1.times)
end

result_n_32_k_32_m_1 = sum(sample_times_n_32_k_32_m_1) / length(sample_times_n_32_k_32_m_1) / 1000000000