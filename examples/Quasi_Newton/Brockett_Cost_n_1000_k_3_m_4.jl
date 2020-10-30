M_n_1000_k_3_m_4 = Stiefel(1000,3)
A_n_1000_k_3_m_4 = randn(1000,1000)
A_n_1000_k_3_m_4 = (A_n_1000_k_3_m_4 + A_n_1000_k_3_m_4')
N_n_1000_k_3_m_4 = diagm(3:-1:1)
F_n_1000_k_3_m_4(X::Array{Float64,2}) = tr(X' * A_n_1000_k_3_m_4 * X * N_n_1000_k_3_m_4)
∇F_n_1000_k_3_m_4(X::Array{Float64,2}) = 2 * A_n_1000_k_3_m_4 * X * N_n_1000_k_3_m_4 - X * X' * A_n_1000_k_3_m_4 * X * N_n_1000_k_3_m_4 - X * N_n_1000_k_3_m_4 * X' * A_n_1000_k_3_m_4 * X
sample_times_n_1000_k_3_m_4 = []

for i in 1:2
    x = random_point(M_n_1000_k_3_m_4)
    norm_gradient_stopping = norm(M_n_1000_k_3_m_4,x,∇F_n_1000_k_3_m_4(x))*10^(-6)
    bench_n_1000_k_3_m_4 = @benchmark quasi_Newton(M_n_1000_k_3_m_4, F_n_1000_k_3_m_4, ∇F_n_1000_k_3_m_4, $x; memory_size = 4, vector_transport_method = ProjectionTransport(), stopping_criterion = StopWhenGradientNormLess($norm_gradient_stopping)) seconds = 20 samples = 3
    append!(sample_times_n_1000_k_3_m_4, bench_n_1000_k_3_m_4.times)
end

result_n_1000_k_3_m_4 = sum(sample_times_n_1000_k_3_m_4) / length(sample_times_n_1000_k_3_m_4) / 1000000000