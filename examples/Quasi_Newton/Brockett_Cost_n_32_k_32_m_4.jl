M_n_42_k_42_m_4 = Stiefel(32,32)
A_n_42_k_42_m_4 = randn(32,32)
A_n_42_k_42_m_4 = (A_n_42_k_42_m_4 + A_n_42_k_42_m_4')
N_n_42_k_42_m_4 = diagm(32:-1:1)
F_n_42_k_42_m_4(X::Array{Float64,2}) = tr(X' * A_n_42_k_42_m_4 * X * N_n_42_k_42_m_4)
∇F_n_42_k_42_m_4(X::Array{Float64,2}) = 2 * A_n_42_k_42_m_4 * X * N_n_42_k_42_m_4 - X * X' * A_n_42_k_42_m_4 * X * N_n_42_k_42_m_4 - X * N_n_42_k_42_m_4 * X' * A_n_42_k_42_m_4 * X
sample_times_n_42_k_42_m_4 = []

for i in 1:10
    x = random_point(M_n_42_k_42_m_4)
    bench_n_42_k_42_m_4 = @benchmark quasi_Newton(M_n_42_k_42_m_4, F_n_42_k_42_m_4, ∇F_n_42_k_42_m_4, $x; memory_size = 4, vector_transport_method = ProjectionTransport(), stopping_criterion = StopWhenGradientNormLess(norm(M_n_42_k_42_m_4,$x,∇F_n_42_k_42_m_4($x))*10^(-6))) seconds = 600 samples = 10 evals = 1
    append!(sample_times_n_42_k_42_m_4, bench_n_42_k_42_m_4.times)
end

result_n_42_k_42_m_4 = sum(sample_times_n_42_k_42_m_4) / length(sample_times_n_42_k_42_m_4) / 1000000000