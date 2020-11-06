import Manifolds: vector_transport_to!
vector_transport_to!(::Stiefel,Y,p,X,q,::ProjectionTransport) = (Y .= project(M_n_32_k_32_m_32, q, X))
M_n_32_k_32_m_32 = Stiefel(32,32)
A_n_32_k_32_m_32 = randn(32,32)
A_n_32_k_32_m_32 = (A_n_32_k_32_m_32 + A_n_32_k_32_m_32')
N_n_32_k_32_m_32 = diagm(32:-1:1)
F_n_32_k_32_m_32(X::Array{Float64,2}) = tr(X' * A_n_32_k_32_m_32 * X * N_n_32_k_32_m_32)
∇F_n_32_k_32_m_32(X::Array{Float64,2}) = 2 * A_n_32_k_32_m_32 * X * N_n_32_k_32_m_32 - X * X' * A_n_32_k_32_m_32 * X * N_n_32_k_32_m_32 - X * N_n_32_k_32_m_32 * X' * A_n_32_k_32_m_32 * X
sample_times_n_32_k_32_m_32 = []

for i in 1:10
    x = random_point(M_n_32_k_32_m_32)
    bench_n_32_k_32_m_32 = @benchmark quasi_Newton(M_n_32_k_32_m_32, F_n_32_k_32_m_32, ∇F_n_32_k_32_m_32, $x; memory_size = 32, vector_transport_method = ProjectionTransport(), retraction_method = QRRetraction(), cautious_update = true, stopping_criterion = StopWhenGradientNormLess(norm(M_n_32_k_32_m_32,$x,∇F_n_32_k_32_m_32($x))*10^(-6))) seconds = 600 samples = 10 evals = 1
    append!(sample_times_n_32_k_32_m_32, bench_n_32_k_32_m_32.times)
end

result_n_32_k_32_m_32 = sum(sample_times_n_32_k_32_m_32) / length(sample_times_n_32_k_32_m_32) / 1000000000