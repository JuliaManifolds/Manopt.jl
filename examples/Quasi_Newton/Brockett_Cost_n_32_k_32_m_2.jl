import Manifolds: vector_transport_to!
vector_transport_to!(::Stiefel,Y,p,X,q,::ProjectionTransport) = (Y .= project(M_n_32_k_32_m_2, q, X))
M_n_32_k_32_m_2 = Stiefel(32,32)
A_n_32_k_32_m_2 = randn(32,32)
A_n_32_k_32_m_2 = (A_n_32_k_32_m_2 + A_n_32_k_32_m_2')
N_n_32_k_32_m_2 = diagm(32:-1:1)
F_n_32_k_32_m_2(X::Array{Float64,2}) = tr(X' * A_n_32_k_32_m_2 * X * N_n_32_k_32_m_2)
∇F_n_32_k_32_m_2(X::Array{Float64,2}) = 2 * A_n_32_k_32_m_2 * X * N_n_32_k_32_m_2 - X * X' * A_n_32_k_32_m_2 * X * N_n_32_k_32_m_2 - X * N_n_32_k_32_m_2 * X' * A_n_32_k_32_m_2 * X
sample_times_n_32_k_32_m_2 = []

for i in 1:10
    x = random_point(M_n_32_k_32_m_2)
    bench_n_32_k_32_m_2 = @benchmark quasi_Newton(M_n_32_k_32_m_2, F_n_32_k_32_m_2, ∇F_n_32_k_32_m_2, $x; memory_size = 2, vector_transport_method = ProjectionTransport(), retraction_method = QRRetraction(), stopping_criterion = StopWhenGradientNormLess(norm(M_n_32_k_32_m_2,$x,∇F_n_32_k_32_m_2($x))*10^(-6))) seconds = 600 samples = 10 evals = 1
    append!(sample_times_n_32_k_32_m_2, bench_n_32_k_32_m_2.times)
end

result_n_32_k_32_m_2 = sum(sample_times_n_32_k_32_m_2) / length(sample_times_n_32_k_32_m_2) / 1000000000