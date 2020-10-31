import Manifolds: vector_transport_to!
vector_transport_to!(::Stiefel,Y,p,X,q,::ProjectionTransport) = (Y .= project(M_n_32_k_32_m_8, q, X))
M_n_32_k_32_m_8 = Stiefel(32,32)
A_n_32_k_32_m_8 = randn(32,32)
A_n_32_k_32_m_8 = (A_n_32_k_32_m_8 + A_n_32_k_32_m_8')
N_n_32_k_32_m_8 = diagm(32:-1:1)
F_n_32_k_32_m_8(X::Array{Float64,2}) = tr(X' * A_n_32_k_32_m_8 * X * N_n_32_k_32_m_8)
∇F_n_32_k_32_m_8(X::Array{Float64,2}) = 2 * A_n_32_k_32_m_8 * X * N_n_32_k_32_m_8 - X * X' * A_n_32_k_32_m_8 * X * N_n_32_k_32_m_8 - X * N_n_32_k_32_m_8 * X' * A_n_32_k_32_m_8 * X
sample_times_n_32_k_32_m_8 = []

for i in 1:10
    x = random_point(M_n_32_k_32_m_8)
    bench_n_32_k_32_m_8 = @benchmark quasi_Newton(M_n_32_k_32_m_8, F_n_32_k_32_m_8, ∇F_n_32_k_32_m_8, $x; memory_size = 8, vector_transport_method = ProjectionTransport(), stopping_criterion = StopWhenGradientNormLess(norm(M_n_32_k_32_m_8,$x,∇F_n_32_k_32_m_8($x))*10^(-6))) seconds = 600 samples = 10 evals = 1
    append!(sample_times_n_32_k_32_m_8, bench_n_32_k_32_m_8.times)
end

result_n_32_k_32_m_8 = sum(sample_times_n_32_k_32_m_8) / length(sample_times_n_32_k_32_m_8) / 1000000000