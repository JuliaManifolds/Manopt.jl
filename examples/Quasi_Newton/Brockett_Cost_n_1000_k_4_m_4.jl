import Manifolds: vector_transport_to!
vector_transport_to!(::Stiefel,Y,p,X,q,::ProjectionTransport) = (Y .= project(M_n_1000_k_4_m_4, q, X))
M_n_1000_k_4_m_4 = Stiefel(1000,4)
A_n_1000_k_4_m_4 = randn(1000,1000)
A_n_1000_k_4_m_4 = (A_n_1000_k_4_m_4 + A_n_1000_k_4_m_4')
N_n_1000_k_4_m_4 = diagm(4:-1:1)
F_n_1000_k_4_m_4(X::Array{Float64,2}) = tr(X' * A_n_1000_k_4_m_4 * X * N_n_1000_k_4_m_4)
∇F_n_1000_k_4_m_4(X::Array{Float64,2}) = 2 * A_n_1000_k_4_m_4 * X * N_n_1000_k_4_m_4 - X * X' * A_n_1000_k_4_m_4 * X * N_n_1000_k_4_m_4 - X * N_n_1000_k_4_m_4 * X' * A_n_1000_k_4_m_4 * X
sample_times_n_1000_k_4_m_4 = []

for i in 1:5
    x = random_point(M_n_1000_k_4_m_4)
    bench_n_1000_k_4_m_4 = @benchmark quasi_Newton(M_n_1000_k_4_m_4, F_n_1000_k_4_m_4, ∇F_n_1000_k_4_m_4, $x; memory_size = 4, vector_transport_method = ProjectionTransport(), retraction_method = QRRetraction(), cautious_update = true, stopping_criterion = StopWhenGradientNormLess(norm(M_n_1000_k_4_m_4,$x,∇F_n_1000_k_4_m_4($x))*10^(-6))) seconds = 600 samples = 1 evals = 1
    append!(sample_times_n_1000_k_4_m_4, bench_n_1000_k_4_m_4.times)
end

result_n_1000_k_4_m_4 = sum(sample_times_n_1000_k_4_m_4) / length(sample_times_n_1000_k_4_m_4) / 1000000000