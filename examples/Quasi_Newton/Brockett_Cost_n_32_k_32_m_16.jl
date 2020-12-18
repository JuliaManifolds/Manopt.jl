import Manifolds: vector_transport_to!
function vector_transport_to!(::Stiefel, Y, p, X, q, ::ProjectionTransport)
    return (Y .= project(M_n_32_k_32_m_16, q, X))
end
M_n_32_k_32_m_16 = Stiefel(32, 32)
A_n_32_k_32_m_16 = randn(32, 32)
A_n_32_k_32_m_16 = (A_n_32_k_32_m_16 + A_n_32_k_32_m_16')
N_n_32_k_32_m_16 = diagm(32:-1:1)
F_n_32_k_32_m_16(X::Array{Float64,2}) = tr(X' * A_n_32_k_32_m_16 * X * N_n_32_k_32_m_16)
function ∇F_n_32_k_32_m_16(X::Array{Float64,2})
    return 2 * A_n_32_k_32_m_16 * X * N_n_32_k_32_m_16 -
           X * X' * A_n_32_k_32_m_16 * X * N_n_32_k_32_m_16 -
           X * N_n_32_k_32_m_16 * X' * A_n_32_k_32_m_16 * X
end
sample_times_n_32_k_32_m_16 = []

for i in 1:10
    x = random_point(M_n_32_k_32_m_16)
    bench_n_32_k_32_m_16 = @benchmark quasi_Newton(
        M_n_32_k_32_m_16,
        F_n_32_k_32_m_16,
        ∇F_n_32_k_32_m_16,
        $x;
        memory_size=16,
        vector_transport_method=ProjectionTransport(),
        retraction_method=QRRetraction(),
        cautious_update=true,
        stopping_criterion=StopWhenGradientNormLess(
            norm(M_n_32_k_32_m_16, $x, ∇F_n_32_k_32_m_16($x)) * 10^(-6)
        ),
    ) seconds = 600 samples = 10 evals = 1
    append!(sample_times_n_32_k_32_m_16, bench_n_32_k_32_m_16.times)
end

result_n_32_k_32_m_16 =
    sum(sample_times_n_32_k_32_m_16) / length(sample_times_n_32_k_32_m_16) / 1000000000
