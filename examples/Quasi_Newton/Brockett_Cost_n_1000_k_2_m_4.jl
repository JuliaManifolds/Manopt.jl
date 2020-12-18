import Manifolds: vector_transport_to!
function vector_transport_to!(::Stiefel, Y, p, X, q, ::ProjectionTransport)
    return (Y .= project(M_n_1000_k_2_m_4, q, X))
end
M_n_1000_k_2_m_4 = Stiefel(1000, 2)
A_n_1000_k_2_m_4 = randn(1000, 1000)
A_n_1000_k_2_m_4 = (A_n_1000_k_2_m_4 + A_n_1000_k_2_m_4')
N_n_1000_k_2_m_4 = diagm(2:-1:1)
F_n_1000_k_2_m_4(X::Array{Float64,2}) = tr(X' * A_n_1000_k_2_m_4 * X * N_n_1000_k_2_m_4)
function ∇F_n_1000_k_2_m_4(X::Array{Float64,2})
    return 2 * A_n_1000_k_2_m_4 * X * N_n_1000_k_2_m_4 -
           X * X' * A_n_1000_k_2_m_4 * X * N_n_1000_k_2_m_4 -
           X * N_n_1000_k_2_m_4 * X' * A_n_1000_k_2_m_4 * X
end
sample_times_n_1000_k_2_m_4 = []

for i in 1:5
    x = random_point(M_n_1000_k_2_m_4)
    bench_n_1000_k_2_m_4 = @benchmark quasi_Newton(
        M_n_1000_k_2_m_4,
        F_n_1000_k_2_m_4,
        ∇F_n_1000_k_2_m_4,
        $x;
        memory_size=4,
        vector_transport_method=ProjectionTransport(),
        retraction_method=QRRetraction(),
        cautious_update=true,
        stopping_criterion=StopWhenGradientNormLess(
            norm(M_n_1000_k_2_m_4, $x, ∇F_n_1000_k_2_m_4($x)) * 10^(-6)
        ),
        debug=[:Iteration, " ", :Cost, "\n", 1],
    ) seconds = 600 samples = 1 evals = 1
    append!(sample_times_n_1000_k_2_m_4, bench_n_1000_k_2_m_4.times)
end

result_n_1000_k_2_m_4 =
    sum(sample_times_n_1000_k_2_m_4) / length(sample_times_n_1000_k_2_m_4) / 1000000000
