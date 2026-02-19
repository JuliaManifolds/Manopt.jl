using Manopt, Manifolds, LinearAlgebra, Test, Chairmarks

M = Euclidean(3)
pts = [
    [0.0, 0.0, 1.0],
    [sqrt(0.19), 0.0, 0.9],
    [-1 / sqrt(2), 1 / sqrt(2), 0.0],
]
p0 = [0.0, 0.0, 78.0]
# We do a full function approach here


struct Fi_block{TPI}
    p_i::TPI
end
(f::Fi_block)(M::AbstractManifold, p) = distance(M, p, f.p_i)

struct jacFi_block{TPI}
    p_i::TPI
end
function (f::jacFi_block)(M::AbstractManifold, Y, p)
    if distance(M, p, f.p_i) == 0
        fill!(Y, 0.0)
    else
        log!(M, Y, p, f.p_i)
        Y ./= -distance(M, p, f.p_i)
    end
    return Y
end

Fi = [ Fi_block(q) for q in pts]
grad_Fi = [ jacFi_block(q) for q in pts]
# Block s normal ones
Fs = [
    VectorGradientFunction(
            [Fi[i]], [grad_Fi[i]], 1;
            evaluation = InplaceEvaluation(), function_type = ComponentVectorialType(), jacobian_type = ComponentVectorialType()
        ) for i in eachindex(pts)
]


qc = median(M, pts)
cost(M, p) = sum(distance(M, p, q) for q in pts)

# Default Residual CG on this approach – works but probably allocates a bit too much (matrices coordinates/vector...)
q1 = LevenbergMarquardt(
    M, Fs, p0;
    β = 8.0, η = 0.2, damping_term_min = 1.0e-5, ε = 1.0e-1, # α_mode = :Strict,
    robustifier = fill((1 / 30) ∘ HuberRobustifier(), length(Fs)),
    debug = [:Iteration, :Cost, " ", :Change, " ", :damping_term, "\n", :Stop, 100],
)
# ... but works
@info "Cost of median (qc) $(cost(M, qc)), Cost of LM (q1): $(cost(M, q1)), difference (of q1 - qc): $(cost(M, q1) - cost(M, qc))"

q2 = LevenbergMarquardt(
    M, Fs, p0;
    β = 8.0, η = 0.2, damping_term_min = 1.0e-5, ε = 1.0e-1, α_mode = :Strict,
    robustifier = fill((1 / 30) ∘ HuberRobustifier(), length(Fs)),
    debug = [:Iteration, :Cost, " ", :Change, " ", :damping_term, "\n", :Stop, 100],
    sub_state = CoordinatesNormalSystemState(M),
)
# ... but works
@info "Cost of mean (qc) $(cost(M, qc)), Cost of LM (q2): $(cost(M, q2)), difference (of q2 - qc): $(cost(M, q2) - cost(M, qc))"

function run_lm_benchmark_1()
    q1b = copy(M, p0)
    LevenbergMarquardt!(M, Fs, q1b; β = 8.0, η = 0.2, damping_term_min = 1.0e-5, robustifier = fill((1 / 30) ∘ HuberRobustifier(), length(Fs)))
    return q1b
end

(@b run_lm_benchmark_1()) |> repr |> println

#@info distance(M, q1, q1b)

function run_lm_benchmark_2()
    q2b = copy(M, p0)
    LevenbergMarquardt!(
        M, Fs, q2b;
        β = 8.0, η = 0.2, damping_term_min = 1.0e-5, robustifier = fill((1 / 30) ∘ HuberRobustifier(), length(Fs)), sub_state = CoordinatesNormalSystemState(M),
    )
    return q2b
end

(
    @b run_lm_benchmark_2()
) |> repr |> println

#@info distance(M, q2, q2b)

function run_n_times(f, n)
    for i in 1:n
        f()
    end
    return
end

using Profile, ProfileView

ProfileView.@profview run_n_times(run_lm_benchmark_1, 500)
