using Chairmarks, CSV, DataFrames, Manopt, Manifolds, LinearAlgebra

raw"""
    generate_data(d)

Generate a data matrix ``A ∈ ℝ^{d× n}`` in a deterministic way,
here by using ``n = \frac{d(d-1)}{2}`` columns.
"""
function generate_data(d)
    # (1) push the first unit vector scaled by 1/2
    n = Int(d * (d + 1) / 2)
    A = zeros(d, n)
    A[1, 1] = 1.0
    k = 2
    dir = zeros(d)
    for i in 2:d # for each following dimension, set point on the diagonal of the preceding dimensions
        dir[1:i] .= 1 / sqrt(i)
        dir[(i + 1):end] .= 0
        v = range(0.0, 1.0, i + 2)[2:(end - 1)]
        for vi in v
            A[:, k] .= vi * dir
            k = k + 1
        end
    end
    return A
end

skew(A) = 0.5 .* (A - A')

"""
    f(M, p; A, B, robustifier = )

For given matrices ``A, B ∈ ℝ^{d,n}`` compute the robust Procrustes cost

```math
F_i(p) = (A - pB)_i = a_i - pb_i
```
"""
f(M, p; A, B) = sum(norm(A[:, i] - p * B[:, i]) for i in 1:size(A, 2))

"""
    Fi(M, p; i, A, B)
    Fi!(M, v, p; i, A, B)

For given matrices ``A, B ∈ ℝ^{d,n}`` compute the residual of the ith column

```math
F_i(p) = (A - pB)_i = a_i - pb_i
```

This can be computed in-place of `v`
"""
Fi(M, p; i, A, B) = A[:, i] - p * B[:, i]
Fi!(M, v, p; i, A, B) = (v .= A[:, i] .- p * B[:, i])

"""
    DFi(M, p, X; i, A, B)
    DFi!(M, y, p, X; i, A, B)

For given matrices ``A, B ∈ ℝ^{d,n}`` compute the differential of the residual of the ith column
with respect to the rotation ``p`` that is, ``X ∈ 𝔰𝔬(d)`` which reads

```math
\\mathcal J_{F_i}(p)[X] = DF_i(p)[X] = -pXb_i
```

This is computed in-place of `y`

"""
DFi(M, p, X; i, A, B) = - p * X * B[:, i]
DFi!(M, y, p, X; i, A, B) = (y .= - p * X * B[:, i])


raw"""
    adjointDFi(M, p, y; i, A, B)
    adjointDFi!(M, X, p, y; i, A, B)

For given matrices ``A, B ∈ ℝ^{d,n}`` compute the adjoint differential of the residual of the ith column
with respect to the rotation ``p`` that is, ``y ∈ ℝn`` but mapping into the Lie algebra ``𝔰𝔬(d)``.
This is also referred to as the Jacobian

```math
D*{F_i}(p)[y] = -\mathrm{skew}(p^{\mathrm{T}}yb_i^{\mathrm{T}})
```

This is computed in-place of `X`.
"""
adjointDFi(M, p, y; i, A, B) = - skew(p' * y * B[:, i]')
adjointDFi!(M, a, p, y; i, A, B) = (a .= - skew(p' * y * B[:, i]'))

"""
    rotation_matrix(d, i, j, α)

Create the rotation matrix in ``ℝ^{d×d}`` with a rotation in the ``i,j``-plance of an angle of `α`.
"""
function rotation_matrix(d, i, j, α)
    R = Matrix{Float64}(I, d, d)
    R[i, i] = cos(α)
    R[j, i] = cos(α)
    R[j, i] = -sin(α)
    R[i, j] = sin(α)
    return R
end

# Statistics:
matrix_sizes = collect(3:15)
num_experiments = length(matrix_sizes)
num_columns = zeros(Int, num_experiments)
manifold_dimensions = zeros(Int, num_experiments)
mean_time_rLM = zeros(Float64, num_experiments)
mean_time_LTMADS = zeros(Float64, num_experiments)
final_cost_rLM = zeros(Float64, num_experiments)
final_cost_LTMADS = zeros(Float64, num_experiments)
iterations_rLM = zeros(Int, num_experiments)
iterations_LTMADS = zeros(Int, num_experiments)

for (i, d) in enumerate(matrix_sizes)
    A = generate_data(d)
    n = size(A, 2) # number of summands in the vectorial cost sum
    p_star = Matrix{Float64}(I, d, d)
    for j in 1:(d - 1)
        p_star *= rotation_matrix(d, j, j + 1, π / (4 * (d - 1)))
    end
    B = copy(A)
    # and generate a few outliers in [3,6] so it also works already for d=3
    B[2, 4] += 0.1;  B[3, 1] += 0.1; B[3, 2] += 0.1; B[1, 6] += 0.1
    mask = B .== A
    # then rotate
    B = p_star' * B
    # Seting cost and vectorial block function
    F(M, p) = [Fi(M, p; i = i, A = A, B = B) for i in 1:n]
    vgfs = [
        VectorDifferentialFunction(
                (M, c, p) -> Fi!(M, c, p; i = i, A = A, B = B),
                (M, y, p, X) -> DFi!(M, y, p, X; i = i, A = A, B = B),
                (M, X, p, y) -> adjointDFi!(M, X, p, y; i = i, A = A, B = B),
                d;
                function_type = FunctionVectorialType(), jacobian_type = FunctionVectorialType(),
                adjoint_jacobian_type = FunctionVectorialType(), evaluation = InplaceEvaluation(),
            ) for i in 1:n
    ]
    rs = [ 1.0e-5 ∘ HuberRobustifier() for _ in 1:n ]

    M = Rotations(d)
    @info "d=$d (n=$n) dim: $(manifold_dimension(M))"
    # Start with the identity
    p0 = Matrix{Float64}(I, d, d)
    #
    # Solver runs. Both (a) an individual run to obtain stats like maxiter
    #
    # Least Squares Hubertized
    state1 = LevenbergMarquardt(
        M, vgfs, p0; robustifier = rs, damping_increase_factor = 4.0,
        candidate_acceptance_threshold = 0.2, damping_term_min = 1.0e-7, return_state = true
    )
    iter1 = get_count(state1, :Iterations)
    p1 = get_solver_result(state1)
    time1 = @be LevenbergMarquardt(
        $M, $vgfs, $p0; robustifier = $rs,
        damping_increase_factor = 4.0, candidate_acceptance_threshold = 0.2, damping_term_min = 1.0e-7,
    ) samples = 5 evals = 3
    state2 = mesh_adaptive_direct_search(
        M, (M, p) -> f(M, p; A = A, B = B), p0;
        stopping_criterion = StoppingCriterion = StopAfterIteration(20000) | StopWhenPollSizeLess(1.0e-10),
        return_state = true
    )
    time2 = @be mesh_adaptive_direct_search(
        $M, $((M, p) -> f(M, p; A = A, B = B)), $p0;
        stopping_criterion = $(StoppingCriterion = StopAfterIteration(20000) | StopWhenPollSizeLess(1.0e-10))
    ) samples = 5 evals = 3
    iter2 = get_count(get_state(state2), :Iterations)
    p2 = get_solver_result(state2)

    # Collect stats
    num_columns[i] = n
    manifold_dimensions[i] = manifold_dimension(M)
    mean_time_rLM[i] = mean(time1).time
    mean_time_LTMADS[i] = mean(time2).time
    final_cost_rLM[i] = f(M, p1; A = A, B = B)
    final_cost_LTMADS[i] = f(M, p2; A = A, B = B)
    iterations_rLM[i] = iter1
    iterations_LTMADS[i] = iter2
    @info "rLM   : #$(iter1) | $(mean(time1).time) s | $(final_cost_rLM[i])"
    @info "LTMADS: #$(iter2) | $(mean(time2).time) s | $(final_cost_LTMADS[i])"
end
CSV.write(
    "SOd.csv",
    DataFrame(;
        d = matrix_sizes,
        dim = manifold_dimensions,
        n = num_columns,
        t1 = mean_time_rLM,
        t2 = mean_time_LTMADS,
        f1 = final_cost_rLM,
        f2 = final_cost_LTMADS,
        iter1 = iterations_rLM,
        iter2 = iterations_LTMADS,
    )
)
