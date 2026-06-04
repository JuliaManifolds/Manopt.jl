using Chairmarks, Manopt, Manifolds, LinearAlgebra

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

"""
    f(M, p; i, A, B, robustifier = )

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
DFi(M, p, X; i, A, B) = - X * B[:, i]
DFi!(M, y, p, X; i, A, B) = (y .= - X * B[:, i])

raw"""
    adjointDFi(M, p, y; i, A, B)
    adjointDFi!(M, X, p, y; i, A, B)

For given matrices ``A, B ∈ ℝ^{d,n}`` compute the adjoint differential of the residual of the ith column
with respect to the rotation ``p`` that is, ``y ∈ ℝn`` but mapping into the tangent space at p
This is also referred to as the Jacobian

Since the Euclidean adoint of DF_i is here just ``-yb_i^{\mathrm{T}}``,
the only thing left is to project onto the tangent space, for which we can employ
`project(M, p, X)`for the Grassmann manifold here.

```math
D^*F_i(p)[y] = (I-pp^{\mathrm{T}})(-yb_i^{\mathrm{T}})
```

This is computed in-place of `X`.
"""
adjointDFi(M, p, y; i, A, B) = project(M, p, - y * B[:, i]')
adjointDFi!(M, X, p, y; i, A, B) = project!(M, X, p, - y * B[:, i]')

#
#
# A first experiment

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

d = 30
k = 25
A = generate_data(d)
n = size(A, 2) # number of summands in the vectorial cost sum
p_star = Matrix{Float64}(I,d,d)
for j = 1:(d-1)
    global p_star *= rotation_matrix(d, j, j+1, π / (4*(d-1)))
end
p_star = p_star
B = copy(A)
# and generate a few outliers
B[2, 4] += 0.1
B[3, 1] += 0.1
B[3, 2] += 0.1
B[1, 6] += 0.1
# Because we can build a mask
B = p_star' * B
B = B[1:k,:]
p_star = p_star[:,1:k]

# Hence on the mask we can measure the actual reconstruction error – or looking at the distance to p_star

# the vectorial cost
function F(M, p)
    return [Fi(M, p; i = i, A = A, B = B) for i in 1:n]
end

# start simple: Allocating: we take each single Fi as one component, so we use
# a vector of Differential functions

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
rs = [ 1.0e-7 ∘ HuberRobustifier() for _ in 1:n ]

# M = Stiefel(d,k)
M = Grassmann(d,k)

# Start with the identity
p0 = Matrix{Float64}(I, d, d)
p0 = p0[:,1:k]
# Least Squares Hubertized
p1 = LevenbergMarquardt(
    M, vgfs, p0; robustifier = rs,
    damping_increase_factor = 4.0, candidate_acceptance_threshold = 0.2, damping_term_min = 1.0e-7,
    debug = [
        # :Iteration, (:Cost, "f(x): %8.8e "), :damping_term, :GradientNorm, "\n",
        :Stop,
    ],
)
@info "d=$d    k=$k    n=$n"
@info "LM time"
time1 = @be LevenbergMarquardt(
    $M, $vgfs, $p0; robustifier = $rs,
    damping_increase_factor = 4.0, candidate_acceptance_threshold = 0.2, damping_term_min = 1.0e-7,
) samples=5 evals=3
show(stdout, MIME"text/plain"(),time1)
println()
if M isa Stiefel
    p2 = mesh_adaptive_direct_search(M, (M, p) -> f(M, p; A = A, B = B), p0; debug = [:Stop])
    @info "LTMADS time"
    time2 = @be mesh_adaptive_direct_search($M, $((M, p) -> f(M, p; A = A, B = B)), $p0)
    println()
else
    # Problem with Grassmann: LTMADS needs an ONB, and we do not have that for Gr
    p2 = NelderMead(M, (M, p) -> f(M, p; A = A, B = B); debug = [:Stop])
    @info "NelderMead time"
    time2 = @be NelderMead($M, $((M, p) -> f(M, p; A = A, B = B))) samples=5 evals=3
end
show(stdout, MIME"text/plain"(),time2)
println()
@info "Solution difference: $(distance(M, p1, p2)); costs LM: $(f(M, p1; A = A, B = B)) LTMADS: $(f(M, p2; A = A, B = B))"
