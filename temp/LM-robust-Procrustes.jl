using Manopt, Manifolds, LinearAlgebra

raw"""
    generate_data(d)

Generate a data matrix ``A ∈ ℝ^{d× n}`` in a deterministic way,
here by using ``n = \frac{d(d-1)}{2}`` columns.
"""
function generate_data(d)
    # (1) push the first unit vector scaled by 1/2
    n = Int(d*(d+1)/2)
    A = zeros(d,n)
    A[1,1] = 1.0
    k = 2
    dir = zeros(d)
    for i = 2:d # for each following dimension, set point on the diagonal of the preceding dimensions
        dir[1:i] .= 1/sqrt(i)
        dir[(i+1):end] .= 0
        v = range(0.0, 1.0, i+2)[2:end-1]
        for vi in v
            A[:, k] .= vi*dir
            k = k+1
        end
    end
    return A
end

skew(A) = 0.5 .* (A-A')

"""
    f(M, p; i, A, B, robustifier = )

For given matrices ``A, B ∈ ℝ^{d,n}`` compute the robust Procrustes cost

```math
F_i(p) = (A - pB)_i = a_i - pb_i
```
"""
f(M, p; A, B) = sum( norm(A[:,i] - p*B[:,i]) for i ∈ 1:size(A,2) )

"""
    Fi(M, p; i, A, B)

For given matrices ``A, B ∈ ℝ^{d,n}`` compute the residual of the ith column

```math
F_i(p) = (A - pB)_i = a_i - pb_i
```
"""
Fi(M, p; i, A, B) = A[:,i] - p*B[:,i]

"""
    DFi!(M, y, p, X; i, A, B)

For given matrices ``A, B ∈ ℝ^{d,n}`` compute the differential of the residual of the ith column
with respect to the rotation ``p`` that is, ``X ∈ 𝔰𝔬(d)`` which reads

```math
\\mathcal J_{F_i}(p)[X] = DF_i(p)[X] = -pXb_i
```

This is computed in-place of `y`

"""
DFi!(M, y, p, X; i, A, B) = (y .= - p*X*B[:,i])

DFi(M, p, X; i, A, B) = - p*X*B[:,i]


raw"""
    adjointDFi!(M, X, p, y; i, A, B)

For given matrices ``A, B ∈ ℝ^{d,n}`` compute the adjoint differential of the residual of the ith column
with respect to the rotation ``p`` that is, ``y ∈ ℝn`` but mapping into the Lie algebra ``𝔰𝔬(d)``.
This is also referred to as the Jacobian

```math
D*{F_i}(p)[y] = -\mathrm{skew}(p^{\mathrm{T}}yb_i^{\mathrm{T}})
```

This is computed in-place of `X`.
"""
adjointDFi!(M, a, p, y; i, A, B) = (a .= - skew(p'*y*B[:,i]'))

adjointDFi(M, p, y; i, A, B) = - skew(p'*y*B[:,i]')


#
#
# A first experiment

d = 3
A = generate_data(3)
n = size(A,2) # number of summands in the vectorial cost sum
p_star = Matrix{Float64}(I, d, d)
# Lets take a very simple rotation for now
p_star[1,1] = cos(π/4)
p_star[2,2] = cos(π/4)
p_star[2,1] = -sin(π/4)
p_star[1,2] = sin(π/4)

B = copy(A)
# and generate a few outliers
B[2,4] += 0.1
B[3,1] += 0.1
B[3,2] += 0.1
B[1,6] += 0.1
# Because we can build a mask
mask = B .== A
# then rotate
B = p_star'*B

# Hence on the mask we can measure the actual reconstruction error – or looking at the distance to p_star

# the vectorial cost
function F(M,p)
    return [Fi(M, p; i=i, A=A, B=B) for i ∈ 1:n]
end

# start simple: Allocating: we take each single Fi as one component, so we use
# a vector of Differential functions

vgfs = [
    VectorDifferentialFunction(
        (M,p) -> Fi(M, p; i=i, A=A, B=B),
        (M, p, X) -> DFi(M, p, X; i=i, A=A, B=B),
        (M, p, y) -> adjointDFi(M, p, y; i=i, A=A, B=B),
        d;
        evaluation = AllocatingEvaluation(),
        function_type = FunctionVectorialType(),
        jacobian_type = FunctionVectorialType(),
        adjoint_jacobian_type = FunctionVectorialType()
    ) for i=1:n
]
rs = [ 1.0e-4 ∘ HuberRobustifier() for _ = 1:n ]

M = Rotations(3)
# Start with the identity
p0 = Matrix{Float64}(I,d,d)

# Least Squares
p_1 = LevenbergMarquardt(
    M, vgfs, p0;
    damping_increase_factor = 2.0, candidate_acceptance_threshold = 0.2, damping_term_min = 1.0e-5,
    damping_increase_threshold = 0.2,
    damping_reduction_threshold = 0.5,
    scaling_threshold = 1.0e-1, scaling_mode = :Strict,
    robustifier = rs,
    debug = [:Iteration, (:Cost, "f(x): %8.8e "), :damping_term, :Change, "\n", :Stop],
)
