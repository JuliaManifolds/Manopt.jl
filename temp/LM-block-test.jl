#
#
# A small test to see how these block based things now work in LM


#
#
# On the same data of unit vectors in R3 this example fits
# * a plane in R3, and hence a problem in R3 (3D)
# * a geodesic curve on the Sphere S2, hence a problem in TS2 (4D)
#
using GLMakie, LinearAlgebra, Manifolds, ManifoldDiff, Manopt, RecursiveArrayTools
#
# A very plain example mainly to check that it is working fine:
# The median

M = Euclidean(3)
pts = [
    [0.0, 0.0, 1.0],
    [sqrt(0.19), 0.0, 0.9],
    [-sqrt(0.19), 0.0, 0.9],
    [0.0, sqrt(0.19), 0.9],
    [0.0, -sqrt(0.19), 0.9],
    [1 / sqrt(2), 1 / sqrt(2), 0.0],
    [-1 / sqrt(2), 1 / sqrt(2), 0.0],
]
p0 = [0.0, 0.0, 78.0]

Fi = [ (M, p) -> distance(M, p, q) for q in pts]
grad_Fi = [ (M, p) -> distance(M, p, q) == 0 ? zero_vector(M, p) : (- log(M, p, q) / distance(M, p, q)) for q in pts]

# Block 1 normal ones
F = VectorGradientFunction(
    Fi[1:5], grad_Fi[1:5], 5;
    evaluation = AllocatingEvaluation(), function_type = ComponentVectorialType(), jacobian_type = ComponentVectorialType()
)
# Block 2 outliers
G = VectorGradientFunction(
    Fi[6:7], grad_Fi[6:7], 2;
    evaluation = AllocatingEvaluation(), function_type = ComponentVectorialType(), jacobian_type = ComponentVectorialType()
)

qc = mean(M, pts)
qR = 1 / length(pts) .* sum(pts)
f(M, p) = sum(distance(M, p, q)^2 for q in pts)
q1 = LevenbergMarquardt(
    M, [F], p0;
    robustifier = [IdentityRobustifier() for _ in 1:2][1:1],
    debug = [:Iteration, :Cost, " ", :damping_term, " ", :Iterate, "\n\n"],
)
@info "---"
@info "Mean cost $(f(M, qc))"
@info "Mean cost (R) $(f(M, qR))"
@info "Cost after optimization $(f(M, q1))"
