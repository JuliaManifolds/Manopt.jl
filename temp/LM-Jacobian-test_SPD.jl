using Manopt, Manifolds, LinearAlgebra, Test, Chairmarks

M = SymmetricPositiveDefinite(2)
# We generate a set of points that are “opposite” each other such that the mean is still I
e = Matrix{Float64}(I, 2, 2)
pts = [
    e, exp(M, e, [0.2 0.1; 0.1 0.0]), exp(M, e, -[0.2 0.1; 0.1 0.0]),
    exp(M, e, [0.2 -0.1; -0.1 0.0]), exp(M, e, -[0.2 -0.1; -0.1 0.0]),
]
# M = Rotations(4)
# pts = rand(M, 5)
p0 = copy(M, pts[4])
# We do a full function approach here

F(M, p) = [distance(M, p, q) for q in pts]
function JF(M, p, B::AbstractBasis = DefaultOrthonormalBasis())
    d = manifold_dimension(M)
    n = length(pts)
    J = zeros(n, d)
    for (i, q) in enumerate(pts)
        dpq = distance(M, p, q)
        if dpq > 0
            J[i, :] = -1 / dpq .* get_coordinates(M, p, log(M, p, q), B)
        end
    end
    return J
end

f = VectorGradientFunction(
    F, JF, length(pts);
    evaluation = AllocatingEvaluation(),
    function_type = FunctionVectorialType(),
    jacobian_type = CoefficientVectorialType(DefaultOrthonormalBasis())
)

qc = mean(M, pts)
cost(M, p) = 0.5 * sum(distance(M, p, q)^2 for q in pts)


# Default Residual CG on this approach – works but probably allocates a bit too much (matrices coordinates/vector...)
q1 = LevenbergMarquardt(
    M, [f], p0;
    β = 8.0, η = 0.2, damping_term_min = 1.0e-5,
    robustifier = [IdentityRobustifier()],
    debug = [:Iteration, :Cost, " ", :Change, " ", :damping_term, "\n", :Stop],
)
# ... but works
@info "Cost of mean (qc) $(cost(M, qc)), Cost of LM (q1): $(cost(M, q1)), difference (of q1 - qc): $(cost(M, q1) - cost(M, qc))"

q2 = LevenbergMarquardt(
    M, [f], p0;
    β = 8.0, η = 0.2, damping_term_min = 1.0e-5,
    robustifier = [IdentityRobustifier()],
    debug = [:Iteration, :Cost, " ", :damping_term, "\n", :Stop],
    sub_state = CoordinatesNormalSystemState(M),
)
# ... but works
@info "Cost of mean (qc) $(cost(M, qc)), Cost of LM (q2): $(cost(M, q2)), difference (of q2 - qc): $(cost(M, q2) - cost(M, qc))"

q1b = copy(M, p0)

(@b LevenbergMarquardt!(M, [f], q1b; β = 8.0, η = 0.2, damping_term_min = 1.0e-5, robustifier = [IdentityRobustifier()])) |> repr |> println
@info distance(M, q1, q1b)

q2b = copy(M, p0)

(
    @b LevenbergMarquardt!(
        M, [f], q2b;
        β = 8.0, η = 0.2, damping_term_min = 1.0e-5, robustifier = [IdentityRobustifier()], sub_state = CoordinatesNormalSystemState(M),
    )
) |> repr |> println

@info distance(M, q2, q2b)
