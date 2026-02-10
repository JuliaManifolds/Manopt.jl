using Manopt, Manifolds, LinearAlgebra, Test, Chairmarks

Rxy(α) = [cos(α) sin(α) 0.0; -sin(α) cos(α) 0; 0 0 1]
Rxz(α) = [cos(α)  0.0 sin(α); 0 1 0; -sin(α) 0 cos(α)]
Ryz(α) = [1.0 0 0; 0 cos(α) sin(α); 0 -sin(α) cos(α)]

M = Rotations(3)
# We generate a set of points that are “opposite” each other such that the mean is still I
pts = [
    Matrix{Float64}(I, 3, 3),
    Rxy(0.25) * Rxz(0.05) * Ryz(-0.125),
    Rxy(-0.25) * Rxz(-0.05) * Ryz(0.125),
    Rxy(-0.05) * Rxz(0.125) * Ryz(-0.25),
    Rxy(0.05) * Rxz(-0.125) * Ryz(0.25),
    #outliers
    #Rxy(0.125)*Rxz(0.25)*Ryz(0.05),
    #Rxy(-0.125)*Rxz(0.25)*Ryz(0.05),
]
# M = Rotations(4)
# pts = rand(M, 5)
p0 = copy(M, pts[2])
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

qc = median(M, pts)
cost(M, p) = sum(distance(M, p, q) for q in pts)


# Default Residual CG on this approach – works but probably allocates a bit too much (matrices coordinates/vector...)
q1 = LevenbergMarquardt(
    M, [f], p0;
    β = 2.0, η = 0.01, damping_term_min = 1.0e-5, ε = 1.0e-1, α_mode = :Strict,
    robustifier = [HuberRobustifier()],
    debug = [:Iteration, :Cost, " ", :Change, " ", :damping_term, "\n", :Stop],
    stopping_criterion = StopWhenGradientNormLess(1.0e-16) | StopAfterIteration(100)
)
# ... but works
@info "Cost of median (qc) $(cost(M, qc)), Cost of LM (q1): $(cost(M, q1)), difference (of q1 - qc): $(cost(M, q1) - cost(M, qc))"

#=
q2 = LevenbergMarquardt(
    M, [f], p0;
    β = 8.0, η = 0.2, damping_term_min = 1.0e-5,
    robustifier = [0.05 ∘ HuberRobustifier()],
    debug = [:Iteration, :Cost, " ", :damping_term, "\n", :Stop],
    sub_state = CoordinatesNormalSystemState(M),
)
# ... but works
@info "Cost of mean (qc) $(cost(M, qc)), Cost of LM (q2): $(cost(M, q2)), difference (of q2 - qc): $(cost(M, q2) - cost(M, qc))"

q1b = copy(M, p0)

(@b LevenbergMarquardt!(M, [f], q1b; β = 8.0, η = 0.2, damping_term_min = 1.0e-5, robustifier = [0.05 ∘ HuberRobustifier()])) |> repr |> println
@info distance(M, q1, q1b)

q2b = copy(M, p0)

(
    @b LevenbergMarquardt!(
        M, [f], q2b;
        β = 8.0, η = 0.2, damping_term_min = 1.0e-5, robustifier = [0.05 ∘ HuberRobustifier()], sub_state = CoordinatesNormalSystemState(M),
    )
) |> repr |> println

@info distance(M, q2, q2b)
=#
