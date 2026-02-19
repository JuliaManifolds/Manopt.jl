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
    Rxy(0.125) * Rxz(0.25) * Ryz(0.05),
    Rxy(-0.125) * Rxz(0.25) * Ryz(0.05),
]
p0 = copy(M, pts[2])

Fi = [ (M, p) -> distance(M, p, q) for q in pts]
grad_Fi = [ (M, p) -> distance(M, p, q) == 0 ? zero_vector(M, p) : (- log(M, p, q) / distance(M, p, q)) for q in pts]

# Block s normal ones
Fs = [
    VectorGradientFunction(
            [Fi[i]], [grad_Fi[i]], 1;
            evaluation = AllocatingEvaluation(), function_type = ComponentVectorialType(), jacobian_type = ComponentVectorialType()
        ) for i in eachindex(pts)
]
hr = (1.0e-4) ∘ HuberRobustifier()
hrs = fill(hr, length(Fs))

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

#
#
# TODO: Try an approach with Jacobians in bases - then 2 should be faster

qc = median(M, pts)
cost(M, p) = sum(distance(M, p, q) for q in pts)

# Default Residual CG on this approach – works but probably allocates a bit too much (matrices coordinates/vector...)
q1 = LevenbergMarquardt(
    M, Fs, p0;
    β = 8.0, η = 0.2, damping_term_min = 1.0e-5, ε = 1.0e-1, α_mode = :Strict,
    robustifier = hrs,
    debug = [:Iteration, :Cost, " ", :Change, " ", :damping_term, "\n", :Stop, 25],
)
@info "Cost of median (qc) $(cost(M, qc)), Cost of LM (q1): $(cost(M, q1)), difference (of q1 - qc): $(cost(M, q1) - cost(M, qc))"

q2 = LevenbergMarquardt(
    M, Fs, p0;
    β = 8.0, η = 0.2, damping_term_min = 1.0e-5, ε = 1.0e-1, α_mode = :Strict,
    robustifier = hrs,
    debug = [:Iteration, (:Cost, "f(x): %8.8e "), :damping_term, "\n", :Stop, 25],
    sub_state = CoordinatesNormalSystemState(M),
)
@info "Cost of mean (qc) $(cost(M, qc)), Cost of LM (q2): $(cost(M, q2)), difference (of q2 - qc): $(cost(M, q2) - cost(M, qc))"

q3 = LevenbergMarquardt(
    M, f, p0;
    β = 8.0, η = 0.2, damping_term_min = 1.0e-5, ε = 1.0e-1, α_mode = :Strict,
    robustifier = hr,
    debug = [:Iteration, (:Cost, "f(x): %8.8e "), :damping_term, "\n", :Stop, 25],
    sub_state = CoordinatesNormalSystemState(M),
)
@info "Cost of mean (qc) $(cost(M, qc)), Cost of LM (q3): $(cost(M, q3)), difference (of q3 - qc): $(cost(M, q3) - cost(M, qc))"

q1b = copy(M, p0)

(
    @b LevenbergMarquardt!(
        M, Fs, q1b; β = 8.0, η = 0.2, damping_term_min = 1.0e-5, robustifier = hrs
    )
) |> repr |> println

@info "Distance from alloc q1 to in place q1b (from benchmark) $(distance(M, q1, q1b))"

q2b = copy(M, p0)
(
    @b LevenbergMarquardt!(
        M, Fs, q2b;
        β = 8.0, η = 0.2, damping_term_min = 1.0e-5,
        robustifier = hrs, sub_state = CoordinatesNormalSystemState(M),
    )
) |> repr |> println

@info "Distance from alloc q2 to in place q2b (from benchmark) $(distance(M, q2, q2b))"
