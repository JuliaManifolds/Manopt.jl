
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

robustifier = 0.05 ∘ HuberRobustifier()

ε0 = 0.5
α_mode = :Default

q2 = LevenbergMarquardt(
    M, [f], p0;
    β = 8.0, η = 0.2, damping_term_min = 1.0e-5, ε = ε0, α_mode = α_mode,
    robustifier = [robustifier],
    debug = [:Iteration, :Cost, " ", :damping_term, "\n", :Stop],
    sub_state = CoordinatesNormalSystemState(M),
)

function test_lm(pk; scale_damping = ε0)

    @show "Testing LM surrogate at point $pk with damping scale $scale_damping"
    # original objective

    F_k = F(M, pk)
    vi = sum(abs2, F_k)
    @show "vi: $vi"
    (a, ap, app) = get_robustifier_values(robustifier, vi)
    @show "a: $a, a': $ap, a'': $app"
    residual_scaling, operator_scaling = Manopt.get_LevenbergMarquardt_scaling(ap, app, vi, ε0, α_mode)
    oc = a/2

    @show "Residual scaling: $residual_scaling, Operator scaling: $operator_scaling"
    @show "Original cost: $oc"

    # surrogate: σ_k with truncated α
    # this one is not guaranteed to match the desired grad and Hess

    # value at which we want to compute the surrogate
    cX = [0.0, 0.0, 0.0]

    y_k = residual_scaling * F_k

    @show y_k
    
    C_k = sqrt(ap) * (I - operator_scaling * (F_k * F_k'))
    J_k = JF(M, pk)
    function calc_surrogate_cost(cX)
        @show "J[X]: ", J_k * cX
        LX = C_k * J_k * cX
        @show "LX: $LX"

        σ_k_X = 0.5 * norm(y_k + LX)^2
        return σ_k_X
    end
    σ_k_X0 = calc_surrogate_cost([0.0, 0.0, 0.0])

    @show "Surrogate cost at 0 vector: $σ_k_X0"
    # surrogate: penalized σ_k with truncated α

    penalty = scale_damping * oc

    @show "Penalty: $penalty"
    μ_k = σ_k_X + penalty * norm(cX)^2 / 2

    @show "Penalized surrogate cost at $cX: $μ_k"

    @show "Penalized normal equation:"
    A = J_k' * C_k' * C_k * J_k + penalty * I
    @show J_k
    @show C_k
    b = -J_k' * C_k' * y_k
    c = A \ b
    @show A
    @show b
    @show "Solution: $c, surrogate cost: $(calc_surrogate_cost(c)), penalty: $penalty, total: $(calc_surrogate_cost(c) + penalty * norm(c)^2 / 2))"

    return penalty
end
