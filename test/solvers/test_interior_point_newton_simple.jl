using Manopt, Manifolds, Plots

M = Euclidean(1)
f(M, p) = p[1]^2
grad_f(M, p) = 2 * p
Hess_f(M, o, X) = [2;;]

g(M, p) = [-p[1] - 1] # -p+1 <= 0 <=> 1 <= p
grad_g(M, p) = [-1]
Hess_g(M, p, X) = [0;;]

p_0 = [2.0]

st = interior_point_Newton(
    M,
    f,
    grad_f,
    Hess_f,
    p_0;
    g=g,
    grad_g=grad_g,
    Hess_g=Hess_g,
    stopping_criterion=StopAfterIteration(5) | StopWhenChangeLess(1e-12),
    stepsize=ConstantStepsize(0.01),
    debug=[
        :Iteration,
        " | ",
        :Iterate,
        " | ",
        :Cost,
        " | ",
        :Stepsize,
        " | ",
        :Change,
        "\n\t",
        :GradientNorm,
        " ",
        :Feasibility,
        " ",
        :σ,
        " ",
        :ρ,
        "\n",
        :Stop,
    ],
    return_state=true,
    return_objective=true,
)
p_1 = get_solver_result(st)
cmo = st[1]
s = get_state(st[2])
m = 1
n = 0
N = M × ℝ^m × ℝ^n × ℝ^m
cmo = res[1]
q = rand(N)
q[N, 1] = get_iterate(s)
q[N, 2] = s.μ
q[N, 4] = s.s

#
# LagrangianCost
L = LagrangianCost(cmo, s.μ, s.λ)
grad_L = LagrangianGradient(cmo, s.μ, s.λ)
check_gradient(M, L, grad_L, p_0, [1.0]; plot=true, error=:info)

#
# KKTVectorField
F = KKTVectorFieldNormSq(cmo)
grad_F = KKTVectorFieldNormSqGradient(cmo)
X = zero_vector(N, q)
X[N, 1] = [1.0]
check_gradient(N, F, grad_F, q, X; plot=true, error=:info)
