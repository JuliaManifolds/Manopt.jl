using Manopt, Manifolds, Plots

M = Euclidean(1)
p = [2.0]
X = [4.0]

f(M, p) = p[1]^2
grad_f(M, p) = 2 * p
Hess_f(M, o, X) = 2 * X
cf = check_gradient(M, f, grad_f, p_0, X; error=:info)
cf2 = check_Hessian(M, f, grad_f, Hess_f, p_0, X; error=:info)

g(M, p) = [-p[1] + 1] # -p+1 <= 0 <=> 1 <= p
grad_g(M, p) = [[-1.0]]
Hess_g(M, p, X) = [[0.0]]

# Chekc first (ond only) component
_g(M, p) = -p[1] + 1 # -p+1 <= 0 <=> 1 <= p
_grad_g(M, p) = [-1.0]
_Hess_g(M, p, X) = [0.0]
cf = check_gradient(M, _g, _grad_g, p_0, X; error=:info)
cf2 = check_Hessian(M, _g, _grad_g, _Hess_g, p_0, X; error=:info)

res = interior_point_Newton(
    M,
    f,
    grad_f,
    Hess_f,
    p;
    g=g,
    grad_g=grad_g,
    Hess_g=Hess_g,
    stopping_criterion=StopAfterIteration(0) | StopWhenChangeLess(1e-12),
    #stepsize=ConstantStepsize(0.01),
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
cmo = res[1]
st = get_state(res[2])
m = 1
n = 0
N = M × ℝ^m × ℝ^n × ℝ^m
cmo = res[1]
p = get_iterate(st)
λ = st.λ
μ = st.μ
s = st.s

q = rand(N)
q[N, 1] = p
q[N, 2] = μ
q[N, 4] = s

#
# LagrangianCost
L = LagrangianCost(cmo, μ, λ)
grad_L = LagrangianGradient(cmo, μ, λ)
Hess_L = LagrangianHessian(cmo, μ, λ)
cL = check_gradient(M, L, grad_L, p, X; plot=true, error=:info)
cL2 = check_Hessian(M, L, grad_L, Hess_L, p, X; error=:info)
#
#
#
K = KKTVectorField(cmo)
JK = KKTVectorFieldJacobian(cmo)
Y = [2.0]
Z = Vector{Float64}[] # h, not necessary
W = [2.0]
JsK = KKTVectorFieldAdjointJacobian(cmo)
JsK(N, q, qX)

qX = zero_vector(N, q)
qX[N, 1], qX[N, 2], qX[N, 3], qX[N, 4] = X, Y, Z, W
K(N, q)
JK(N, q, qX)

#
#
# Condensed with barrier β
β = 3.5
CK = CondensedKKTVectorField(cmo, μ, s, β)
CKJ = CondensedKKTVectorFieldJacobian(cmo, μ, s, β)
Tq2N2 = get_manifold(st.sub_problem)
N2 = base_manifold(Tq2N2)
q2 = get_manifold(st.sub_problem).point
q2[N2, 1] = p
q2X = zero_vector(N2, q2)
q2X[N, 1] = X
A = CKJ(N2, q2, q2X)
b = CK(N2, q2)

#
# KKTVectorField Norm Sq
F = KKTVectorFieldNormSq(cmo)
grad_F = KKTVectorFieldNormSqGradient(cmo)
qY = zero_vector(N, q)
qY[N, 1] = [1.0]
check_gradient(N, F, grad_F, q, qY; plot=true, error=:info)
