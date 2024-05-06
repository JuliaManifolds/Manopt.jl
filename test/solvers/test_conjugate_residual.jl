using Manifolds, Manopt, LinearAlgebra, Random



M = Manifolds.Sphere(2)
p = [0.0, 0.0, 1.0]
TpM = TangentSpace(M, p)

A = (M, p, X) -> [2.0 -1.0 0.0; -1.0 2.0 -1.0; 0.0 -1.0 2.0]*X
b = [2.0, -1.0, 0.0]

f(M, p) = 1
grad_f(M, p) = zeros(3)
Hess_f(M, p, X) = zeros(3)

g(M, p) = p
grad_g(M, p) = I

cmo = ConstrainedManifoldObjective(
    ManifoldHessianObjective(f, grad_f, Hess_f), g, grad_g)

o = Manopt.SymmetricLinearSystemObjective(cmo, A, b)

record = [:Iterate]

res = conjugate_residual(
    TpM,
    o,
    [0.0,1.0,0.0];
    stop=StopWhenGradientNormLess(1e-5) | StopAfterIteration(5),
    record=record,
    return_state=true,
)

rec = get_record(res)
