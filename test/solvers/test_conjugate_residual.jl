using Manifolds, Manopt, LinearAlgebra, Random

M = Manifolds.Sphere(2)
p = rand(M)
TpM = TangentSpace(M, p)

A = (M, p, X) -> [2.0 -1.0 0.0; -1.0 2.0 -1.0; 0.0 -1.0 2.0] * X
b = (M, p) -> [2.0, -1.0, 0.0]

o = Manopt.SymmetricLinearSystemObjective(A, b)

record = [:Iterate]

X0 = rand(TpM)

res = conjugate_residual(
    TpM,
    o,
    X0;
    stop=StopWhenGradientNormLess(1e-5) | StopAfterIteration(20),
    record=record,
    return_state=true,
)

rec = get_record(res)
