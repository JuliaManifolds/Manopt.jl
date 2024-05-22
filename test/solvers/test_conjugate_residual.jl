using Manifolds, Manopt, LinearAlgebra, Random

<<<<<<< HEAD

M = Manifolds.Sphere(2) × ℝ^2
p = rand(M)
=======
M = Manifolds.Sphere(2)
p = [0.0, 0.0, 1.0]
>>>>>>> 360bc994c59c5eb51df7269259dfc54c4e0b351a
TpM = TangentSpace(M, p)

A = (M, p, X) -> [2.0 -1.0 0.0; -1.0 2.0 -1.0; 0.0 -1.0 2.0] * X
b = (M, p) -> [2.0, -1.0, 0.0]

o = Manopt.SymmetricLinearSystemObjective(A, b)

record = [:Iterate]

res = conjugate_residual(
    TpM,
    o,
    [0.0, 1.0, 0.0];
    stop=StopWhenGradientNormLess(1e-5) | StopAfterIteration(5),
    record=record,
    return_state=true,
)

rec = get_record(res)
