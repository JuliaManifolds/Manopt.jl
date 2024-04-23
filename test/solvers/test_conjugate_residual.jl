using Manifolds, Manopt, LinearAlgebra, Random

M = Sphere(2)
p = rand(M)

TpM = TangentSpace(M, p)



metric = (x, y) -> inner(M, p, x, y)

A = (I-p*p') * [2 1 0; 1 2 1; 0 1 2] 

b = rand(TpM)

mho = ManifoldHessianObjective(
        (TpM, x)    -> 1/2 * metric(X, A*x) - metric(b, x),
        (TpM, x)    -> A*x - b,
        (TpM, x, y) -> A*y
)

record = [:Iterate]

res = conjugate_residual(
    TpM, mho, rand(TpM); stop=StopWhenGradientNormLess(1e-5)|StopAfterIteration(100), record=record, return_state=true
    )

x = rand(TpM)












