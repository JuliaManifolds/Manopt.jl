using Manifolds, Manopt, LinearAlgebra, Random

M = Sphere(2)
p = rand(M)

TpM = TangentSpace(M, p)

metric = (x, y) -> inner(M, p, x, y)

A = (I-p*p') * [5 -1 -1; -1 5 -1; -1 -1 5] 

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

rec = get_record(res)












