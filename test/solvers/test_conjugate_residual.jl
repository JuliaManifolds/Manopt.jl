using Manifolds, Manopt, LinearAlgebra, Random

M = Sphere(2)
p = rand(M)
X = rand(M, vector_at = p)

TpM = TangentSpace(M, p)

B = [2 1 0; 1 2 1; 0 1 2]

function A(x)
    return B*x
end

b = [1, 1, 1]

metric = (x, y) -> inner(M, p, x, y)

mho = ManifoldHessianObjective(
        (TpM, x)    -> metric(X, A(x)) - metric(b, x),
        (TpM, x)    -> A(x) - b,
        (TpM, x, y) -> A(y)
)







