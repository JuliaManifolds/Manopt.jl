using Manifolds, Manopt, LinearAlgebra, Random
Random.seed!(42);

A = [2 0 0; 0 2 0; 0 0 1]

function f(M, p)
    return 0.5*p'*A*p
end

function grad_f(M, p)
    return (I - p*p')*A*p
end

function Hess_f(M, p, X)
    return (I - p*p')*A*X - f(M, p)*X
end

function g(M, p)
    return -[p[3]]
end

function grad_g(M, p)
    return [0 0 -1] * (I - p*p')
end

M = Sphere(2)

p1 = 2*rand()-1
p2 = sqrt(1-p1^2)*rand()
p3 = sqrt(1-p1^2-p2^2)

p = [p1, p2, p3]

mho = ManifoldHessianObjective(f, grad_f, Hess_f)
cmo = ConstrainedManifoldObjective(mho, g, grad_g)

interior_point_Newton!(M, cmo, p)