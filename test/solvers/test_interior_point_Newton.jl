using Manifolds, Manopt, LinearAlgebra, Random

A = [2 -1 -1; -1 2 -1; -1 -1 2]

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

p_x = rand()
p_y = sqrt(1-p_x^2)*rand()
p_z = sqrt(1-p_x^2-p_y^2)

p_0 = [p_x, p_y, p_z]

record = [:Iterate]

res = interior_point_Newton(
    M, f, grad_f, Hess_f, p_0; g=g, grad_g=grad_g, record=record, return_state=true
    )

rec = get_record(res)

