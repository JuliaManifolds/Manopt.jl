using Manopt, Manifolds, ManifoldDiff#, ManoptExamples

#
# L2-TV with 2 points: data ∈ Hyperbolic(2)^2
# argmin_q d^2(data,p) + d(p_1,p_2)
#
# Since we have a gradient for the (components of) the fist summand and a prox for the second

M = Hyperbolic(2)^2
data = [-sqrt(2) sqrt(2); sqrt(2) -sqrt(2); sqrt(5) sqrt(5)]

# smooth and nonsmooth parts
g(M, p) = distance(M, p, data)^2
grad_g(M, p) = ManifoldDiff.grad_distance(M, data, p, 2)
h(M, p) = distance(M.manifold, p[M, 1], p[M, 2])
#=
function prox_h(M, λ, p)
    q = copy(M, p)
    (q[M, 1], q[M, 2]) = ManoptExamples.prox_Total_Variation(
        M.manifold, λ, (p[M, 1], p[M, 2])
    )
    return q
end
f(M, p) = g(M, p) + h(M, p)

q = proximal_gradient_method(
    M, f, grad_g, prox_h, data; debug=[:Iteration, :Cost, :Change, "\n", :Stop]
)
=#
