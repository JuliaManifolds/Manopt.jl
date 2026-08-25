using Manopt, Manifolds, Random, LinearAlgebra
Random.seed!(42)
d = 4
M = Sphere(d - 1)
v0 = project(M, [ones(2)..., zeros(d - 2)...])
Z = v0 * v0'
#Cost and gradient
f(M, p) = -tr(transpose(p) * Z * p) / 2
grad_f(M, p) = project(M, p, -transpose.(Z) * p / 2 - Z * p / 2)
# Constraints
g(M, p) = -p # now p ≥ 0
mI = -Matrix{Float64}(I, d, d)
# Vector of gradients of the constraint components
grad_g(M, p) = [project(M, p, mI[:, i]) for i in 1:d]
p0 = project(M, [ones(2)..., zeros(d - 3)..., 0.1])

debug = [:Iteration, :Time, ", ", :Cost, " | ", (:ϵ,"ϵ: %.8f"), 25, "\n", :Stop]
p1 = exact_penalty_method(
    M, f, grad_f, p0; g=g, grad_g=grad_g,
    debug = debug
);
