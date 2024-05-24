using Manifolds, Manopt, LinearAlgebra, Random, Plots

Random.seed!(42)
A = -Symmetric(rand(3,3))
# A = [2.0 -1.0 0.0; -1.0 2-0 -1.0; 0.0 -1.0 2.0]

f(M, p) = 0.5 * p' * A * p
grad_f(M, p) = (I - p * p') * A * p
Hess_f(M, p, X) = (I - p * p') * A * X - f(M, p) * X

g(M, p) = -p
grad_g(M, p) = -(I - p * p')
M = Manifolds.Sphere(2)

x = rand()
y = sqrt(1 - x^2) * rand()
z = sqrt(1 - x^2 - y^2)

# p_0 = [x, y, z]
p_0 = (1.0 / (sqrt(3.0))) .* [1.0, 1.0, 1.0]

record = [:Iterate]

res = interior_point_Newton(
    M,
    f,
    grad_f,
    Hess_f,
    p_0;
    g=g,
    grad_g=grad_g,
    stop=StopAfterIteration(200) | StopWhenChangeLess(1e-9),
    debug=[:Iteration, " | ", :Cost, " | ", :Stepsize, " | ", :Change, " ", :GradientNorm, " ", :Feasibility,"\n", :Stop],
    record=record,
    return_state=true,
)

rec = get_record(res)

prepend!(rec, [p_0])

rec .+= 0.005 * rec

#-------------------------------------------------------------------------------------------------#

using GLMakie, Makie, GeometryTypes

n = 30

π1(x) = x[1]
π2(x) = x[2]
π3(x) = x[3]

h(x) = [cos(x[1])sin(x[2]), sin(x[1])sin(x[2]), cos(x[2])]

U = [[θ, ϕ] for θ in LinRange(0, 2π, n), ϕ in LinRange(0, π, n)]
V = [[θ, ϕ] for θ in LinRange(0, π / 2, n), ϕ in LinRange(0, π / 2, n)]

pts = h.(U)
pts_ = h.(V)

f_ = p -> f(M, p)

s = maximum(f_.(pts)) - minimum(f_.(pts))
s_ = maximum(f_.(pts_)) - minimum(f_.(pts_))

x1 = π1.(pts)
x2 = π2.(pts)
x3 = π3.(pts)

x1_ = π1.(pts_)
x2_ = π2.(pts_)
x3_ = π3.(pts_)

grads = grad_f.(Ref(M), pts)
normgrads = grads ./ norm.(grads)

v1 = π1.(normgrads)
v2 = π2.(normgrads)
v3 = π3.(normgrads)

scene = Scene();

cam3d!(scene)

range_f = (minimum(f_.(pts)), maximum(f_.(pts)))

surface!(
    scene,
    x1,
    x2,
    x3;
    color=f_.(pts),
    colormap=(:temperaturemap, 0.4),
    backlight=1.0f0,
    colorrange=range_f,
)

surface!(
    scene,
    x1_,
    x2_,
    x3_;
    color=f_.(pts_),
    colormap=(:temperaturemap, 1.0),
    backlight=1.0f0,
    colorrange=range_f,
)

scatter!(scene, π1.(rec), π2.(rec), π3.(rec); color=:black)

# Makie.arrows!(
#     scene, vec(x1), vec(x2), vec(x3), vec(v1), vec(v2), vec(v3);
#     arrowsize   = vec(norm.(grads))/80,
#     arrowcolor  = vec(norm.(grads)),
#     linecolor   = vec(norm.(grads)),
#     linewidth   = vec(norm.(grads))/200,
#     lengthscale = 0.08,
#     colormap    = :reds
# )

scene