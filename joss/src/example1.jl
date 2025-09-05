## First example block
using Manopt, Manifolds, LinearAlgebra, Random
Random.seed!(42)
M = Sphere(2)
n = 40
p = 1 / sqrt(3) .* ones(3)
B = DefaultOrthonormalBasis()
pts = [exp(M, p, get_vector(M, p, 0.425 * randn(2), B)) for _ in 1:n]

F(M, y) = sum(1 / (2 * n) * distance.(Ref(M), pts, Ref(y)) .^ 2)
gradF(M, y) = sum(1 / n * grad_distance.(Ref(M), pts, Ref(y)))

x_mean = gradient_descent(M, F, gradF, pts[1])

euclidean_mean = mean(pts)
print("Norm of Euclidean mean:", norm(euclidean_mean), "\n\n")
euclidean_mean_normed = euclidean_mean / norm(euclidean_mean)

## Second example block
o = gradient_descent(
    M,
    F,
    gradF,
    pts[1];
    debug = [:Iteration, " | ", :Change, " | ", :Cost, "\n", :Stop],
    record = [:x, :Change, :Cost],
    return_state = true,
)
x_mean2 = get_solver_result(o)
values = get_record(o) # vector with a tuple per iteration
iterates = get_record(o, :Iteration, :x) # from each iteration get the recorded iterate x.

## Export with Paul Tols colors https://personal.sron.nl/~pault/
using Colors
TolVibrantOrange = RGBA{Float64}(colorant"#EE7733")
TolVibrantTeal = RGBA{Float64}(colorant"#009988")
TolVibrantCyan = RGBA{Float64}(colorant"#33BBEE")
asymptote_export_S2_signals(
    joinpath(@__DIR__, "img/MeanIllustr.asy");
    points = [[x_mean], [euclidean_mean_normed], pts],
    colors = Dict(:points => [TolVibrantOrange, TolVibrantCyan, TolVibrantTeal]),
    dot_sizes = [3.5, 2.5, 3.5],
    camera_position = (0.7, 0.7, 0.5),
)
render_asymptote(joinpath(@__DIR__, "img/MeanIllustr.asy"); render = 4)  #src
