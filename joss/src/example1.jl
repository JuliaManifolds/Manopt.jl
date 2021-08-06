## First example block
using Manopt, Manifolds, LinearAlgebra, Random
Random.seed!(42)
M = Sphere(2)
n = 100
pts = [ normalize(rand(3)) for _ in 1:n ]

F(M, y) = sum(1/(2*n) * distance.(Ref(M), pts, Ref(y)).^2)
gradF(M, y) = sum(1/n * grad_distance.(Ref(M), pts, Ref(y)))

x_mean = gradient_descent(M, F, gradF, pts[1])

## Second example block
o = gradient_descent(M, F, gradF, pts[1],
    debug=[:Iteration, " | ", :Change, " | ", :Cost, "\n", :Stop],
    record=[:x, :Change, :Cost],
    return_options=true
)
xMean3 = get_solver_result(o)
values = get_record(o) # vector with a tuple per iteration
iterates = get_record(o, :Iteration, :x) # from each iteration get the recorded iterate x.

## Export
using Colors
TolVibrantOrange = RGBA{Float64}(colorant"#EE7733")
TolVibrantTeal = RGBA{Float64}(colorant"#009988")
asymptote_export_S2_signals(joinpath( @__DIR__, "img/MeanIllustr.asy");
    points=[[xMean,], pts],
    colors=Dict(:points => [TolVibrantOrange, TolVibrantTeal]),
    dot_size=3.5, camera_position=(1.0, 0.75, 0.5),
)
render_asymptote(joinpath( @__DIR__, "img/MeanIllustr.asy"); render=4)  #src
