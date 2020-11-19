#
# Minimize the acceleration of a composite Bézier curve on the Sphere S2 with interpolation
#
# This example appeared in Sec. 5.2, second example, of
# > Bergmann, R. and Gousenbourger, P.-Y.: _A variational model for data fitting on manifolds
# > by minimizing the acceleration of a Bézier curve_.
# > Frontiers in Applied Mathematics and Statistics, 2018.
# > doi: [10.3389/fams.2018.00059](https://dx.doi.org/10.3389/fams.2018.00059),
# > arXiv: [1807.10090](https://arxiv.org/abs/1807.10090)
#
using Manopt, Manifolds, Colors, ColorSchemes
results_folder = joinpath(@__DIR__, "Minimize_Acceleration_Bezier")
experiment_name = "Bezier_Interpolation"
asy_export = true #export data and results to asyExport

curve_samples = [range(0, 3; length=101)...] # sample curve for the gradient
curve_samples_plot = [range(0, 3; length=201)...] # sample curve for asy exports

cameraPosition = (-1.0, -0.7, 0.3)
curveColor = RGBA{Float64}(colorant"#000000")
sColor = RGBA{Float64}(colorant"#BBBBBB")
dColor = RGBA{Float64}(colorant"#EE7733") # data Color: Tol Vibrant Orange
pColor = RGBA{Float64}(colorant"#0077BB") # control point data color: Tol Virbant Blue
ξColor = RGBA{Float64}(colorant"#33BBEE") # tangent vector: Tol Vibrant blue
bColor = RGBA{Float64}(colorant"#009988") # inner control points: Tol Vibrant teal
#
# Data
#
M = Sphere(2)
B = artificial_S2_composite_bezier_curve()
cP = de_casteljau(M, B, curve_samples_plot)
# export original data.
if asy_export
    asymptote_export_S2_signals(
        joinpath(results_folder, experiment_name * "-orig.asy");
        curves=[cP],
        points=[get_bezier_junctions(M, B), get_bezier_inner_points(M, B)],
        tVectors=[[
            Tuple(a)
            for
            a in
            zip(get_bezier_junctions(M, B, true), get_bezier_junction_tangent_vectors(M, B))
        ]],
        colors=Dict(
            :curves => [curveColor], :points => [dColor, bColor], :tvectors => [ξColor]
        ),
        cameraPosition=cameraPosition,
        arrowHeadSize=10.0,
        lineWidths=[1.5, 1.5],
        dotSize=4.0,
    )
    render_asymptote(joinpath(results_folder, experiment_name * "-orig.asy"); render=4)
end
pB = get_bezier_points(M, B, :differentiable)
N = PowerManifold(M, NestedPowerRepresentation(), length(pB))
F(pB) = cost_acceleration_bezier(M, pB, get_bezier_degrees(M, B), curve_samples)
∇F(pB) = ∇acceleration_bezier(M, pB, get_bezier_degrees(M, B), curve_samples)
x0 = pB
pB_opt = gradient_descent(
    N,
    F,
    ∇F,
    x0;
    stepsize=ArmijoLinesearch(1.0, ExponentialRetraction(), 0.5, 0.0001), # use Armijo lineSearch
    stopping_criterion=StopWhenAny(
        StopWhenChangeLess(10.0^(-7)),
        StopWhenGradientNormLess(7 * 10.0^-5),
        StopAfterIteration(300),
    ),
    debug=[
        :Stop,
        :Iteration,
        " | ",
        :Cost,
        " | ",
        DebugGradientNorm(),
        " | ",
        DebugStepsize(),
        " | ",
        :Change,
        "\n",
    ],
)
B_opt = get_bezier_segments(M, pB_opt, get_bezier_degrees(M, B), :differentiable)
if asy_export
    asymptote_export_S2_signals(
        joinpath(results_folder, experiment_name * "-result.asy");
        curves=[de_casteljau(M, B_opt, curve_samples_plot), cP],
        points=[get_bezier_junctions(M, B_opt), get_bezier_inner_points(M, B_opt)],
        tVectors=[[
            Tuple(a)
            for
            a in zip(
                get_bezier_junctions(M, B_opt, true),
                get_bezier_junction_tangent_vectors(M, B_opt),
            )
        ]],
        colors=Dict(
            :curves => [curveColor, pColor],
            :points => [dColor, bColor],
            :tvectors => [ξColor],
        ),
        cameraPosition=cameraPosition,
        arrowHeadSize=10.0,
        lineWidths=[1.5, 0.75, 1.5],
        dotSize=4.0,
    )
    render_asymptote(joinpath(results_folder, experiment_name * "-result.asy"); render=4)
end
