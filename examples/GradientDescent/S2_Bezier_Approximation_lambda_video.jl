#
# Minimize the acceleration of a composite Bézier curve on the Sphere S2 with interpolation
#
# This example appeared in Sec. 5.2, second example, of
#
# > Bergmann, R. and Gousenbourger, P.-Y.: _A variational model for data fitting on manifolds
# > by minimizing the acceleration of a Bézier curve_.
# > Frontiers in Applied Mathematics and Statistics, 2018.
# > doi: [10.3389/fams.2018.00059](https://dx.doi.org/10.3389/fams.2018.00059),
# > arXiv: [1807.10090](https://arxiv.org/abs/1807.10090)
#

#
# Load Manopt and required packages
#
using Manopt, Manifolds, Colors, ColorSchemes
import Printf.@sprintf
import ColorSchemes.viridis

results_folder = joinpath(@__DIR__, "Minimize_Acceleration_Bezier", "video")
!isdir(results_folder) && mkdir(results_folder)
final_folder = joinpath(@__DIR__, "Minimize_Acceleration_Bezier")
!isdir(final_folder) && mkdir(final_folder)
experiment_name = "Bezier_Approximation_video"

render_detail = 2
asy_export = true
asy_export_summary = true
render_video = true
λRange = collect(range(10.0, 0.0; length=1001))[1:(end - 1)] #exclude zero.
colors = RGBA.(get.(Ref(viridis), range(0.0, 1.0; length=length(λRange))))

curve_samples = [range(0, 3; length=101)...] # sample curve for the gradient
curve_samples_plot = [range(0, 3; length=201)...] # sample curve for asy exports

camera_position = (-1.0, -0.7, 0.3)
curveColor = RGBA{Float64}(colorant"#AAAAAA")
M = Sphere(2)
B = artificial_S2_composite_bezier_curve()
degs = get_bezier_degrees(M, B)
cP = de_casteljau(M, B, curve_samples_plot)
d = get_bezier_junctions(M, B)
pB = get_bezier_points(M, B, :differentiable)
results = [similar(pB) for i in eachindex(λRange)]
resulting_curves = [similar(cP) for i in eachindex(λRange)]
N = PowerManifold(M, NestedPowerRepresentation(), length(pB))
x0 = pB
for i in eachindex(λRange)
    λ = λRange[i]
    F(M, pB) = cost_L2_acceleration_bezier(M.manifold, pB, degs, curve_samples, λ, d)
    gradF(M, pB) = grad_L2_acceleration_bezier(M.manifold, pB, degs, curve_samples, λ, d)
    results[i] = gradient_descent(
        N,
        F,
        gradF,
        x0;
        stepsize=ArmijoLinesearch(1.0, ExponentialRetraction(), 0.5, 0.001),
        stopping_criterion=StopWhenAny(
            StopWhenChangeLess(5 * 10.0^(-6)), StopAfterIteration(15000)
        ),
        debug=[
            :Stop,
            "$(λ) ",
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
            200,
        ],
    )
    B_opt = get_bezier_segments(M, results[i], degs, :differentiable)
    resulting_curves[i] = de_casteljau(M, B_opt, curve_samples_plot)
end
if asy_export
    print("Exporting...\n")
    s = joinpath(results_folder, experiment_name)
    for i in eachindex(λRange)
        print("$(λRange[i])..")
        B_opt = get_bezier_segments(M, results[i], degs, :differentiable)
        asymptote_export_S2_signals(
            s * "-$(@sprintf "%04.0f" i)-result.asy";
            curves=[resulting_curves[i], cP],
            points=[get_bezier_junctions(M, B_opt), get_bezier_inner_points(M, B_opt)],
            colors=Dict(
                :curves => [colors[i], curveColor], :points => [colors[i], colors[i]]
            ),
            camera_position=camera_position,
            line_widths=[1.0, 0.5],
            dot_size=2.0,
        )
        render_asymptote(s * "-$(@sprintf "%04.0f" i)-result.asy"; render=render_detail)
    end
end
if asy_export_summary
    s = joinpath(final_folder, experiment_name)
    asymptote_export_S2_signals(
        s * "-Summary-result.asy";
        curves=[cP, resulting_curves...],
        colors=Dict(:curves => [curveColor, colors...]),
        camera_position=camera_position,
        line_widths=[0.75, [1.5 for i in eachindex(λRange)]...],
    )
    render_asymptote(s * "-Summary-result.asy"; render=render_detail)
end
if render_video
    s = joinpath(results_folder, experiment_name)
    s2 = joinpath(final_folder, experiment_name)
    cmd = `ffmpeg -i $(s)-%04d-result.png -r 25 -framerate 60 -c:v libx264 -crf 20 -pix_fmt yuv420p -y $(s2)-movie.mp4`
    run(cmd)
end
