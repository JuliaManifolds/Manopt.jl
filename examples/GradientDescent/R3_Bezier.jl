#
# Minimize the acceleration of a composite Bézier curve on $ℝ^3$ with approximation
#
# This example appeared in Sec. 5.2, second example, of
#
# > Bergmann, R. and Gousenbourger, P.-Y.: _A variational model for data fitting on manifolds
# > by minimizing the acceleration of a Bézier curve_.
# > Frontiers in Applied Mathematics and Statistics, 2018.
# > doi: [10.3389/fams.2018.00059](https://dx.doi.org/10.3389/fams.2018.00059),
# > arXiv: [1807.10090](https://arxiv.org/abs/1807.10090)
#
using Manopt, Manifolds, Colors, ColorSchemes, Makie

experiment_name = "Bezier_R3_Approximation"
results_folder = joinpath(@__DIR__, "Minimize_Acceleration_Bezier")
!isdir(results_folder) && mkdir(results_folder)
λ = 50.0
curve_samples = [range(0, 3; length=1601)...] # sample curve for the gradient
curve_samples_plot = [range(0, 3; length=1601)...] # sample curve for asy exports

sColor = RGBA{Float64}(colorant"#BBBBBB")
dColor = RGBA{Float64}(colorant"#EE7733") # data Color: Tol Vibrant Orange
pColor = RGBA{Float64}(colorant"#0077BB") # control point data color: Tol Virbant Blue
ξColor = RGBA{Float64}(colorant"#33BBEE") # tangent vector: Tol Vibrant blue
bColor = RGBA{Float64}(colorant"#009988") # inner control points: Tol Vibrant teal
#
# Data
#
M = Euclidean(3)
p0 = [0.0, 0.0, 1.0]
p1 = [0.0, -1.0, 0.0]
p2 = [-1.0, 0.0, 0.0]
p3 = 1 / sqrt(82) * [0.0, -1.0, -9.0]
t0p = π / (8 * sqrt(2)) * [1.0, -1.0, 0.0];
t1p = -π / (4 * sqrt(2)) * [1.0, 0.0, 1.0];
t2p = π / (4 * sqrt(2)) * [0.0, 1.0, -1.0];
t3m = π / 8 * [-1.0, 0.0, 0.0]

B = [
    BezierSegment([p0, exp(M, p0, t0p), exp(M, p1, -t1p), p1]),
    BezierSegment([p1, exp(M, p1, t1p), exp(M, p2, -t2p), p2]),
    BezierSegment([p2, exp(M, p2, t2p), exp(M, p3, t3m), p3]),
]

cP = de_casteljau(M, B, curve_samples_plot)
cPmat = hcat([[b...] for b in cP]...)
dataP = get_bezier_junctions(M, B)
pB = get_bezier_points(M, B, :differentiable)
N = PowerManifold(M, NestedPowerRepresentation(), length(pB))
function F(pB)
    return cost_L2_acceleration_bezier(
        M, pB, get_bezier_degrees(M, B), curve_samples, λ, dataP
    )
end
function gradF(pB)
    return grad_L2_acceleration_bezier(
        M, pB, get_bezier_degrees(M, B), curve_samples, λ, dataP
    )
end
x0 = pB
pB_opt = gradient_descent(
    N,
    F,
    gradF,
    x0;
    stepsize=ArmijoLinesearch(1.0, ExponentialRetraction(), 0.5, 0.0001), # use Armijo lineSearch
    stopping_criterion=StopWhenAny(
        StopWhenChangeLess(10.0^(-16)),
        StopWhenGradientNormLess(10.0^-9),
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
res_cp = get_bezier_junctions(M, B_opt)
res_curve = de_casteljau(M, B_opt, curve_samples_plot)
resPmat = hcat([[b...] for b in res_curve]...)

scene = lines(cPmat[1, :], cPmat[2, :], cPmat[3, :])
scatter!(
    scene,
    [p0[1], p1[1], p2[1], p3[1]],
    [p0[2], p1[2], p2[2], p3[2]],
    [p0[3], p1[3], p2[3], p3[3]];
    color=pColor,
)
lines!(scene, resPmat[1, :], resPmat[2, :], resPmat[3, :]; color=ξColor, linewidth=1.5)
scatter!(
    scene,
    [res_cp[1][1], res_cp[2][1], res_cp[3][1], res_cp[4][1]],
    [res_cp[1][2], res_cp[2][2], res_cp[3][2], res_cp[4][2]],
    [res_cp[1][3], res_cp[2][3], res_cp[3][3], res_cp[4][3]];
    color=dColor,
)
Makie.save(joinpath(results_folder, experiment_name * "-result.png"), scene)
