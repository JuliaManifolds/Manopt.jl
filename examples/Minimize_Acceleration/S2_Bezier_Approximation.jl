"""
# Minimize the acceleration of a composite Bézier curve on the Sphere S2 with interpolation

This example appeared in Sec. 5.2, second example, of
> R. Bergmann, P.-Y. Gousenbourger: _A variational model for data fitting on manifolds
> by minimizing the acceleration of a Bézier curve_.
> Frontiers in Applied Mathematics and Statistics, 2018.
> doi: [10.3389/fams.2018.00059](https://dx.doi.org/10.3389/fams.2018.00059)
"""
#
# Load Manopt and required packages
#
using Manopt, Manifolds, Colors, ColorSchemes
asyExport = true #export data and results to asyExport
λ = 10.0

curve_samples = [range(0,3,length=31)...] # sample curve for the gradient
curve_samples_plot = [range(0,3,length=11)...] # sample curve for asy exports

experimentFolder = "examples/Minimize_Acceleration/S2_Bezier/"
experimentName = "Bezier_Approximation"

cameraPosition = (-1., -0.7, 0.3)
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
cP = de_casteljau(M,B,curve_samples_plot)
dataP = get_bezier_junctions(M,B)
# export original data.
if asyExport
    asymptote_export_S2_signals(experimentFolder*experimentName*"-orig.asy";
        curves = [cP],
        points = [ get_bezier_junctions(M,B), get_bezier_inner_points(M,B) ],
        tVectors = [[Tuple(a) for a in zip(get_bezier_junctions(M,B,true), get_bezier_junction_tangent_vectors(M,B))]],
        colors = Dict(:curves => [curveColor], :points => [dColor, bColor], :tvectors => [ξColor]),
        cameraPosition = cameraPosition,
        arrowHeadSize = 10.,
        lineWidths= [1.5, 1.5], dotSize = 4.
    )
    render_asymptote(experimentFolder*experimentName*"-orig.asy"; render = 4)
end
tup2mat(B) = hcat([[b...] for b in B]...)
mat2tup(matB) = [Tuple(matB[:,i]) for i=1:size(matB,2)]
matB = tup2mat(B)
N = PowerManifold(M, NestedPowerRepresentation(), size(matB)...)
F(matB) = cost_L2_acceleration_bezier(M, mat2tup(matB), curve_samples,λ,dataP)
∇F(matB) = tup2mat(∇L2_acceleration_bezier(M, mat2tup(matB), curve_samples,λ,dataP))
x0 = matB
Bmat_opt = steepest_descent(N, F, ∇F, x0;
    stepsize = ArmijoLinesearch(0.05,ExponentialRetraction(),0.99,0.001), # use Armijo lineSearch
    stopping_criterion = StopWhenAny(StopWhenChangeLess(10.0^(-5)),
                                    StopWhenGradientNormLess(10.0^-5),
                                    StopAfterIteration(300),
                                ),
    debug = [:Stop, :Iteration," | ",
        :Cost, " | ", DebugGradientNorm(), " | ", DebugStepsize(), " | ", :Change, "\n"]
  )
B_opt = mat2tup(Bmat_opt)
res_curve = de_casteljau(M,B_opt,curve_samples_plot)
#res_curve ./= norm.(res_curve)
if asyExport
      asymptote_export_S2_signals(experimentFolder*experimentName*"-result.asy";
        curves = [res_curve],
        points = [ get_bezier_junctions(M,B_opt), get_bezier_inner_points(M,B_opt) ],
        tVectors = [
            [
                Tuple(a) for a in zip(
                    get_bezier_junctions(M,B_opt,true),
                    get_bezier_junction_tangent_vectors(M,B_opt)
                )
            ]
        ],
        colors = Dict(:curves => [curveColor], :points => [dColor, bColor], :tvectors => [ξColor]),
        cameraPosition = cameraPosition,
        arrowHeadSize = 10.,
        lineWidths= [1.5, 1.5], dotSize = 4.
    )
    render_asymptote(experimentFolder*experimentName*"-result.asy"; render = 4)
end