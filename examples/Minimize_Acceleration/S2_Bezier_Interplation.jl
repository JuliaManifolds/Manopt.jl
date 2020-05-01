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

curve_samples = [range(0,3,length=31)...] # sample curve for the gradient
curve_samples_plot = [range(0,3,length=11)...] # sample curve for asy exports

experimentFolder = "examples/Minimize_Acceleration/S2_Bezier/"
experimentName = "Bezier_Interpolation"

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

matB = hcat(B...)
N = M^size(matB)
F(B) = costAccelerationBezier(N,B, curve_samples)
∇F(B) = gradAccelerationBezier(N,B, curve_samples)
x0 = PowPoint(matB)
PowBMinIP = steepestDescent(N, F, ∇F, x0;
    stepsize = ArmijoLinesearch(0.05,exp,0.99,0.01), # use Armijo lineSearch
    stoppingCriterion = stopWhenAny(stopWhenChangeLess(10.0^(-5)),
                                    stopWhenGradientNormLess(10.0^-5),
                                    stopAfterIteration(300)
                                ),
    debug = [:Stop, :Iteration," | ",
        :Cost, " | ", DebugGradientNorm(), " | ", DebugStepsize(), " | ", :Change, "\n"]
  )
BMinIP = [ getValue(PowBMinIP)[:,i] for i=1:size(getValue(PowBMinIP))[2] ]

if asyExport
  renderAsymptote(experimentFolder*experimentName*"-result.asy", asyExportS2Signals;
    curves = [cP, Casteljau(M,BMinIP,curve_samples_plot)],
    points = [ get_bezier_junctions(M,BMinIP), get_bezier_inner_points(M,BMinIP) ],
    tVectors = [ get_bezier_(M,BMinIP,true) ],
    colors = Dict(:curves => [sColor, curveColor], :points => [dColor, bColor], :tvectors => [ξColor]),
    cameraPosition = cameraPosition,
    lineWidth = 1., dotSize = 3.
  )
end