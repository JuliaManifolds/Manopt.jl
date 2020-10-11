using Manopt, Manifolds, Plots
#
# Settings
resultsFolder = "examples/Total_Variation/S1_TV/"
experimentName = "S1_TV12"
plotAndExportResult = true

n = 500
σ = 0.2
α = 0.5
β = 1.0

dataColor = RGBA{Float64}(colorant"#BBBBBB")
s2dColor = RGBA{Float64}(colorant"#EE7733") # data Color: Tol Vibrant Orange
s1Color = RGBA{Float64}(colorant"#0077BB") # control point data color: Tol Virbant Blue
nColor = RGBA{Float64}(colorant"#33BBEE") # tangent vector: Tol Vibrant Teal

if !isdir(resultsFolder)
    mkdir(resultsFolder)
end
#
# Manifolds and Data
M = Circle()
N = PowerManifold(M, n)
f = artificial_S1_signal(n)
xCompare = f
fn = exp.(Ref(M), f, random_tangent.(Ref(M), f, Val(:Gaussian), σ))
data = fn
t = range(0.0, 1.0; length=n)

if plotAndExportResult
    scene = scatter(
        t,
        f;
        markersize=2,
        markercolor=dataColor,
        markerstrokecolor=dataColor,
        lab="original",
    )
    scatter!(
        scene,
        t,
        fn;
        markersize=2,
        markercolor=nColor,
        markerstrokecolor=nColor,
        lab="noisy",
    )
    yticks!(
        [-π, -π / 2, 0, π / 2, π],
        [raw"$-\pi$", raw"$-\frac{\pi}{2}$", raw"$0$", raw"$\frac{\pi}{2}$", raw"$\pi$"],
    )
    png(scene, "$(resultsFolder)$(experimentName)-original.png")
end
#
# Setup and Optimize
F = x -> costL2TVTV2(N, data, α, β, x)
proxes = (
    (λ, x) -> prox_distance(N, λ, data, x, 2),
    (λ, x) -> prox_TV(N, α * λ, x),
    (λ, x) -> prox_TV2(N, β * λ, x),
)

o = cyclic_proximal_point(
    N,
    F,
    proxes,
    data;
    λ=i -> π / (2 * i),
    debug=Dict(
        :Stop => DebugStoppingCriterion(),
        :Step => DebugEvery(
            DebugGroup([
                DebugIteration(),
                DebugDivider(),
                DebugProximalParameter(),
                DebugDivider(),
                DebugCost(),
                DebugDivider(),
                DebugChange(),
                DebugDivider("\n"),
            ]),
            1000,
        ),
        :Start => DebugDivider("Starting the solver\n"),
    ),
    record=[:Iteration, :Cost, :Change, :Iterate],
    return_options=true,
)
fR = get_solver_result(o)
r = get_record(o)
#
# Result
if plotAndExportResult
    scene = scatter(
        t,
        f;
        markersize=2,
        markercolor=dataColor,
        markerstrokecolor=dataColor,
        lab="original",
    )
    scatter!(
        scene,
        t,
        fR;
        markersize=2,
        markercolor=nColor,
        markerstrokecolor=nColor,
        lab="reconstruction",
    )
    yticks!(
        [-π, -π / 2, 0, π / 2, π],
        [raw"$-\pi$", raw"$-\frac{\pi}{2}$", raw"$0$", raw"$\frac{\pi}{2}$", raw"$\pi$"],
    )
    png(scene, "$(resultsFolder)$(experimentName)-result.png")
end

print("MSE (input):  ", 1 / n * distance(N, xCompare, data)^2, "\n")
print("MSE (result): ", 1 / n * distance(N, xCompare, fR)^2, "\n")
