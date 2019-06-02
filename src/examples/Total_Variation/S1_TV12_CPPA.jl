using Manopt, Plots
#
# Settings
resultsFolder = "src/examples/Total_Variation/S1_TV/"
experimentName = "S1_TV12"
plotAndExportResult = true

n = 500
σ = 0.2
α = 0.5
β = 1.

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
N = Power(M,(n,))
f = artificialS1Signal(n)
xCompare = PowPoint(f)
fn = addNoise.(Ref(M),f,:Gaussian,σ)
data = PowPoint(fn)
t = range(0.,1.,length=n)

if plotAndExportResult
    scene = scatter(t,getValue.(f),
        markersize=2, markercolor = dataColor, markerstrokecolor=dataColor,
        lab="original")
    scatter!(scene,t,getValue.(fn),
        markersize=2, markercolor = nColor, markerstrokecolor=nColor,
        lab="noisy")
    yticks!([-π,-π/2,0,π/2,π], ["-\\pi", "- \\pi/2", "0", "\\pi/2", "\\pi"])
    png(scene,"$(resultsFolder)$(experimentName)-original.png")
end
#
# Setup and Optimize
F = x -> costL2TVTV2(N,data,α,β,x)
proxes = [ (λ,x) -> proxDistance(N,λ,data,x),
    (λ,x) -> proxTV(N,α*λ,x),
    (λ,x) -> proxTV2(N,β*λ,x) ]

fR, r = cyclicProximalPoint(N,F,proxes, data;
    λ = i -> π/i,
    debug = Dict(:Stop => DebugStoppingCriterion(),
                 :Step => DebugEvery(DebugGroup([
                    DebugIteration(), DebugDivider(),
                    DebugProximalParameter(), DebugDivider(),
                    DebugCost(), DebugDivider(),DebugChange(),
                    DebugDivider("\n"),
                  ]),1000),
                 :Start => DebugDivider("Starting the solver\n")
            ),
    record = [:Iteration, :Cost, :Change, :Iterate]
)
#
# Result
if plotAndExportResult
    scene = scatter(t,getValue.(f),
        markersize=2, markercolor = dataColor, markerstrokecolor=dataColor,
        lab="original")
    scatter!(scene,t,getValue.(getValue(fR)),
        markersize=2, markercolor = nColor, markerstrokecolor=nColor,
        lab="reconstruction")
    yticks!([-π,-π/2,0,π/2,π], ["-\\pi", "- \\pi/2", "0", "\\pi/2", "\\pi"])
    png(scene,"$(resultsFolder)$(experimentName)-result.png")
end

print("MSE (input):  ",1/n*distance(N,xCompare,data)^2,"\n")
print("MSE (result): ",1/n*distance(N,xCompare,fR)^2,"\n")