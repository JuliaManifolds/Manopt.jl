
#
# Settings
#
experimentFolder = "src/examples/Total_Variation/S1_TV/"
experimentName = "S1_TV12"
pngExport = true

n = 500
σ = 0.2
α = 0.5
β = 1.

using Manopt, Plots

dataColor = RGBA{Float64}(colorant"#BBBBBB")
s2dColor = RGBA{Float64}(colorant"#EE7733") # data Color: Tol Vibrant Orange
s1Color = RGBA{Float64}(colorant"#0077BB") # control point data color: Tol Virbant Blue
nColor = RGBA{Float64}(colorant"#33BBEE") # tangent vector: Tol Vibrant Teal

M = Circle()
N = Power(M,(n,))

f = artificialS1Signal(n)
xCompare = PowPoint(f)
fn = addNoise.(Ref(M),f,:Gaussian,σ)
data = PowPoint(fn)
t = range(0.,1.,length=n)


scene = scatter(t,getValue.(f),
    markersize=2, markercolor = dataColor, markerstrokecolor=dataColor,
    lab="original")
scatter!(scene,t,getValue.(fn),
    markersize=2, markercolor = nColor, markerstrokecolor=nColor,
    lab="noisy")
yticks!([-π,-π/2,0,π/2,π], ["-\\pi", "- \\pi/2", "0", "\\pi/2", "\\pi"])
png(scene,"$(experimentFolder)$(experimentName)-original.png")

F = x -> costL2TVplusTV2(N,data,α,β,x)
proxes = [ (λ,x) -> proxDistance(N,λ,data,x),
    (λ,x) -> proxTV(N,α*λ,x),
    (λ,x) -> proxTV2(N,β*λ,x) ]

fR, r = cyclicProximalPoint(N,F,proxes, data;
    λ = i -> π/i,
    debug = Dict(:Stop => DebugStoppingCriterion(),
                 :Step => DebugEvery(DebugGroup([
                    DebugIteration(), DebugDivider(),
                    DebugProximalParameter(), DebugDivider(),
                    DebugCost(), DebugDivider(),DebugChange(data),
                    DebugDivider("\n"),
                  ]),1000),
                 :Init => DebugDivider("Starting the solver\n")
            ),
    record = RecordGroup([RecordIteration(), RecordCost(), RecordChange(), RecordIterate(data)])
  )

scene = scatter(t,getValue.(f),
    markersize=2, markercolor = dataColor, markerstrokecolor=dataColor,
    lab="original")
scatter!(scene,t,getValue.(getValue(fR)),
    markersize=2, markercolor = nColor, markerstrokecolor=nColor,
    lab="reconstruction")
yticks!([-π,-π/2,0,π/2,π], ["-\\pi", "- \\pi/2", "0", "\\pi/2", "\\pi"])
png(scene,"$(experimentFolder)$(experimentName)-result.png")

print("MSE (input):  ",1/n*distance(N,xCompare,data)^2,"\n")
print("MSE (result): ",1/n*distance(N,xCompare,fR)^2,"\n")