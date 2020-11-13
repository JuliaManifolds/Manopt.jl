#
# Denoise an SPD Example with Douglas Rachford
#
# L2-TV functional with anisotropic TV
#
#
#
using Manopt, Manifolds
using Images, CSV, DataFrames, LinearAlgebra, JLD2, Dates
#
# Settings
ExportOrig = false
ExportResult = true
ExportTable = true
asy_render_detail = 2
resultsFolder = "examples/Total_Variation/SPD_TV/"
comparisonData = "ImageCPPA-CostValue.jld2"
experimantName = "ImageCP"
if !isdir(resultsFolder)
    mkdir(resultsFolder)
end
#
# Manifold & Data
f = artificial_SPD_image2(32)
if ExportOrig
    asymptote_export_SPD(
        resultsFolder * experimantName * "orig.asy"; data=f, scaleAxes=(7.5, 7.5, 7.5)
    )
    render_asymptote(resultsFolder * experimentName * "-orig.asy"; render=asy_render_detail)
end
#
# Setup & Optimize
try
    cost_threshold = load(resultsFolder * comparisonData)["compareCostFunctionValue"]
    global sC = StopWhenAny(
        StopAfterIteration(400), StopWhenChangeLess(10^-5), StopWhenCostLess(cost_threshold)
    )
    @info "Comparison to CPPA (`SPDImage_CPPA.jl`) and its cost of $cost_threshold."
catch y
    if isa(y, SystemError)
        @info "Comparison to CPPA only possible after runninng `SPDImage_CPPA.jl` its cost was stored."
    end
end

#
# ## Parameters
#
L = sqrt(8)
α = 6.
σ = 0.40
τ = 0.40
θ = 1.
γ = 0.2
pixelM = SymmetricPositiveDefinite(3)
#
# load TV model
#
include("CP_TVModelFunctions.jl")

m = fill(Matrix(I,3,3),size(f))
n = Λ(m)
x0 = f
ξ0 = TBTVector(zeroTVector(M2,m2(m)), zeroTVector(M2,m2(m)))

#
# Stoping Criterion: Beat CPPA
#
sC = stopAfterIteration(400)
try
  costFunctionThreshold = load(resultsFolder*comparisonData)["compareCostFunctionValue"]
  global sC = stopWhenCostLess(costFunctionThreshold)
  @info "Comparison to CPPA (`SPDImage_CPPA.jl`) and its cost of $costFunctionThreshold."
catch y
  if isa(y, SystemError)
    @info "Comparison to CPPA only possible after runninng `SPDImage_CPPA.jl` its cost was stored using 400 iterations as a stopping criterion."
  end
end
@info "with parameters σ: $σ | τ: $τ | θ: $θ | γ: $γ."
@time a = linearizedChambollePock(M, N, cost,
  x0, ξ0, m, n,
  DΛ,AdjDΛ, proxFidelity, proxPriorDual;
  primalStepSize = σ, dualStepSize = τ, relaxation = θ, acceleration = γ,
  relaxType = relaxDual,
  debug = useDebug ? [:Iteration," | ", :Cost, "\n",10,:Stop] : missing,
  record = exportTable ? [:Iteration, :Cost ] : missing,
  stoppingCriterion = sC
)
if exportTable
  y = a[1]
  r = a[2]
else
  y=a
end

numIter = length(r)
if ExportResult
    asymptote_export_SPD(
        resultsFolder *
        experimantName *
        "img-result-$(numIter)-α$(replace(string(α), "." => "-")).asy";
        data=y,
        render=4,
        scaleAxes=(7.5, 7.5, 7.5),
    )
    render_asymptote(
        resultsFolder *
        experimentName *
        "img-result-$(numIter)-α$(replace(string(α), "." => "-")).asy";
        render=asy_render_detail,
    )
end
if ExportTable
    A = cat([ri[1] for ri in r], [ri[2] for ri in r]; dims=2)
    CSV.write(resultsFolder * experimantName * "-Cost.csv", DataFrame(A); writeheader=false)
end
