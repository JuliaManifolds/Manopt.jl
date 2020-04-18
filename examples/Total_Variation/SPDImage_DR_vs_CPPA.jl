#
# Denoise an SPD Example with Douglas Rachford
#
# L2-TV functional with anisotropic TV
#
#
#
using Manopt
using Images, CSV, DataFrames, LinearAlgebra, JLD2, Dates
#
# Settings
ExportOrig = false
ExportResult = true
ExportTable = true
resultsFolder = "src/examples/Total_Variation/SPD_TV/"
comparisonData = "ImageCPPA-CostValue.jld2"
experimantName = "ImageDR"
if !isdir(resultsFolder)
    mkdir(resultsFolder)
end
#
# Manifold & Data
f = artificial_SPD_image2(32)
if ExportOrig
  asymptote_export_SPD(resultsFolder*experimantName*"orig.asy"; data=f, scaleAxes=(7.5,7.5,7.5))
end
#
# Parameters
η  = 0.58
λ = 0.93
α = 6.
#
# Build Problem for L2-TV
pixelM = SymmetricPositiveDefinite(3);
M = PowerManifold(pixelM,size(f))
d = length(size(f))
rep(d) = (d>1) ? [ones(Int,d)...,d] : d
fidelity(x) = 1/2*distance(M,x,f)^2
Λ(x) = forward_logs(M,x) # on T_xN
prior(x) = norm(norm.(Ref(pixelM), repeat(x,rep(d)...), Λ(x),1)
#
# Setup & Optimize
print("--- Douglas–Rachford with η: ",η," and λ: ",λ," ---\n")
cost(x) = fidelity(x) + α*prior(x)
prox1 = (η,x) -> cat( proxDistance(M,η,f,x[1]), proxParallelTV(M,α*η,x[2:5]), dims=1)
prox2 = (η,x) -> fill(mean(M,x;stoppingCriterion=StopAfterIteration(20)),5)
sC = StopAfterIteration(400)
try
  cost_threshold = load(resultsFolder*comparisonData)["compareCostFunctionValue"]
  global sC = StopWhenCostLess(cost_threshold)
  @info "Comparison to CPPA (`SPDImage_CPPA.jl`) and its cost of $cost_threshold."
catch y
  if isa(y, SystemError)
    @info "Comparison to CPPA only possible after runninng `SPDImage_CPPA.jl` its cost was stored."
  end
end
x0 = f
@time o = DouglasRachford(M, cost, [prox1,prox2], f;
  λ = i -> η, α = i -> λ, # map from Paper notation of BPS16 to toolbox notation
  debug = [:Iteration," | ", :Change, " | ", :Cost,"\n",10,:Stop],
  record = [:Iteration, :Cost ],
  stoppingCriterion = sC,
  parallel=5,
  returnOptions = true
)
y = get_solver_result(o)
r = get_record(o)
#
# Result
numIter = length(r)
if ExportResult
  asymptote_export_SPD(resultsFolder*experimantName*"img-result-$(numIter)-α$(replace(string(α), "." => "-")).asy"; data=y, render=4, scaleAxes=(7.5,7.5,7.5) )
end
if ExportTable
  A = cat( [ri[1] for ri in r], [ri[2] for ri in r]; dims=2 )
  CSV.write(resultsFolder*experimantName*"-Cost.csv",  DataFrame(A), writeheader=false);
end