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
#
ExportOrig = false
ExportResult = true
ExportTable = true

f = artificialSPDImage2(;pts=32)

resultsFolder = "src/examples/Total_Variation/SPD_TV/"
comparisonData = "ImageCPPA-CostValue.jld2"
experimantName = "ImageDR"
if ExportOrig
  renderAsymptote(resultsFolder*experimantName*"orig.asy", asyExportSPDData; data=f, scaleAxes=(7.5,7.5,7.5))
end
#
# Parameters 
#

η  = 0.58
λ = 0.93
print("--- Douglas–Rachford with η: ",η," and λ: ",λ," ---\n")

α = 6.

#
# Build Problem for L2-TV
#

pixelM = SymmetricPositiveDefinite(3);
M = Power(pixelM,size(f))
d = length(size(f))
rep(d) = (d>1) ? [ones(Int,d)...,d] : d
fidelity(x) = 1/2*distance(M,x,f)^2
Λ(x) = forwardLogs(M,x) # on T_xN
function prior(x)
  return norm(norm.(Ref(pixelM),getValue(repeat(x,rep(d)...)), getValue(Λ(x)) ), 1)
end

cost(x) = fidelity(x) + α*prior(x)
  
prox1 = (η,x) -> PowPoint(cat( proxDistance(M,η,f,x[1]), proxParallelTV(M,α*η,getValue(x[2:5])), dims=1))
prox2 = (η,x) -> PowPoint(fill(mean(M,getValue(x);stoppingCriterion=stopAfterIteration(20)),5))

sC = stopAfterIteration(400)
try 
  costFunctionThreshold = load(resultsFolder*comparisonData)["compareCostFunctionValue"]
  global sC = stopWhenCostLess(costFunctionThreshold)
  @info "Comparison to CPPA (`SPDImage_CPPA.jl`) and that cost of $costFunctionThreshold."
catch y
  if isa(y, SystemError)
    @info "Comparison to CPPA only possible after runninng `SPDImage_CPPA.jl` and that cost was stored."
  end
end
x0 = f
print(sC)
@time y, r = DouglasRachford(M, cost, [prox1,prox2], f;
  λ = i -> η, α = i -> λ, # map from Paper notation of BPS16 to toolbox notation
  debug = [:Iteration," | ", :Change, " | ", :Cost,"\n",10,:Stop],
  record = [:Iteration, :Cost ],
  stoppingCriterion = sC,
  parallel=5
)
numIter = length(r)
if ExportResult
  renderAsymptote(resultsFolder*experimantName*"img-result-$(numIter)-α$(replace(string(α), "." => "-")).asy", asyExportSPDData; data=y, render=4, scaleAxes=(7.5,7.5,7.5) )
end
if ExportTable
  A = cat( [ri[1] for ri in r], [ri[2] for ri in r]; dims=2 )
  CSV.write(resultsFolder*experimantName*"-Cost.csv",  DataFrame(A), writeheader=false);
end