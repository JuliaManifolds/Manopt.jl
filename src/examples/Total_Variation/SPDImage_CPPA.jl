#
# Denoise an SPD Example with Cyclic Proximal Point applied to the
#
# L2-TV functional with anisotropic TV
#
# where the example is the same data as for the corresponding CP algorithm
#
using Manopt
using Images, CSV, DataFrames, LinearAlgebra, JLD2
#
# Settings
ExportResult = true
ExportOrig = true
ExportResultVideo = false
ExportTable = true
#
# Manifold and Data
f = artificialSPDImage2(32)
pixelM = SymmetricPositiveDefinite(3)
resultsFolder = "src/examples/Total_Variation/SPD_TV/"
experimentName = "ImageCPPA"
if !isdir(resultsFolder)
    mkdir(resultsFolder)
end
if ExportOrig
    renderAsymptote(resultsFolder*experimentName*"-orig.asy"; data=f, scaleAxes=(7.5,7.5,7.5))
end
#
# Parameters 
α = 6.
maxIterations = 4000
#
# Build Problem for L2-TV
M = Power(pixelM,size(f))
d = length(size(f))
rep(d) = (d>1) ? [ones(Int,d)...,d] : d
fidelity(x) = 1/2*distance(M,x,f)^2
Λ(x) = forwardLogs(M,x) # on T_xN
prior(x) = norm(norm.(Ref(pixelM),getValue(repeat(x,rep(d)...)),getValue(Λ(x))),1)
#
# Setup and Optimize
cost(x) =  fidelity(x) + α*prior(x)
proximalMaps = [(λ,x) -> proxDistance(M,λ,f,x,2), (λ,x) -> proxTV(M,α*λ,x,1)]
x0 = f
@time y, yRec = cyclicProximalPoint(M,cost,proximalMaps,x0;
debug = [:Iteration," | ", DebugProximalParameter()," | ", :Change, " | ", :Cost, "\n",100,:Stop],
record = [:Iteration, :Iterate, :Cost],
stoppingCriterion = stopAfterIteration(maxIterations)
)
#
# Results
if ExportResult
    renderAsymptote(resultsFolder*experimentName*"-result-$(maxIterations)-α$(replace(string(α), "." => "-")).asy"; data=y, render=4, scaleAxes=(7.5,7.5,7.5) )
end
if ExportTable
    A = cat( [ y[1] for y in yRec], [y[3] for y in yRec]; dims=2 )
    CSV.write(string(resultsFolder*experimentName*"ResultCost.csv"),
    DataFrame(A), writeheader=false);
    save(resultsFolder*experimentName*"-CostValue.jld2",
    Dict("compareCostFunctionValue" => last(yRec)[3])
    )
end