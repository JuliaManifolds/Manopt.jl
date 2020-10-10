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
experimantName = "ImageDR"
if !isdir(resultsFolder)
    mkdir(resultsFolder)
end
#
# Manifold & Data
f = artificial_SPD_image2(32)
if ExportOrig
    asymptote_export_SPD(
        resultsFolder * experimantName * "orig.asy";
        data = f,
        scaleAxes = (7.5, 7.5, 7.5),
    )
    render_asymptote(
        resultsFolder * experimentName * "-orig.asy";
        render = asy_render_detail,
    )
end
#
# Parameters
η = 0.58
λ = 0.93
α = 6.0
#
# Build Problem for L2-TV
pixelM = SymmetricPositiveDefinite(3);
M = PowerManifold(pixelM, NestedPowerRepresentation(), size(f)...)
d = length(size(f))
rep(d) = (d > 1) ? [ones(Int, d)..., d] : d
fidelity(x) = 1 / 2 * distance(M, x, f)^2
Λ(x) = forward_logs(M, x) # on T_xN
prior(x) = norm(norm.(Ref(pixelM), repeat(x, rep(d)...), Λ(x)), 1)
#
# Setup & Optimize
print("--- Douglas–Rachford with η: $(η) and λ: $(λ) ---\n")
cost(x) = fidelity(x[1]) + α * prior(x[1])
N = PowerManifold(M, NestedPowerRepresentation(), 5)
prox1 = (η, x) -> [ prox_distance(M, η, f, x[1]), prox_parallel_TV(M, α * η, x[2:5]) ... ]
prox2 = (η, x) -> fill(mean(M, x), 5)
sC = StopAfterIteration(400)
try
    cost_threshold = load(resultsFolder * comparisonData)["compareCostFunctionValue"]
    global sC = StopWhenCostLess(cost_threshold)
    @info "Comparison to CPPA (`SPDImage_CPPA.jl`) and its cost of $cost_threshold."
catch y
    if isa(y, SystemError)
        @info "Comparison to CPPA only possible after runninng `SPDImage_CPPA.jl` its cost was stored."
    end
end
x0 = fill(f, 5)
@time o = DouglasRachford(
    N,
    cost,
    [prox1, prox2],
    x0;
    λ = i -> η,
    α = i -> λ, # map from Paper notation of BPS16 to toolbox notation
    debug = [:Iteration, " | ", :Change, " | ", :Cost, "\n", 10, :Stop],
    record = [:Iteration, :Cost],
    stopping_criterion = sC,
    return_options = true,
)
y = get_solver_result(o)
r = get_record(o)
#
# Result
numIter = length(r)
if ExportResult
    asymptote_export_SPD(
        resultsFolder *
        experimantName *
        "img-result-$(numIter)-α$(replace(string(α), "." => "-")).asy";
        data = y,
        render = 4,
        scaleAxes = (7.5, 7.5, 7.5),
    )
    render_asymptote(
        resultsFolder *
        experimentName *
        "img-result-$(numIter)-α$(replace(string(α), "." => "-")).asy";
        render = asy_render_detail,
    )
end
if ExportTable
    A = cat([ri[1] for ri in r], [ri[2] for ri in r]; dims = 2)
    CSV.write(
        resultsFolder * experimantName * "-Cost.csv",
        DataFrame(A);
        writeheader = false,
    )
end
