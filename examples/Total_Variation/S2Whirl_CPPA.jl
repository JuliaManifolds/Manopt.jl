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
resultsFolder = "src/examples/Total_Variation/S2_TV/"
experimentName = "WhirlCPPA"
if !isdir(resultsFolder)
    mkdir(resultsFolder)
end
#
# Manifold & Data
f = artificial_S2_whirl_image(64)
pixelM = Sphere(2);

if ExportOrig
    asymptote_export_S2_data(resultsFolder * experimentName * "-orig.asy"; data = f) # (5)
end
#
# Parameters
α = 1.5
maxIterations = 4000
#
# Build Problem for L2-TV
M = PowerManifold(pixelM, size(f))
d = length(size(f))
iRep = [Integer.(ones(d))..., d]
fidelity(x) = 1 / 2 * distance(M, x, f)^2
Λ(x) = forward_logs(M, x) # on T_xN
prior(x) = norm(norm.(Ref(pixelM), repeat(x, iRep...), Λ(x)), 1)
#
# Setup and Optimize
cost(x) = fidelity(x) + α * prior(x)
proxes = [(λ, x) -> proxDistance(M, λ, f, x, 2), (λ, x) -> proxTV(M, α * λ, x, 1)]
x0 = f
@time o = cyclic_proximal_point(
    M,
    cost,
    proxes,
    x0;
    debug = [
        :Iteration,
        " | ",
        DebugProximalParameter(),
        " | ",
        :Change,
        " | ",
        :Cost,
        "\n",
        100,
        :Stop,
    ],
    record = [:Iteration, :Iterate, :Cost],
    stoppingCriterion = StopAfterIteration(maxIterations),
    λ = i -> π / (2 * i),
    returnOptions = true,
)
y = get_solver_result(o)
yRec = get_record(o)
#
# Results
if ExportResult
    asymptote_export_S2_data(
        resultsFolder *
        experimentName *
        "-result-$(maxIterations)-α$(replace(string(α), "." => "-")).asy";
        data = y,
        render = 4,
    ) #(6)
end
if ExportTable
    A = cat([y[1] for y in yRec], [y[3] for y in yRec]; dims = 2)
    CSV.write(
        string(resultsFolder * experimentName * "-Result.csv"),
        DataFrame(A),
        writeheader = false,
    )
    save(
        resultsFolder * experimentName * "-CostValue.jld2",
        Dict("compareCostFunctionValue" => last(yRec)[3]),
    )
end
