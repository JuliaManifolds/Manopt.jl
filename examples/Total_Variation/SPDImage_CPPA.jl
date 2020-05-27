#
# Denoise an SPD Example with Cyclic Proximal Point applied to the
#
# L2-TV functional with anisotropic TV
#
# where the example is the same data as for the corresponding CP algorithm
#
using Manopt, Manifolds
using Images, CSV, DataFrames, LinearAlgebra, JLD2
#
# Settings
ExportResult = true
ExportOrig = true
ExportTable = true
asy_render_detail = 2
#
# Manifold and Data
f = artificial_SPD_image2(32)
pixelM = SymmetricPositiveDefinite(3)
resultsFolder = "examples/Total_Variation/SPD_TV/"
experimentName = "ImageCPPA"
if !isdir(resultsFolder)
    mkdir(resultsFolder)
end
if ExportOrig
    asymptote_export_SPD(
        resultsFolder * experimentName * "-orig.asy";
        data = f,
        scaleAxes = (7.5, 7.5, 7.5),
    )
    render_asymptote(
        resultsFolder * experimentName * "-orig.asy",
        render = asy_render_detail,
    )
end
#
# Parameters
α = 6.0
maxIterations = 4000
#
# Build Problem for L2-TV
M = PowerManifold(pixelM, NestedPowerRepresentation(), size(f)...)
d = length(size(f))
rep(d) = (d > 1) ? [ones(Int, d)..., d] : d
fidelity(x) = 1 / 2 * distance(M, x, f)^2
Λ(x) = forward_logs(M, x) # on T_xN
prior(x) = norm(norm.(Ref(pixelM), repeat(x, rep(d)...), Λ(x)), 1)
#
# Setup and Optimize
cost(x) = fidelity(x) + α * prior(x)
proxes = [(λ, x) -> prox_distance(M, λ, f, x, 2), (λ, x) -> prox_TV(M, α * λ, x, 1)]
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
    stopping_criterion = StopAfterIteration(maxIterations),
    return_options = true,
)
y = get_solver_result(o)
yRec = get_record(o)
#
# Results
if ExportResult
    asymptote_export_SPD(
        resultsFolder *
        experimentName *
        "-result-$(maxIterations)-α$(replace(string(α), "." => "-")).asy";
        data = y,
        scaleAxes = (7.5, 7.5, 7.5),
    )
    render_asymptote(
        resultsFolder *
        experimentName *
        "-result-$(maxIterations)-α$(replace(string(α), "." => "-")).asy";
        render = asy_render_detail,
    )
end
if ExportTable
    A = cat([y[1] for y in yRec], [y[3] for y in yRec]; dims = 2)
    CSV.write(
        string(resultsFolder * experimentName * "ResultCost.csv"),
        DataFrame(A),
        writeheader = false,
    )
    save(
        resultsFolder * experimentName * "-CostValue.jld2",
        Dict("compareCostFunctionValue" => last(yRec)[3]),
    )
end
