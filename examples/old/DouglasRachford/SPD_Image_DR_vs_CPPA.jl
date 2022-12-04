#
# Denoise an SPD Example with Douglas Rachford
#
# Denoise an SPD Example with Parallel Douglas-Rachford to minimize the
#
# L2-TV functional with anisotropic TV
#
# where the example is the same data and cost as for SPD_Image_CP_vs_CPPA
#
# This example and its data is used for comparison in Section 6.2 of
#
# > R. Bergmann, R. Herzog, M. Silva Louzeiro, D. Tenbrinck, J. Vidal Núñez:
# > _Fenchel Duality Theory and a Primal-Dual Algorithm on Riemannian Manifolds_,
# > arXiv: [1908.02022](https://arxiv.org/abs/1908.02022)
#
using Manopt, Manifolds
using Images, CSV, DataFrames, LinearAlgebra, JLD2, Dates
#
# Settings
experiment_name = "SPD_Image_DR"
export_orig = true
export_result = true
export_table = true
use_debug = true
asy_render_detail = 2
results_folder = joinpath(@__DIR__, "Image_TV")
comparison_data = joinpath(
    @__DIR__, "..", "CyclicProximalPoint", "Image_TV", "SPD_Image_CPPA-cost.jld2"
)
!isdir(results_folder) && mkdir(results_folder)
#
# Parameters
η = 0.58
λ = 0.93
α = 6.0
#
# Manifold & Data
f = artificial_SPD_image2(32)
if export_orig
    fn = joinpath(results_folder, experiment_name * "-orig.asy")
    asymptote_export_SPD(fn; data=f, scale_axes=(7.5, 7.5, 7.5))
    render_asymptote(fn; render=asy_render_detail)
end
sC = StopAfterIteration(400)
try
    cppa_data = load(comparison_data)
    cost_threshold = cppa_data["cost_function_value"]
    cppa_iter = cppa_data["iterations"]
    global sC = StopWhenAny(StopAfterIteration(400), StopWhenCostLess(cost_threshold))
    @info "Comparison to CPPA (`SPDImage_CPPA.jl`) and its cost after $(cppa_iter) iterations of $cost_threshold."
catch e
    msg = sprint(showerror, e)
    @info "Comparison to CPPA (`CyclicProximalPoint/SPDImage_CPPA.jl`) not possible, data file $(comparison_data) either missing or corrupted.\n Error: $(msg).\n\n Starting with a default stopping criterion."
end
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
cost(x) = 1 / α * fidelity(x[1]) + prior(x[1])
N = PowerManifold(M, NestedPowerRepresentation(), 5)
prox1 = (η, x) -> [prox_distance(M, η, f, x[1]), prox_parallel_TV(M, α * η, x[2:5])...]
prox2 = (η, x) -> fill(mean(M, x, GradientDescentEstimation(); stop_iter=4), 5)
x0 = fill(f, 5)
@time o = DouglasRachford(
    N,
    cost,
    [prox1, prox2],
    x0;
    λ=i -> η,
    α=i -> λ, # map from Paper notation of BPS16 to toolbox notation
    debug=use_debug ? [:Iteration, " | ", :Cost, "\n", 10, :Stop] : missing,
    record=export_table ? [:Iteration, :Cost] : missing,
    stopping_criterion=sC,
    return_state=true,
)
y = get_solver_result(o)[1]
export_table && (r = get_record(o))
#
# Result
numIter = length(r)
if export_result
    fn = joinpath(
        results_folder,
        experiment_name * "img-result-$(numIter)-α$(replace(string(α), "." => "-")).asy",
    )
    asymptote_export_SPD(fn; data=y, scale_axes=(7.5, 7.5, 7.5))
    render_asymptote(fn; render=asy_render_detail)
end
if export_table
    # scale cost back for saving such that its comparable with the other two results
    A = cat([ri[1] for ri in r], [ri[2] / α for ri in r]; dims=2)
    CSV.write(
        joinpath(results_folder, experiment_name * "-Cost.csv"),
        DataFrame(A);
        writeheader=false,
    )
end
