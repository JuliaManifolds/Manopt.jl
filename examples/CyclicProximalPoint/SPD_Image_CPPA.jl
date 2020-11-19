#
# Denoise an SPD Example with Cyclic Proximal Point to minimize the
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
using Images, CSV, DataFrames, LinearAlgebra, JLD2
#
# Settings
experiment_name = "SPD_Image_CPPA"
export_orig = true # export and render input image
export_result = true # export resulting image
export_table = true # export a table of data recorded through the iterations
export_function_value = true # export the final cost function value for ChambollePock
asy_render_detail = 4 # resolution in asymptote, a high resolution is e.g. 4
results_folder = joinpath(@__DIR__, "Image_TV")
!isdir(results_folder) && mkdir(results_folder)

#
# Algorithm Parameters
α = 6.0
maxIterations = 4000

#
# Manifold and Data
f = artificial_SPD_image2(32)
pixelM = SymmetricPositiveDefinite(3)
if export_orig
    orig_filename = joinpath(results_folder, experiment_name * "-orig.asy")
    asymptote_export_SPD(orig_filename; data=f, scaleAxes=(7.5, 7.5, 7.5))
    render_asymptote(orig_filename; render=asy_render_detail)
end

#
# Build Problem for L2-TV & proximal maps
M = PowerManifold(pixelM, NestedPowerRepresentation(), size(f)...)
d = length(size(f))
rep(d) = (d > 1) ? [ones(Int, d)..., d] : d
fidelity(x) = 1 / 2 * distance(M, x, f)^2
Λ(x) = forward_logs(M, x) # on T_xN
prior(x) = norm(norm.(Ref(pixelM), repeat(x, rep(d)...), Λ(x)), 1)
cost(x) = (1 / α) * fidelity(x) + prior(x)
proxes = ((λ, x) -> prox_distance(M, λ, f, x, 2), (λ, x) -> prox_TV(M, α * λ, x, 1))

x0 = f
@time o = cyclic_proximal_point(
    M,
    cost,
    proxes,
    x0;
    λ=i -> 4 / i,
    stopping_criterion=StopAfterIteration(maxIterations),
    debug=[
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
    record=[:Iteration, :Iterate, :Cost],
    return_options=true,
)
y = get_solver_result(o)
yRec = get_record(o)
#
# Results
if export_result
    result_filename = joinpath(results_folder, experiment_name * "-result.asy")
    asymptote_export_SPD(result_filename; data=y, scaleAxes=(7.5, 7.5, 7.5))
    render_asymptote(result_filename; render=asy_render_detail)
end
if export_table
    A = cat([y[1] for y in yRec], [y[3] for y in yRec]; dims=2)
    table_filename = joinpath(results_folder, experiment_name * "-recorded-cost.csv")
    CSV.write(table_filename, DataFrame(A); writeheader=false)
end
if export_function_value
    fctval_filename = joinpath(results_folder, experiment_name * "-cost.jld2")
    values = Dict("cost_function_value" => last(yRec)[3], "iterations" => length(yRec) - 1)
    save(fctval_filename, values)
end
