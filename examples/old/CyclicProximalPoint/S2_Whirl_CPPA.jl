#
# Denoise an S2-valued image Example with Cyclic Proximal Point applied to the
#
# L2-TV functional with anisotropic TV
#
# where the example is the same data as for the corresponding CP algorithm
#
using Manopt, Manifolds
using Images, CSV, DataFrames, LinearAlgebra, JLD2
#
# Settings
experiment_name = "S2_Whirl_CPPA"
results_folder = joinpath(@__DIR__, "S2_TV")
export_result = true
export_orig = true
export_table = true
export_function_value = true
asymptote_render_detail = 2
!isdir(results_folder) && mkdir(results_folder)
#
# Manifold & Data
f = artificial_S2_whirl_image(64)
pixelM = Sphere(2);

if export_orig
    orig_filename = joinpath(results_folder, experiment_name * "-orig.asy")
    asymptote_export_S2_data(orig_filename; data=f)
    render_asymptote(orig_filename; render=asymptote_render_detail)
end
#
# Parameters
α = 1.5
maxIterations = 4000
#
# Build Problem for L2-TV
M = PowerManifold(pixelM, NestedPowerRepresentation(), size(f)...)
d = length(size(f))
iRep = [Integer.(ones(d))..., d]
fidelity(M, x) = 1 / 2 * distance(M, x, f)^2
Λ(M, x) = forward_logs(M, x) # on T_xN
prior(M, x) = norm(norm.(Ref(pixelM), repeat(x, iRep...), Λ(M, x)), 1)
#
# Setup and Optimize
cost(M, x) = fidelity(M, x) + α * prior(M, x)
proxes = ((M, λ, x) -> prox_distance(M, λ, f, x, 2), (M, λ, x) -> prox_TV(M, α * λ, x, 1))
x0 = f
@time o = cyclic_proximal_point(
    M,
    cost,
    proxes,
    x0;
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
    stopping_criterion=StopAfterIteration(maxIterations),
    λ=i -> π / (2 * i),
    return_options=true,
)
y = get_solver_result(o)
yRec = get_record(o)
#
# Results
if export_result
    result_filename = joinpath(
        results_folder,
        experiment_name * "-result-$(maxIterations)-α$(replace(string(α), "." => "-")).asy",
    )
    asymptote_export_S2_data(result_filename; data=y)
    render_asymptote(result_filename; render=asymptote_render_detail)
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
