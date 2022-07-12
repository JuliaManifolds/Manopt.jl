#
# Minimize total variation of a signal of R3 data.
#
# This example is part of Example 6.1 in the publication
#
# > W. Diepeveen, J. Lellmann:
# > _An Inexact Semismooth Newton Method on Riemannian Manifolds with Application to Duality-Based Total Variation Denoising_,
# > SIAM Journal on Imaging Sciences, 2021.
# > doi: [10.1137/21M1398513](https://doi.org/10.1137/21M1398513)
#
using Manopt, Manifolds, LinearAlgebra

#
# Script Settings
experiment_name = "R1_Signal_TV_CP"
export_orig = true
export_primal = true
export_table = true
use_debug = true
#
# Automatic Script Settings
current_folder = @__DIR__
export_any = export_orig || export_primal || export_table
results_folder = joinpath(current_folder, "Signal_TV")
# Create folder if we have an export
(export_any && !isdir(results_folder)) && mkdir(results_folder)

#
# Example Settings
signal_section_size = 2
α = 0.5
σ = 0.5
τ = 0.5
θ = 1.0
γ = 0.0
max_iterations = 500
noise_level = 0.0
noise_type = :Gaussian

pixelM = Euclidean(1)
base = [0.0]
X = π / 4 * [0.0, 1.0, 0.0]
# Generate a signal with two sections - same signal as in S2_Signal_TV
p1 = [1.0]
p2 = [-1.0]
f = vcat(fill(p1, signal_section_size), fill(p2, signal_section_size))
#
# Compute exact minimizer
include("Signal_TV_commons.jl")
jump_height = distance(pixelM, f[signal_section_size], f[signal_section_size + 1])
δ = min(2 / (size(f, 1) * jump_height) * α, 1 / 2)
x_hat = shortest_geodesic(M, f, reverse(f), δ)
# include("Ck.jl")

# add noise
if noise_level > 0
    f = [exp(pixelM, p, random_tangent(pixelM, p, noise_type, noise_level)) for p in f]
end
# if export_orig
#     orig_file = joinpath(results_folder, experiment_name * "-original.asy")
#     asymptote_export_S2_data(orig_file; data=f)
#     render_asymptote(orig_file)
# end
#
# Initial values
m = fill(base, size(f))
n = Λ(M, m)
x0 = deepcopy(f)
ξ0 = zero_vector(M, m)
# ξ0 = ProductRepr(zero_vector(M, m), zero_vector(M, m))

storage = StoreOptionsAction((:x, :n, :ξbar))

@time o = ChambollePock(
    M,
    N,
    cost,
    x0,
    ξ0,
    m,
    n,
    prox_F,
    prox_G_dual,
    adjoint_DΛ;
    linearized_forward_operator=DΛ,
    primal_stepsize=σ,
    dual_stepsize=τ,
    relaxation=θ,
    acceleration=γ,
    relax=:dual,
    variant=:linearized,
    # debug=if use_debug
    #     [
    #         :Iteration,
    #         " ",
    #         DebugPrimalChange(),
    #         " | ",
    #         DebugCk(storage),
    #         " | ",
    #         :Cost,
    #         "\n",
    #         100,
    #         :Stop,
    #     ]
    # else
    #     missing
    # end,
    record=if export_table
        [:Iteration, RecordPrimalChange(), RecordDualChange(), :Cost]
    else
        missing
    end,
    stopping_criterion=StopAfterIteration(max_iterations),
    return_options=true,
)
y = get_solver_result(o)
# if has_record(o)
#     r = get_record(o)
#     Ck_values = [s[5] for s in r]
#     println("The Ck Estimate lies between $(minimum(Ck_values)) and $(maximum(Ck_values))")
# end
println("Distance from result to minimizer: ", distance(M, x_hat, y), "\n")
print("y = $(y)\n")
# if export_primal
#     orig_file = joinpath(results_folder, experiment_name * "-result.asy")
#     asymptote_export_S2_data(orig_file; data=y)
#     render_asymptote(orig_file)
# end
@time o_pdrssn = primal_dual_semismooth_Newton(
    M,
    N,
    cost,
    x0,
    ξ0,
    m,
    n,
    prox_F,
    Dprox_F,
    prox_G_dual,
    Dprox_G_dual,
    DΛ,
    adjoint_DΛ;
    primal_stepsize=σ,
    dual_stepsize=τ,
    record=if export_table
        [:Iteration, RecordPrimalChange(), RecordDualChange(), :Cost]
    else
        missing
    end,
    stopping_criterion=StopAfterIteration(3),
    # stopping_criterion=StopAfterIteration(max_iterations),
    return_options=true,
)

y_pdrssn = get_solver_result(o_pdrssn)
# if has_record(o)
#     r = get_record(o)
#     Ck_values = [s[5] for s in r]
#     println("The Ck Estimate lies between $(minimum(Ck_values)) and $(maximum(Ck_values))")
# end
println("Distance from result to minimizer: ", distance(M, x_hat, y_pdrssn), "\n")
print("y_pdrssn = $(y_pdrssn)\n")
# if export_primal
#     orig_file = joinpath(results_folder, experiment_name * "-result-pdrssn.asy")
#     asymptote_export_S2_data(orig_file; data=y_pdrssn)
#     render_asymptote(orig_file)
# end
