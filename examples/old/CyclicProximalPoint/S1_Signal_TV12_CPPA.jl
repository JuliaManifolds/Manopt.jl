#
#  Denoising a cyclic (S1-valued) Signal with first and second order TV.
#  using the Cyclic Proximal Point Algorithm
#
# This example was in a similar form part of the paper
#
# > Bergmann, R., Laus, F., Steidl, G., Weinmann, A.:
# > _Second Order Differences of Cyclic Data and Applications in Variational Denoising_
# > SIAM J. Imaging Sci., 7(4), 2916–2953, 2014.
# > doi: [10.1137/140969993](https://dx.doi.org/10.1137/140969993)
# > arXiv: [1405.5349](https://arxiv.org/abs/1405.5349)
#
using Manopt, Manifolds, Plots
#
# Settings
results_folder = joinpath(@__DIR__, "Signal_TV")
experiment_name = "S1_Signal_TV12_CPPA"
export_orig = true
export_result = true
!isdir(results_folder) && mkdir(results_folder)
#
# Parameters
n = 500
σ = 0.2
α = 0.5
β = 1.0
#
# Coloes
data_color = RGBA{Float64}(colorant"#BBBBBB")
result_color = RGBA{Float64}(colorant"#EE7733") # data Color: Tol Vibrant Orange
orig_color = RGBA{Float64}(colorant"#33BBEE") # tangent vector: Tol Vibrant Teal
#
# Manifolds and Data
M = Circle()
N = PowerManifold(M, n)
f = artificial_S1_signal(n)
xCompare = f
fn = exp.(Ref(M), f, random_tangent.(Ref(M), f, Val(:Gaussian), σ))
data = fn
t = range(0.0, 1.0; length=n)

if export_orig
    scene = scatter(
        t,
        f;
        markersize=2,
        markercolor=data_color,
        markerstrokecolor=data_color,
        lab="original",
    )
    scatter!(
        scene,
        t,
        fn;
        markersize=2,
        markercolor=orig_color,
        markerstrokecolor=orig_color,
        lab="noisy",
    )
    yticks!(
        [-π, -π / 2, 0, π / 2, π],
        [raw"$-\pi$", raw"$-\frac{\pi}{2}$", raw"$0$", raw"$\frac{\pi}{2}$", raw"$\pi$"],
    )
    png(scene, joinpath(results_folder, experiment_name * "-original.png"))
end
#
# Setup and Optimize
F = (N, x) -> costL2TVTV2(N, data, α, β, x)
proxes = (
    (N, λ, x) -> prox_distance(N, λ, data, x, 2),
    (N, λ, x) -> prox_TV(N, α * λ, x),
    (N, λ, x) -> prox_TV2(N, β * λ, x),
)

o = cyclic_proximal_point(
    N,
    F,
    proxes,
    data;
    λ=i -> π / (2 * i),
    debug=[
        :Iteration,
        " | ",
        DebugProximalParameter(),
        " | ",
        :Cost,
        " | ",
        :Change,
        "\n",
        1000,
        :Stop,
    ],
    record=[:Iteration, :Cost, :Change, :Iterate],
    return_options=true,
)
fR = get_solver_result(o)
r = get_record(o)
#
# Result
if export_result
    scene = scatter(
        t,
        f;
        markersize=2,
        markercolor=data_color,
        markerstrokecolor=data_color,
        lab="original",
    )
    scatter!(
        scene,
        t,
        fR;
        markersize=2,
        markercolor=result_color,
        markerstrokecolor=result_color,
        lab="reconstruction",
    )
    yticks!(
        [-π, -π / 2, 0, π / 2, π],
        [raw"$-\pi$", raw"$-\frac{\pi}{2}$", raw"$0$", raw"$\frac{\pi}{2}$", raw"$\pi$"],
    )
    png(scene, joinpath(results_folder, experiment_name * "-result.png"))
end

print("MSE (input):  ", 1 / n * distance(N, xCompare, data)^2, "\n")
print("MSE (result): ", 1 / n * distance(N, xCompare, fR)^2, "\n")
