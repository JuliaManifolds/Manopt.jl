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
export_original = false
export_result = true
export_table = true
use_debug = true
asy_render_detail = 2
results_folder = joinpath(@__DIR__,"Image_TV")
comparison_data = joinpath(@__DIR__,"data","ImageCPPA-CostValue.jld2")
experiment_name = "SPD_Image_CP"
!isdir(results_folder) && mkdir(results_folder)
#
# Manifold & Data
f = artificial_SPD_image2(32)
if export_original
    fn = joinpath(results_folder, experiment_name * "orig.asy")
    asymptote_export_SPD(fn; data=f, scaleAxes=(7.5, 7.5, 7.5) )
    render_asymptote(fn; render=asy_render_detail)
end
cost_threshold = load(comparison_data)["compareCostFunctionValue"]
sC = StopWhenAny(
    StopAfterIteration(400), StopWhenCostLess(cost_threshold)
)
@info "Comparison to CPPA (`SPDImage_CPPA.jl`) and its cost of $cost_threshold."
#
# ## Parameters
#
L = sqrt(8)
α = 6.
σ = 0.40
τ = 0.40
θ = 1.
γ = 0.2
pixelM = SymmetricPositiveDefinite(3)
#
# load TV model
#
include("Image_TV_commons.jl")

m = fill(Matrix{Float64}(I,3,3),size(f))
n = Λ(m)
x0 = f
ξ0 = ProductRepr(zero_tangent_vector(M2,m2(m)), zero_tangent_vector(M2,m2(m)))

@info "with parameters σ: $σ | τ: $τ | θ: $θ | γ: $γ."
@time o = ChambollePock(M, N, cost, x0, ξ0, m, n, proxFidelity, proxPriorDual, DΛ, AdjDΛ;
    primal_stepsize = σ, dual_stepsize = τ, relaxation = θ, acceleration = γ,
    relax = :dual,
    debug = use_debug ? [:Iteration," | ", :Cost, "\n",10,:Stop] : missing,
    record = export_table ? [:Iteration, :Cost ] : missing,
    stopping_criterion = sC,
    variant = :linearized,
    return_options = true
)
y = get_solver_result(o)
export_table && (r = get_record(o))

numIter = length(r)
if export_result
    fn = joinpath(
        results_folder,
        experiment_name * "img-result-$(numIter)-α$(replace(string(α), "." => "-")).asy"
    )
    asymptote_export_SPD(fn; data=y, scaleAxes=(7.5, 7.5, 7.5) )
    render_asymptote(fn; render=asy_render_detail)
end
if export_table
    A = cat([ri[1] for ri in r], [ri[2] for ri in r]; dims=2)
    CSV.write(
        joinpath(results_folder, experiment_name * "-Cost.csv"),
        DataFrame(A);
        writeheader=false
    )
end