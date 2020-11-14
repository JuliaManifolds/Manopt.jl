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
results_folder = "examples/Total_Variation/SPD_TV/"
comparison_data = "ImageCPPA-CostValue.jld2"
experimant_name = "ImageCP"
if !isdir(results_folder)
    mkdir(results_folder)
end
#
# Manifold & Data
f = artificial_SPD_image2(32)
if export_original
    asymptote_export_SPD(
        results_folder * experimant_name * "orig.asy"; data=f, scaleAxes=(7.5, 7.5, 7.5)
    )
    render_asymptote(results_folder * experimentName * "-orig.asy"; render=asy_render_detail)
end
#
# Setup & Optimize
try
    cost_threshold = load(results_folder * comparison_data)["compareCostFunctionValue"]
    global sC = StopWhenAny(
        StopAfterIteration(400), StopWhenChangeLess(10^-5), StopWhenCostLess(cost_threshold)
    )
    @info "Comparison to CPPA (`SPDImage_CPPA.jl`) and its cost of $cost_threshold."
catch y
    if isa(y, SystemError)
        @info "Comparison to CPPA only possible after runninng `SPDImage_CPPA.jl` its cost was stored."
    end
end

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
include("CP_TVModelFunctions.jl")

m = fill(Matrix{Float64}(I,3,3),size(f))
n = Λ(m)
x0 = f
ξ0 = ProductRepr(zero_tangent_vector(M2,m2(m)), zero_tangent_vector(M2,m2(m)))

#
# Stoping Criterion: Beat CPPA
#
sC = StopAfterIteration(400)
try
  costFunctionThreshold = load(results_folder*comparison_data)["compareCostFunctionValue"]
  global sC = stopWhenCostLess(costFunctionThreshold)
  @info "Comparison to CPPA (`SPDImage_CPPA.jl`) and its cost of $costFunctionThreshold."
catch y
  if isa(y, SystemError)
    @info "Comparison to CPPA only possible after runninng `SPDImage_CPPA.jl` its cost was stored using 400 iterations as a stopping criterion."
  end
end
@info "with parameters σ: $σ | τ: $τ | θ: $θ | γ: $γ."
@time a = ChambollePock(M, N, cost, x0, ξ0, m, n, proxFidelity, proxPriorDual, DΛ, AdjDΛ;
  primal_stepsize = σ, dual_stepsize = τ, relaxation = θ, acceleration = γ,
  relax = :dual,
  debug = use_debug ? [:Iteration," | ", :Cost, "\n",10,:Stop] : missing,
  record = export_table ? [:Iteration, :Cost ] : missing,
  stopping_criterion = sC,
  type = :linearized
)
if export_table
  y = a[1]
  r = a[2]
else
  y=a
end

numIter = length(r)
if export_result
    asymptote_export_SPD(
        results_folder *
        experimant_name *
        "img-result-$(numIter)-α$(replace(string(α), "." => "-")).asy";
        data=y,
        render=4,
        scaleAxes=(7.5, 7.5, 7.5),
    )
    render_asymptote(
        results_folder *
        experimentName *
        "img-result-$(numIter)-α$(replace(string(α), "." => "-")).asy";
        render=asy_render_detail,
    )
end
if export_table
    A = cat([ri[1] for ri in r], [ri[2] for ri in r]; dims=2)
    CSV.write(results_folder * experimant_name * "-Cost.csv", DataFrame(A); writeheader=false)
end
