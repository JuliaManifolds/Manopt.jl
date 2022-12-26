#
# Denoise an S2 valued image demonstrating the effect of using different base points m
#
# L2-TV functional with anisotropic TV
#
# This example is used in Section 6.2 of
#
# > R. Bergmann, R. Herzog, M. Silva Louzeiro, D. Tenbrinck, J. Vidal Núñez:
# > _Fenchel Duality Theory and a Primal-Dual Algorithm on Riemannian Manifolds_,
# > arXiv: [1908.02022](https://arxiv.org/abs/1908.02022)
#
using CSV, DataFrames, LinearAlgebra, Manopt, Manifolds
#
# Settings
experiment_name = "S2_WhirlImage_CP"
export_orig = false
export_primal = false
export_primal_video = false
export_table = false
asy_render_size = 2
#
# Automatic Settings
export_any = export_orig || export_primal || export_primal_video || export_table
results_folder = joinpath(@__DIR__, "Image_TV")
video_folder = joinpath(results_folder, "video")
(export_any && !isdir(results_folder)) && mkdir(results_folder)
(export_primal_video && !isdir(video_folder)) && mkdir(video_folder)
#
# Experiment Parameters
σ = 0.25
τ = 0.25
γ = 0.2
θ = 1.0
α = 1.5
#
# Data
pixelM = Sphere(2)
f = artificial_S2_whirl_image(64)
#
# load TV model
#
include("Image_TV_commons.jl")

if export_orig
    orig_file = joinpath(results_folder, experiment_name * "-orig.asy")
    asymptote_export_S2_data(orig_file; data=f)
    render_asymptote(orig_file; render=asy_render_size)
end
struct DebugRenderAsy <: DebugAction
    name::String
    folder::String
    DebugRenderAsy(f, n) = new(n, f)
end
function (d::DebugRenderAsy)(::TwoManifoldProblem, ::ChambollePockState, i)
    if i >= 0
        orig_file = joinpath(d.folder, d.name * "-vid-$(lpad(i[1],7,"0")).asy")
        asymptote_export_S2_data(orig_file; data=f)
        render_asymptote(orig_file; render=asy_render_size)
    end
end
#
# Experiments (entries overwrite defaults from below)
data_mean = mean(pixelM, vec(f), GradientDescentEstimation(); stop_iter=5)
mean_image = fill(data_mean, size(f))
west_image = fill([1.0, 0.0, 0.0], size(f))
#
# Build Experiments
experiments = [
    Dict(:name => "mMean", :m => deepcopy(mean_image)),
    Dict(:name => "mWest", :m => deepcopy(west_image)),
]

#
# Defaults
x0 = deepcopy(f)
m = deepcopy(mean_image)
records = Array{Array{Tuple{Int,Float64,Array},1},1}()
for e in experiments
    name = e[:name]
    print("\n --- Running Experiment $name ---")
    println(
        " Values σ:",
        get(e, :σ, σ),
        " | τ:",
        get(e, :τ, τ),
        " | θ:",
        get(e, :θ, θ),
        " | γ:",
        get(e, :γ, γ),
    )
    #
    # Any Entry in the dictionary overwrites the above default
    @time o = ChambollePock(
        M,
        N,
        cost,
        get(e, :Iterate, x0),
        get(e, :ξ, zero_vector(N, get(e, :n, Λ(get(e, :m, m))))),
        get(e, :m, m),
        get(e, :n, Λ(get(e, :m, m))),
        proxFidelity,
        proxPriorDual,
        AdjDΛ;
        linearized_forward_operator=DΛ,
        primal_stepsize=get(e, :σ, σ),
        dual_stepsize=get(e, :τ, τ),
        relaxation=get(e, :θ, θ),
        acceleration=get(e, :γ, γ),
        relax=:dual,
        debug=[
            :Iteration,
            " | ",
            :Cost,
            "\n",
            export_primal_video ? DebugRenderAsy(video_folder, experiment_name * name) : "",
            100,
            :Stop,
        ],
        record=[:Iteration, :Cost, :Iterate],
        stopping_criterion=StopAfterIteration(get(e, :maxIter, 300)),
        variant=:linearized,
        return_state=true,
    )
    push!(records, get_record(o))
    if export_primal
        result_file = joinpath(results_folder, experiment_name * "-result.asy")
        asymptote_export_S2_data(result_file; data=get_solver_result(o))
        render_asymptote(result_file; render=asy_render_size)
    end
    if export_primal_video
        source = join_path(video_folder, experiment_name * name * "-vid-*.png")
        dest = join_path(results_folder, experiment_name * name * ".mp4")
        # run(`ffmpeg -framerate 15 -pattern_type glob -i $(source) -c:v libx264 -vf pad=ceil\(iw/2\)*2:ceil\(ih/2\)*2 -pix_fmt yuv420p $(dest)`)
    end
end
#
# Finalize - export costs
if export_table
    A = cat(
        first.(records[1]),
        [[r[2] for r in records[i]] for i in 1:length(records)]...;
        dims=2,
    )
    CSV.write(
        joinpath(results_folder, experiment_name * "-Result.csv"),
        DataFrame(A);
        header=["i", [e[:name] for e in experiments]...],
    )
end
