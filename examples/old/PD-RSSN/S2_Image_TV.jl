#
# Denoise an S2 valued image demonstrating the effect of improper initialisation of dual iterate ξ
#
# L2-TV functional with isotropic TV
#
# This example is used in Section 6.2 of
#
# > W. Diepeveen, J. Lellmann:
# > _An Inexact Semismooth Newton Method on Riemannian Manifolds with Application to Duality-Based Total Variation Denoising_,
# > SIAM Journal on Imaging Sciences, 2021.
# > doi: [10.1137/21M1398513](https://doi.org/10.1137/21M1398513)
#
using CSV, DataFrames, LinearAlgebra, Manopt, Manifolds
#
# Settings
experiment_name = "S2_Image_CP"
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
signal_section_size = 4
σ = 0.25
τ = 0.25
γ = 0.2
θ = 1.0
α = 0.5
#
# Data
pixelM = Sphere(2)
base = [1.0, 0.0, 0.0]
X = π / 4 * [0.0, 1.0, 0.0]
# Generate a signal with two sections
p1 = exp(pixelM, base, X)
p2 = exp(pixelM, base, -X)
f = vcat(
    fill(p1, (signal_section_size, 2 * signal_section_size)),
    fill(p2, (signal_section_size, 2 * signal_section_size)),
)
#
# load TV model
#
include("Signal_TV_commons.jl")

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

#
# Build Experiments
experiments = [
    Dict(:name => "base"),
    # Dict(:name => "mWest", :m => deepcopy(west_image)),
]

#
# Defaults
x0 = deepcopy(f)
m = deepcopy(fill(base, size(f)))
ξ0 = zero_vector(M2, m2(m))
records = Array{Array{Tuple{Int,Float64,Array,Array},1},1}()
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
        M2,
        cost,
        get(e, :Iterate, x0),
        get(e, :X, ξ0),
        get(e, :m, m),
        get(e, :n, m2(get(e, :m, m))),
        prox_F,
        prox_G_dual,
        adjoint_DΛ;
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
            # export_primal_video ? DebugRenderAsy(video_folder, experiment_name * name) : "",
            10,
            :Stop,
        ],
        record=[:Iteration, :Cost, :Iterate, :X],
        stopping_criterion=StopAfterIteration(get(e, :maxIter, 100)),
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

    # TODO get dual
    @time o_pdrssn = primal_dual_semismooth_Newton(
        M,
        M2,
        cost,
        get(e, :Iterate, x0),
        get(e, :X, get_record(o)[1][4]),
        get(e, :m, m),
        get(e, :n, m2(get(e, :m, m))),
        prox_F,
        Dprox_F,
        prox_G_dual,
        Dprox_G_dual,
        DΛ,
        adjoint_DΛ;
        primal_stepsize=get(e, :σ, σ),
        dual_stepsize=get(e, :τ, τ),
        debug=[
            :Iteration,
            " | ",
            :Cost,
            "\n",
            # export_primal_video ? DebugRenderAsy(video_folder, experiment_name * name) : "",
            # 10,
            :Stop,
        ],
        record=[:Iteration, :Cost, :Iterate],
        stopping_criterion=StopAfterIteration(get(e, :maxIter, 10)),
        # variant=:linearized,
        return_state=true,
    )
end
#
# Finalize - export costs
# if export_table
#     A = cat(
#         first.(records[1]),
#         [[r[2] for r in records[i]] for i in 1:length(records)]...;
#         dims=2,
#     )
#     CSV.write(
#         joinpath(results_folder, experiment_name * "-Result.csv"),
#         DataFrame(A);
#         header=["i", [e[:name] for e in experiments]...],
#     )
# end
