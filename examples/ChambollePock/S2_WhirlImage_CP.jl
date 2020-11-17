"""
Denoise an S2 valued image with TV using the linearized Chambolle-Pock algorithm.
"""

using Manopt, Manifolds
using Images, CSV, DataFrames, LinearAlgebra

#
# Settings
experiment_name = "S2_WhirlImage_CP"
export_orig = false
export_primal = true
export_primal_video = false
export_table = true
asy_render_size = 2
#
# Automatic Settings
current_folder = @__DIR__
export_any = export_orig || export_primal || export_primal_video || export_table
results_folder = joinpath(current_folder,"Image_TV")
video_folder = joinpath(results_folder,"video")
# Create folder if we have an export
(export_any && !isdir(results_folder)) && mkdir(results_folder)
(export_primal_video && !isdir(video_folder)) && mkdir(video_folder)
#
# Experiment Parameters
σ = 0.35
τ = 0.35
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
    orig_file = joinpath(results_folder, experiment_name*"-orig.asy")
    asymptote_export_S2_data(orig_file,data=f)
    render_asymptote(orig_file; render = asy_render_size)
end
struct DebugRenderAsy <: DebugAction
    name::String
    folder::String
    DebugRenderAsy(f,n) = new(n,f)
end
function (d::DebugRenderAsy)(p::P,o::ChambollePockOptions, i::Int) where {P <: PrimalDualProblem}
    if i >=0
        orig_file = joinpath(d.folder, d.name*"-vid-$(lpad(i[1],7,"0")).asy")
        asymptote_export_S2_data(orig_file,data=f)
        render_asymptote(orig_file; render = asy_render_size)
    end
end

#
# Experiments (entries overwrite defaults from below)
#
data_mean = mean(pixelM, vec(f));
mean_image = fill(data_mean,size(f))
west_image = fill([1., 0., 0.], size(f))
front_image = fill([0., 1., 0.], size(f))
north_image = fill([0., 0., 1.], size(f))
south_image = fill([0., 0., -1.], size(f))

# Compute patch mean
pW = 1
function patchMean(f)
    mP = deepcopy(f)
    for i=1:64, j=1:64
        sub = f[max(i-pW,1):min(i+pW,64),max(j-pW,1):min(j+pW,64)]
        mP[i,j] = mean(pixelM,vec(sub))
    end
    return mP
end
mP = patchMean(f)
#
# Build Experiments
#

experiments = [
     Dict( :name => "mMean", :m => deepcopy(mean_image)),
     Dict( :name => "mWest", :m => deepcopy(west_image) ),
]

#
# Defaults
x0 = deepcopy(f)
m = mP
ξ0 = ProductRepr(zero_tangent_vector(M2,m2(m)), zero_tangent_vector(M2,m2(m)))
records = Array{Array{Tuple{Int,Float64,Array},1},1}()
for e in experiments
    name = e[:name]
    print("\n --- Running Experiment $name ---")
    println(" Values σ:",get(e, :σ, σ), " | τ:", get(e,:τ,τ), " | θ:", get(e,:θ,θ)," | γ:",get(e,:γ,γ))
    #
    # Any Entry in the dictionary overwrites the above default
    @time o = ChambollePock(M, N, cost,
        get(e,:x, x0),
        get(e, :ξ, zero_tangent_vector(N, get(e, :n, Λ( get(e, :m, m) ) ))),
        get(e, :m, m),
        get(e, :n, Λ( get(e, :m, m) ) ),
        proxFidelity, proxPriorDual,
        DΛ,AdjDΛ;
        primal_stepsize = get(e, :σ, σ),
        dual_stepsize = get(e, :τ, τ),
        relaxation = get(e, :θ, θ),
        acceleration = get(e, :γ, γ),
        relax = :dual,
        debug = [:Iteration," | ", :Cost,"\n",
        export_primal_video ? DebugRenderAsy(video_folder, experiment_name * name) : "", 1,:Stop],
        record = [:Iteration, :Cost, :Iterate],
        stopping_criterion = StopAfterIteration(get(e, :maxIter, 300)),
        variant = :linearized,
    )
    push!(records, get_record(r))
    if export_primal
        result_file = joinpath(results_folder, experiment_name*"-result.asy")
        asymptote_export_S2_data(result_file,data=get_solver_result(o))
        render_asymptote(result_file; render = asy_render_size)
    end
    if export_primal_video
        source = join_path(video_folder, experiment_name*name*"-vid-*.png")
        dest = join_path(results_folder, experiment_name*name*".mp4")
        #run(`ffmpeg -framerate 15 -pattern_type glob -i $(source) -c:v libx264 -vf pad=ceil\(iw/2\)*2:ceil\(ih/2\)*2 -pix_fmt yuv420p $(dest)`)
    end
end
#
# Finalize - export costs
if export_table
    A = cat( first.(records[1]),
        [ [r[2] for r in records[i]] for i=1:length(records) ] ...
        ; dims=2 )
    CSV.write(joinpath(results_folder, experiment_name*"-Result.csv"), DataFrame(A),
        header=["i", [ e[:name] for e in experiments ]...] )
end