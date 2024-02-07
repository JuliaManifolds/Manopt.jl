using Manifolds
using PythonCall
include("benchmark_comparison.jl")

# This script requires optuna to be available through PythonCall
# You can install it for example using
# using CondaPkg
# ]conda add optuna

optuna = pyimport("optuna")

function test_objective(trial)
    x = trial.suggest_float("x", -100, 100)
    trial.report(abs(x), 1)
    if pyconvert(Bool, trial.should_prune().__bool__())
        throw(PyException(optuna.TrialPruned()))
    end
    return x^2
end

function test_study()
    study = optuna.create_study()
    # The optimization finishes after evaluating 1000 times or 3 seconds.
    study.optimize(test_objective; n_trials=1000, timeout=3)
    return println("Best params is $(study.best_params) with value $(study.best_value)")
end

struct TTsuggest_int
    suggestions::Dict{String,Int}
end
function (s::TTsuggest_int)(name::String, a, b)
    return s.suggestions[name]
end
struct TTsuggest_float
    suggestions::Dict{String,Float64}
end
function (s::TTsuggest_float)(name::String, a, b; log::Bool=false)
    return s.suggestions[name]
end
struct TTsuggest_categorical
    suggestions::Dict{String,Any}
end
function (s::TTsuggest_categorical)(name::String, vals)
    return s.suggestions[name]
end
struct TTreport
    reported_vals::Vector{Float64}
end
function (r::TTreport)(val, i)
    return push!(r.reported_vals, val)
end
struct TTshould_prune end
(::TTshould_prune)() = Py(false)
struct TracingTrial
    suggest_int::TTsuggest_int
    suggest_float::TTsuggest_float
    suggest_categorical::TTsuggest_categorical
    report::TTreport
    should_prune::TTshould_prune
end

"""
    ObjectiveData

ensure `pruning_losses` is actually somewhat realistic,
otherwise there is too little pruning (if values here are too high)
or too much pruning (if values here are too low)
regenerate using `lbfgs_compute_pruning_losses`
"""
mutable struct ObjectiveData{TObj,TGrad}
    obj::TObj
    grad::TGrad
    N_range::Vector{Int}
    gtol::Float64
    vts::Vector{AbstractVectorTransportMethod}
    retrs::Vector{AbstractRetractionMethod}
    manifold_constructors::Vector{Tuple{String,Any}}
    pruning_losses::Vector{Float64}
    manopt_stepsize::Vector{Tuple{String,Any}}
end

function (objective::ObjectiveData)(trial)
    mem_len = trial.suggest_int("mem_len", 2, 30)

    vt = objective.vts[pyconvert(
        Int,
        trial.suggest_categorical(
            "vector_transport_method", Vector(eachindex(objective.vts))
        ),
    )]
    retr = objective.retrs[pyconvert(
        Int,
        trial.suggest_categorical("retraction_method", Vector(eachindex(objective.retrs))),
    )]

    manifold_name, manifold_constructor = objective.manifold_constructors[pyconvert(
        Int,
        trial.suggest_categorical(
            "manifold", Vector(eachindex(objective.manifold_constructors))
        ),
    )]

    manopt_stepsize_name, manopt_stepsize_constructor = objective.manopt_stepsize[pyconvert(
        Int,
        trial.suggest_categorical(
            "manopt_stepsize", Vector(eachindex(objective.manopt_stepsize))
        ),
    )]

    local c1_val, c2_val
    if manopt_stepsize_name == "Wolfe-Powell"
        c1_val = pyconvert(
            Float64, trial.suggest_float("Wolfe-Powell c1", 1e-5, 1e-2; log=true)
        )
        c2_val =
            1.0 - pyconvert(
                Float64, trial.suggest_float("Wolfe-Powell 1-c2", 1e-4, 1e-2; log=true)
            )
    end

    loss = sum(objective.pruning_losses)

    # here iterate over problems we want to optimize for
    # from smallest to largest; pruning should stop the iteration early
    # if the hyperparameter set is not promising
    cur_i = 0
    for N in objective.N_range
        x0 = zeros(N)
        x0[1] = 1
        M = manifold_constructor(N)
        local ls
        if manopt_stepsize_name == "Wolfe-Powell"
            ls = manopt_stepsize_constructor(M, c1_val, c2_val)
        else
            ls = manopt_stepsize_constructor(M)
        end
        manopt_time, manopt_iters, manopt_obj = benchmark_time_state(
            ManoptQN(),
            M,
            N,
            objective.obj,
            objective.grad,
            x0,
            ls,
            pyconvert(Int, mem_len),
            objective.gtol;
            vector_transport_method=vt,
            retraction_method=retr,
        )
        # TODO: take objective_value into account for loss?
        loss -= objective.pruning_losses[cur_i + 1]
        loss += manopt_time
        trial.report(loss, cur_i)
        if pyconvert(Bool, trial.should_prune().__bool__())
            throw(PyException(optuna.TrialPruned()))
        end
        cur_i += 1
    end
    return loss
end

function lbfgs_study()
    Ns = [2^n for n in 1:3:16]
    ls_hz = LineSearches.HagerZhang()
    od = ObjectiveData(
        f_rosenbrock,
        g_rosenbrock!,
        Ns,
        1e-5,
        AbstractVectorTransportMethod[ParallelTransport(), ProjectionTransport()],
        [ExponentialRetraction(), ProjectionRetraction()],
        Tuple{String,Any}[("Sphere", N -> Manifolds.Sphere(N - 1))],
        zeros(Float64, eachindex(Ns)),
        Tuple{String,Any}[
            ("LS-HZ", M -> Manopt.LineSearchesStepsize(ls_hz)),
            ("Wolfe-Powell", (M, c1, c2) -> Manopt.WolfePowellLinesearch(M, c1, c2)),
        ],
    )
    pruning_losses = vcat(
        [15.95, 38.961, 74.9733, 411.8313333, 2561.789333333333, 3.7831363008333333e6],
        zeros(100),
    )
    pruning_losses = compute_pruning_losses(
        od,
        Dict("mem_len" => 4),
        Dict("Wolfe-Powell c1" => 1e-4, "Wolfe-Powell 1-c2" => 1e-3),
        Dict(
            "vector_transport_method" => 1,
            "retraction_method" => 1,
            "manifold" => 1,
            "manopt_stepsize" => 1,
        ),
    )
    od.pruning_losses = pruning_losses

    study = optuna.create_study(; study_name="L-BFGS")
    study.optimize(od; n_trials=1000, timeout=500)
    return println("Best params is $(study.best_params) with value $(study.best_value)")
end

function compute_pruning_losses(
    od::ObjectiveData,
    int_suggestions::Dict{String,Int},
    float_suggestions::Dict{String,Float64},
    categorical_suggestions::Dict{String,Int},
)
    tt = TracingTrial(
        TTsuggest_int(int_suggestions),
        TTsuggest_float(float_suggestions),
        TTsuggest_categorical(categorical_suggestions),
        TTreport(Float64[]),
        TTshould_prune(),
    )
    od(tt)
    return tt.report.reported_vals
end
