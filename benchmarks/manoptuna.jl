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
    suggest_categorical::TTsuggest_categorical
    report::TTreport
    should_prune::TTshould_prune
end

function lbfgs_compute_pruning_losses()
    # suggestions here need to be kept updated with what `lbfgs_objective` expects
    tt = TracingTrial(
        TTsuggest_int(Dict("mem_len" => 4)),
        TTsuggest_categorical(Dict("vector_transport_method" => 1)),
        TTreport(Float64[]),
        TTshould_prune(),
    )
    lbfgs_objective(tt)
    return tt.report.reported_vals
end

struct ObjectiveData{TObj,TGrad}
    obj::TObj
    grad::TGrad
    N_range::Vector{Int}
    gtol::Float64
end

function (objective::ObjectiveData)(trial)
    mem_len = trial.suggest_int("mem_len", 2, 30)
    manifold_name = :Sphere
    ls_hz = LineSearches.HagerZhang()

    vts = [ParallelTransport(), ProjectionTransport()]
    vt = vts[pyconvert(Int, trial.suggest_categorical("vector_transport_method", (1, 2)))]

    # TODO: ensure this actually somewhat realistic,
    # otherwise there is too little pruning (if values here are too high)
    # or too much pruning (if values here are too low)
    # regenerate using
    # pruning_losses = lbfgs_compute_pruning_losses()
    # *but* with zeroed-out pruning_losses
    # padded with zeros for convenience
    pruning_losses = vcat(
        [15.95, 38.961, 74.9733, 411.8313333, 2561.789333333333, 3.7831363008333333e6],
        zeros(100),
    )

    loss = sum(pruning_losses)

    # here iterate over problems we want to optimize for
    # from smallest to largest; pruning should stop the iteration early
    # if the hyperparameter set is not promising
    cur_i = 0
    for N in objective.N_range
        x0 = zeros(N)
        x0[1] = 1
        manopt_time, manopt_iters, manopt_obj = benchmark_time_state(
            ManoptQN(),
            manifold_name,
            N,
            objective.obj,
            objective.grad,
            x0,
            Manopt.LineSearchesStepsize(ls_hz),
            pyconvert(Int, mem_len),
            objective.gtol;
            vector_transport_method=vt,
        )
        # TODO: take objective_value into account for loss?
        loss -= pruning_losses[cur_i + 1]
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
    od = ObjectiveData(f_rosenbrock, g_rosenbrock!, [2^n for n in 1:3:16], 1e-5)
    study = optuna.create_study(; study_name="L-BFGS")
    study.optimize(od; n_trials=1000, timeout=500)
    return println("Best params is $(study.best_params) with value $(study.best_value)")
end
