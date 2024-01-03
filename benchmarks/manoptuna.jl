
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

struct TTsuggest_int end
function (::TTsuggest_int)(name::String, a, b)
    if name == "mem_len"
        return 4
    end
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
    report::TTreport
    should_prune::TTshould_prune
end

function lbfgs_compute_pruning_losses()
    tt = TracingTrial(TTsuggest_int(), TTreport(Float64[]), TTshould_prune())
    lbfgs_objective(tt)
    return tt.report.reported_vals
end

function lbfgs_objective(trial)
    mem_len = trial.suggest_int("mem_len", 2, 30)
    gtol = 1e-5
    manifold_name = :Sphere
    ls_hz = LineSearches.HagerZhang()

    N_range = [2^n for n in 1:3:16]

    # TODO: ensure this actually somewhat realistic,
    # otherwise there is too little pruning (if values here are too low)
    # or too much pruning (if values here are too high)
    # regenerate using
    # prunining_losses = lbfgs_compute_pruning_losses()
    # *but* with zeroed-out prunning_losses
    prunining_losses = [56.403, 69.438, 96.36449999999999, 409.749, 2542.482, 6.366860307e6]

    loss = sum(prunining_losses)

    cur_i = 0
    for N in N_range
        x0 = zeros(N)
        x0[1] = 1
        manopt_time, manopt_iters, manopt_obj = benchmark_time_state(
            ManoptQN(),
            manifold_name,
            N,
            f_rosenbrock_manopt,
            g_rosenbrock_manopt!,
            x0,
            Manopt.LineSearchesStepsize(ls_hz),
            pyconvert(Int, mem_len),
            gtol,
        )
        loss -= prunining_losses[cur_i + 1]
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
    study = optuna.create_study(; study_name="L-BFGS")
    study.optimize(lbfgs_objective; n_trials=1000, timeout=200)
    return println("Best params is $(study.best_params) with value $(study.best_value)")
end
