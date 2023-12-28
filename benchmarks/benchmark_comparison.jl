using Revise
using Optim, Manopt
using Manifolds
using LineSearches

using Profile
using ProfileView
using BenchmarkTools
using Plots

"""
    StopWhenGradientNormLess <: StoppingCriterion

A stopping criterion based on the current gradient norm.

# Constructor

    StopWhenGradientNormLess(ε::Float64)

Create a stopping criterion with threshold `ε` for the gradient, that is, this criterion
indicates to stop when [`get_gradient`](@ref) returns a gradient vector of norm less than `ε`.
"""
mutable struct StopWhenGradientInfNormLess <: StoppingCriterion
    threshold::Float64
    reason::String
    at_iteration::Int
    StopWhenGradientInfNormLess(ε::Float64) = new(ε, "", 0)
end
function (c::StopWhenGradientInfNormLess)(
    mp::AbstractManoptProblem, s::AbstractManoptSolverState, i::Int
)
    M = get_manifold(mp)
    if i == 0 # reset on init
        c.reason = ""
        c.at_iteration = 0
    end
    if (norm(get_gradient(s), Inf) < c.threshold) && (i > 0)
        c.reason = "The algorithm reached approximately critical point after $i iterations; the gradient norm ($(norm(M,get_iterate(s),get_gradient(s)))) is less than $(c.threshold).\n"
        c.at_iteration = i
        return true
    end
    return false
end
function Manopt.status_summary(c::StopWhenGradientInfNormLess)
    has_stopped = length(c.reason) > 0
    s = has_stopped ? "reached" : "not reached"
    return "|grad f|ₒₒ < $(c.threshold): $s"
end
Manopt.indicates_convergence(c::StopWhenGradientInfNormLess) = true
function Base.show(io::IO, c::StopWhenGradientInfNormLess)
    return print(
        io, "StopWhenGradientInfNormLess($(c.threshold))\n    $(status_summary(c))"
    )
end

function f_rosenbrock(x)
    result = 0.0
    for i in 1:2:length(x)
        result += (1.0 - x[i])^2 + 100.0 * (x[i + 1] - x[i]^2)^2
    end
    return result
end
function f_rosenbrock_manopt(::Euclidean, x)
    return f_rosenbrock(x)
end

optimize(f_rosenbrock, [0.0, 0.0], Optim.NelderMead())

function g_rosenbrock!(storage, x)
    for i in 1:2:length(x)
        storage[i] = -2.0 * (1.0 - x[i]) - 400.0 * (x[i + 1] - x[i]^2) * x[i]
        storage[i + 1] = 200.0 * (x[i + 1] - x[i]^2)
    end
    return storage
end

optimize(f_rosenbrock, g_rosenbrock!, [0.0, 0.0], LBFGS())

function g_rosenbrock_manopt!(::Euclidean, storage, x)
    g_rosenbrock!(storage, x)
    if isnan(x[1])
        error("nan")
    end
    return storage
end

M = Euclidean(2)
Manopt.NelderMead(M, f_rosenbrock_manopt)

qn_opts = quasi_Newton(
    M,
    f_rosenbrock_manopt,
    g_rosenbrock_manopt!,
    [0.0, 0.0];
    evaluation=InplaceEvaluation(),
    return_state=true,
)

function test_f(f_manopt, g_manopt!, x0, N::Int)
    M = Euclidean(N)
    return quasi_Newton(
        M, f_manopt, g_manopt!, x0; evaluation=InplaceEvaluation(), return_state=true
    )
end

function prof()
    N = 32
    x0 = zeros(N)
    test_f(f_rosenbrock_manopt, g_rosenbrock_manopt!, x0, N)

    Profile.clear()
    @profile for i in 1:10000
        test_f(f_rosenbrock_manopt, g_rosenbrock_manopt!, x0, N)
    end
    return ProfileView.view()
end

function generate_cmp(f, g!, f_manopt, g_manopt!)
    plt = plot()
    xlabel!("dimension")
    ylabel!("time per iteration [ms]")
    title!("Optimization times for $f")

    times_manopt = Float64[]
    times_optim = Float64[]

    N_vals = [2^n for n in 1:3:16]
    strong_wolfe_ls = LineSearches.StrongWolfe()

    gtol = 1e-6
    println("Benchmarking $f for gtol=$gtol")
    for N in N_vals
        println("Benchmarking for N=$N")
        M = Euclidean(N)
        method_optim = LBFGS(; linesearch=strong_wolfe_ls)

        x0 = zeros(N)
        manopt_sc = StopWhenGradientInfNormLess(gtol) | StopAfterIteration(1000)
        bench_manopt = @benchmark quasi_Newton(
            $M,
            $f_manopt,
            $g_manopt!,
            $x0;
            # this causes an error for larger N
            #stepsize=$(Manopt.LineSearchesStepsize(strong_wolfe_ls)),
            evaluation=$(InplaceEvaluation()),
            memory_size=10,
            stopping_criterion=$(manopt_sc),
        )

        manopt_state = quasi_Newton(
            M,
            f_manopt,
            g_manopt!,
            x0;
            #stepsize=Manopt.LineSearchesStepsize(strong_wolfe_ls),
            evaluation=InplaceEvaluation(),
            return_state=true,
            memory_size=10,
            stopping_criterion=manopt_sc,
        )
        manopt_iters = get_count(manopt_state, :Iterations)
        push!(times_manopt, median(bench_manopt.times) / (1000 * manopt_iters))
        println("Manopt.jl iterations: $(manopt_iters)")

        options_optim = Optim.Options(; g_tol=gtol)
        bench_optim = @benchmark optimize($f, $g!, $x0, $method_optim, $options_optim)

        optim_state = optimize(f_rosenbrock, g_rosenbrock!, x0, method_optim, options_optim)
        push!(times_optim, median(bench_optim.times) / (1000 * optim_state.iterations))
        println("Optim.jl  iterations: $(optim_state.iterations)")
    end
    plot!(N_vals, times_manopt; label="Manopt.jl", xaxis=:log, yaxis=:log)
    plot!(N_vals, times_optim; label="Optim.jl", xaxis=:log, yaxis=:log)
    xticks!(N_vals, string.(N_vals))

    return plt
end

generate_cmp(f_rosenbrock, g_rosenbrock!, f_rosenbrock_manopt, g_rosenbrock_manopt!)
