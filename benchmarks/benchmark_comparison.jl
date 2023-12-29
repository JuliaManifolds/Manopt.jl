using Revise
using Optim, Manopt
using Manifolds
using LineSearches

using Profile
using ProfileView
using BenchmarkTools
using Plots

"""
    StopWhenGradientInfNormLess <: StoppingCriterion

A stopping criterion based on the current gradient infinity norm in a basis arbitrarily
chosen for each manifold.

# Constructor

    StopWhenGradientInfNormLess(ε::Float64)

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
function f_rosenbrock_manopt(::AbstractManifold, x)
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

function g_rosenbrock_manopt!(M::AbstractManifold, storage, x)
    g_rosenbrock!(storage, x)
    if isnan(x[1])
        error("nan")
    end
    riemannian_gradient!(M, storage, x, storage)
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
    @profile for i in 1:100000
        test_f(f_rosenbrock_manopt, g_rosenbrock_manopt!, x0, N)
    end
    return ProfileView.view()
end

function manifold_maker(name::Symbol, N, lib::Symbol)
    if lib === :Manopt
        if name === :Euclidean
            return Euclidean(N)
        elseif name === :Sphere
            return Manifolds.Sphere(N - 1)
        end
    elseif lib === :Optim
        if name === :Euclidean
            return Optim.Flat()
        elseif name === :Sphere
            return Optim.Sphere()
        end
    else
        error("Unknown library: $lib")
    end
end

function generate_cmp(
    problem_for_N; mem_len::Int=2, manifold_names=[:Euclidean, :Sphere], gtol::Real=1e-5
)
    plt = plot()
    xlabel!(plt, "dimension")
    ylabel!(plt, "time [ms]")
    title!(plt, "Optimization times")

    N_vals = [2^n for n in 1:3:16]
    ls_hz = LineSearches.HagerZhang()

    for manifold_name in manifold_names
        times_manopt = Float64[]
        times_optim = Float64[]

        println("Benchmarking for gtol=$gtol on $manifold_name")
        for N in N_vals
            f, g!, f_manopt, g_manopt! = problem_for_N(N)
            println("Benchmarking for N=$N, f=$(typeof(f))")
            M = manifold_maker(manifold_name, N, :Manopt)
            method_optim = LBFGS(;
                m=mem_len,
                linesearch=ls_hz,
                manifold=manifold_maker(manifold_name, N, :Optim),
            )

            x0 = zeros(N)
            x0[1] = 1
            manopt_sc = StopWhenGradientInfNormLess(gtol) | StopAfterIteration(1000)
            bench_manopt = @benchmark quasi_Newton(
                $M,
                $f_manopt,
                $g_manopt!,
                $x0;
                stepsize=$(Manopt.LineSearchesStepsize(ls_hz)),
                evaluation=$(InplaceEvaluation()),
                memory_size=$mem_len,
                stopping_criterion=$(manopt_sc),
            )

            manopt_state = quasi_Newton(
                M,
                f_manopt,
                g_manopt!,
                x0;
                stepsize=Manopt.LineSearchesStepsize(ls_hz),
                evaluation=InplaceEvaluation(),
                return_state=true,
                memory_size=mem_len,
                stopping_criterion=manopt_sc,
            )
            manopt_iters = get_count(manopt_state, :Iterations)
            push!(times_manopt, median(bench_manopt.times) / 1000)
            println("Manopt.jl time: $(median(bench_manopt.times) / 1000) ms")
            println("Manopt.jl iterations: $(manopt_iters)")
            println("Manopt.jl objective: $(f(manopt_state.p))")

            options_optim = Optim.Options(; g_tol=gtol)
            bench_optim = @benchmark optimize($f, $g!, $x0, $method_optim, $options_optim)

            optim_state = optimize(f, g!, x0, method_optim, options_optim)
            println("Optim.jl  time: $(median(bench_optim.times) / 1000) ms")
            push!(times_optim, median(bench_optim.times) / 1000)
            println("Optim.jl  iterations: $(optim_state.iterations)")
            println("Optim.jl  objective: $(optim_state.minimum)")
        end
        plot!(
            plt,
            N_vals,
            times_manopt;
            label="Manopt.jl ($manifold_name)",
            xaxis=:log,
            yaxis=:log,
        )
        plot!(
            plt,
            N_vals,
            times_optim;
            label="Optim.jl ($manifold_name)",
            xaxis=:log,
            yaxis=:log,
        )
    end
    xticks!(plt, N_vals, string.(N_vals))

    return plt
end

# generate_cmp(N -> (f_rosenbrock, g_rosenbrock!, f_rosenbrock_manopt, g_rosenbrock_manopt!), mem_len=4)
# function gsn_problem_for_cmp(N)
#     (f, g!) = make_gsn_problem(N, div(N, 10))
#     return (f, g!, f, g!)
# end
# generate_cmp(gsn_problem_for_cmp, manifold_names=[:Sphere], mem_len=4)

function test_case_manopt()
    N = 2^16
    mem_len = 2
    M = Manifolds.Sphere(N - 1)
    ls_hz = LineSearches.HagerZhang()

    x0 = zeros(N)
    x0[1] = 1
    manopt_sc = StopWhenGradientInfNormLess(1e-6) | StopAfterIteration(1000)

    return quasi_Newton(
        M,
        f_rosenbrock_manopt,
        g_rosenbrock_manopt!,
        x0;
        stepsize=Manopt.LineSearchesStepsize(ls_hz),
        evaluation=InplaceEvaluation(),
        return_state=true,
        memory_size=mem_len,
        stopping_criterion=manopt_sc,
    )
end

function test_case_optim()
    N = 4
    mem_len = 2
    ls_hz = LineSearches.HagerZhang()
    method_optim = LBFGS(; m=mem_len, linesearch=ls_hz, manifold=Optim.Flat())
    options_optim = Optim.Options(; g_tol=1e-6)

    x0 = zeros(N)
    x0[1] = 1
    optim_state = optimize(f_rosenbrock, g_rosenbrock!, x0, method_optim, options_optim)
    return optim_state
end
