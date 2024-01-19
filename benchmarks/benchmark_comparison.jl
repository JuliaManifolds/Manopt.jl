using Revise
using Optim, Manopt
using Manifolds
using LineSearches
using LinearAlgebra

using Profile
using ProfileView
using BenchmarkTools
using Plots
using ManoptExamples
using ImprovedHagerZhangLinesearch

norm_inf(M::AbstractManifold, p, X) = norm(X, Inf)

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

abstract type AbstractOptimConfig end
struct ManoptQN <: AbstractOptimConfig end

function benchmark_time_state(
    ::ManoptQN,
    manifold_name::Symbol,
    N,
    f_manopt,
    g_manopt!,
    x0,
    stepsize,
    mem_len::Int,
    gtol::Real;
    kwargs...,
)
    manopt_sc = StopWhenGradientNormLess(gtol; norm=norm_inf) | StopAfterIteration(1000)
    M = manifold_maker(manifold_name, N, :Manopt)
    mem_len = min(mem_len, manifold_dimension(M))
    bench_manopt = @benchmark quasi_Newton(
        $M,
        $f_manopt,
        $g_manopt!,
        $x0;
        stepsize=$(stepsize),
        evaluation=$(InplaceEvaluation()),
        memory_size=$mem_len,
        stopping_criterion=$(manopt_sc),
        $kwargs...,
    )
    manopt_state = quasi_Newton(
        M,
        f_manopt,
        g_manopt!,
        x0;
        stepsize=stepsize,
        evaluation=InplaceEvaluation(),
        return_state=true,
        memory_size=mem_len,
        stopping_criterion=manopt_sc,
        kwargs...,
    )
    iters = get_count(manopt_state, :Iterations)
    final_val = f_manopt(M, manopt_state.p)
    return median(bench_manopt.times) / 1000, iters, final_val
end

struct OptimQN <: AbstractOptimConfig end

function benchmark_time_state(
    ::OptimQN, manifold_name, N, f, g!, x0, stepsize, mem_len::Int, gtol::Real
)
    mem_len = min(mem_len, manifold_dimension(manifold_maker(manifold_name, N, :Manopt)))
    options_optim = Optim.Options(; g_tol=gtol)
    method_optim = LBFGS(;
        m=mem_len, linesearch=stepsize, manifold=manifold_maker(manifold_name, N, :Optim)
    )

    bench_optim = @benchmark optimize($f, $g!, $x0, $method_optim, $options_optim)

    optim_state = optimize(f, g!, x0, method_optim, options_optim)
    iters = optim_state.iterations
    final_val = optim_state.minimum
    return median(bench_optim.times) / 1000, iters, final_val
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
            x0 = zeros(N)
            x0[1] = 1
            manopt_time, manopt_iters, manopt_obj = benchmark_time_state(
                ManoptQN(),
                manifold_name,
                N,
                f_manopt,
                g_manopt!,
                x0,
                HagerZhangLinesearch(M),
                mem_len,
                gtol;
                vector_transport_method=ParallelTransport(),
            )

            push!(times_manopt, manopt_time)
            println("Manopt.jl time: $(manopt_time) ms")
            println("Manopt.jl iterations: $(manopt_iters)")
            println("Manopt.jl objective: $(manopt_obj)")

            optim_time, optim_iters, optim_obj = benchmark_time_state(
                OptimQN(), manifold_name, N, f, g!, x0, ls_hz, mem_len, gtol
            )
            println("Optim.jl  time: $(optim_time) ms")
            push!(times_optim, optim_time)
            println("Optim.jl  iterations: $(optim_iters)")
            println("Optim.jl  objective: $(optim_obj)")
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

function generate_rayleigh_problem(N::Int)
    A = Symmetric(randn(N, N))
    f_manopt = ManoptExamples.RayleighQuotientCost(A)
    g_manopt! = ManoptExamples.RayleighQuotientGrad!!(A)
    M = Manifolds.Sphere(N - 1)
    f(x) = f_manopt(M, x)
    g!(storage, x) = g_manopt!(M, storage, x)
    return (f, g!, f_manopt, g_manopt!)
end
# generate_cmp(generate_rayleigh_problem, manifold_names=[:Sphere], mem_len=4)

function test_case_manopt()
    N = 128
    mem_len = 1
    M = Manifolds.Sphere(N - 1)
    ls_hz = LineSearches.HagerZhang()

    x0 = zeros(N)
    x0[1] = 1
    manopt_sc = StopWhenGradientNormLess(1e-6; norm=norm_inf) | StopAfterIteration(1000)

    return quasi_Newton(
        M,
        f_rosenbrock_manopt,
        g_rosenbrock_manopt!,
        x0;
        stepsize=HagerZhangLinesearch(M),
        evaluation=InplaceEvaluation(),
        vector_transport_method=ProjectionTransport(),
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
    x0[1] = 0
    optim_state = optimize(f_rosenbrock, g_rosenbrock!, x0, method_optim, options_optim)
    return optim_state
end
