using LineSearches
using Manifolds, Manopt
using Test

@testset "LineSearches.jl compatibility" begin
    p = [1.0, 100.0]
    function rosenbrock(::AbstractManifold, x)
        val = zero(eltype(x))
        for i in 1:(length(x) - 1)
            val += (p[1] - x[i])^2 + p[2] * (x[i + 1] - x[i]^2)^2
        end
        return val
    end
    function rosenbrock_grad!(M::AbstractManifold, storage, x)
        storage .= 0.0
        for i in 1:(length(x) - 1)
            storage[i] += -2.0 * (p[1] - x[i]) - 4.0 * p[2] * (x[i + 1] - x[i]^2) * x[i]
            storage[i + 1] += 2.0 * p[2] * (x[i + 1] - x[i]^2)
        end
        project!(M, storage, x, storage)
        return storage
    end

    n_dims = 5
    M = Manifolds.Sphere(n_dims)
    x0 = vcat(zeros(n_dims - 1), 1.0)
    ls_hz = Manopt.LineSearchesStepsize(M, LineSearches.HagerZhang())
    x_opt = quasi_Newton(
        M,
        rosenbrock,
        rosenbrock_grad!,
        x0;
        stepsize = ls_hz,
        debug = [],
        evaluation = InplaceEvaluation(),
        stopping_criterion = StopAfterIteration(1000) | StopWhenGradientNormLess(1.0e-6),
        return_state = true,
    )

    @test rosenbrock(M, get_iterate(x_opt)) < 1.503084
    @test startswith(sprint(show, ls_hz), "LineSearchesStepsize(HagerZhang")

    # make sure get_last_stepsize works
    mgo = ManifoldGradientObjective(
        rosenbrock, rosenbrock_grad!; evaluation = InplaceEvaluation()
    )
    mp = DefaultManoptProblem(M, mgo)
    @test get_last_stepsize(mp, x_opt, 1) > 0.0

    # this tests catching LineSearchException
    @test_throws LineSearchException ls_hz(mp, x_opt, 1, NaN * zero_vector(M, x0))

    # test rethrowing errors
    function rosenbrock_throw(::AbstractManifold, x)
        return error("test exception")
    end
    mgo_throw = Manopt.ManifoldGradientObjective(
        rosenbrock_throw, rosenbrock_grad!; evaluation = InplaceEvaluation()
    )
    mp_throw = DefaultManoptProblem(M, mgo_throw)
    st_qn = QuasiNewtonState(M; p = x0)
    initialize_solver!(mp, st_qn)
    ls_mt = Manopt.LineSearchesStepsize(M, LineSearches.MoreThuente())
    @test_throws ErrorException ls_mt(mp_throw, st_qn, 1; fp = rosenbrock(M, x0))

    # test max stepsize limit enforcement
    @test ls_hz(mp, st_qn, 1, [1.0, 2.0, 3.0, 4.0, 0.0]; stop_when_stepsize_exceeds = 0.1) == 0.1

    @testset "max stepsize limit setting" begin
        lss = [
            LineSearches.MoreThuente(),
            LineSearches.HagerZhang(),
        ]
        for ls in lss
            nls = Manopt.linesearches_set_max_alpha(ls, 0.5)
            @test Manopt.linesearches_get_max_alpha(nls) == 0.5
            nls2 = Manopt.linesearches_set_max_alpha(ls, Inf)
            @test Manopt.linesearches_get_max_alpha(nls2) == Inf
        end
    end
end
