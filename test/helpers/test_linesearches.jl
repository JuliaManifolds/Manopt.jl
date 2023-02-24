
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
    for ils in [
        LineSearches.InitialStatic(),
        LineSearches.InitialPrevious(),
        LineSearches.InitialQuadratic(),
        LineSearches.InitialConstantChange(),
    ]
        ls_hz = Manopt.LineSearchesStepsize(M, LineSearches.HagerZhang(), ils)
        x_opt = quasi_Newton(
            M,
            rosenbrock,
            rosenbrock_grad!,
            x0;
            stepsize=ls_hz,
            evaluation=InplaceEvaluation(),
            stopping_criterion=StopAfterIteration(1000) | StopWhenGradientNormLess(1e-6),
            return_state=true,
        )

        @test rosenbrock(M, get_iterate(x_opt)) < 1.503084
        @test startswith(sprint(show, ls_hz), "LineSearchesStepsize(HagerZhang")

        mgo = ManifoldGradientObjective(rosenbrock, rosenbrock_grad!)
        mp = DefaultManoptProblem(M, mgo)
        @test get_last_stepsize(mp, x_opt, x_opt.stepsize, 1) == x_opt.stepsize.alpha

        stepsize_storage = Manopt.StepsizeStorage(M, ls_hz; p_init=x0)
        # this tests catching LineSearchException
        @test iszero(stepsize_storage(mp, x_opt, 1, NaN * zero_vector(M, x0)))
    end
end
