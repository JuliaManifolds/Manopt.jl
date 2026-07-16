using Manifolds, Manopt, Test

struct DummyState <: AbstractManoptSolverState end

@testset "Test callback utilities" begin
    # (a) providing a symbol and an array of symbol keys, expands the array
    cb = [:A => sin, [:B, :C] => cos]
    cb2 = Manopt.process_callbacks_arg(cb)
    @test cb2 == Dict(:A => sin, :B => cos, :C => cos)
    # Providing something else than these two yields an error
    @test_throws ArgumentError Manopt.process_callbacks_arg(["A" => sin])
    # test warning for unknown callbacks – GD does not have a :A callback
    @test_logs (:warn,) Manopt.process_callbacks_arg([:A => sin], GradientDescentState)

    @test Manopt._get_callbacks(DummyState(), Val(false)) == Dict{Symbol, Any}()

    M = Euclidean(2)
    f(M, x) = norm(x)^2
    dmp = DefaultManoptProblem(M, ManifoldCostObjective(f))
    @test Manopt._MANOPT_EMPTY_ANY_CALLBACK(:Any, dmp, DummyState(), 1) === nothing
end
