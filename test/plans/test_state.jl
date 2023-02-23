using Manifolds, Manopt, Test, ManifoldsBase
import Manopt: set_manopt_parameter!
using Dates

struct TestStateProblem{M<:AbstractManifold} <: AbstractManoptProblem{M} end
mutable struct TestState <: AbstractManoptSolverState
    storage::Vector{Float64}
end
TestState() = TestState([])
set_manopt_parameter!(s::TestState, ::Val, v) = s

@testset "Manopt Solver State" begin
    @testset "Generic State" begin
        M = Euclidean(3)
        pr = TestStateProblem{typeof(M)}()
        s = TestState()
        @test repr(Manopt.ReturnSolverState(s)) == "ReturnSolverState($s)"
        @test Manopt.status_summary(Manopt.ReturnSolverState(s)) == "TestState(Float64[])"
        a = ArmijoLinesearch(M; initial_stepsize=1.0)
        @test get_last_stepsize(pr, s, a) == 1.0
        @test get_initial_stepsize(a) == 1.0
        set_manopt_parameter!(s, :Dummy, 1)
    end

    @testset "Decreasing Stepsize" begin
        dec_step = DecreasingStepsize(;
            length=10.0, factor=1.0, subtrahend=0.0, exponent=1.0
        )
        @test get_initial_stepsize(dec_step) == 10.0
        M = Euclidean(3)
        pr = TestStateProblem{typeof(M)}()
        @test dec_step(pr, TestState(), 1) == 10.0
        @test dec_step(pr, TestState(), 2) == 5.0
    end

    @testset "Decorator State" begin
        s = TestState(zeros(3))
        r = RecordSolverState(s, RecordIteration())
        d = DebugSolverState(s, DebugIteration())
        ret = Manopt.ReturnSolverState(s)
        dr = DebugSolverState(r, DebugIteration())
        M = Euclidean()

        @test has_record(s) == has_record(d)
        @test !has_record(s)
        @test has_record(r)
        @test has_record(r) == has_record(dr)

        @test is_state_decorator(r)
        @test is_state_decorator(dr)
        @test is_state_decorator(d)
        @test !is_state_decorator(s)

        @test dispatch_state_decorator(r) === Val(true)
        @test dispatch_state_decorator(dr) === Val(true)
        @test dispatch_state_decorator(ret) === Val(true)
        @test dispatch_state_decorator(d) === Val(true)
        @test dispatch_state_decorator(s) === Val(false)

        @test get_state(r) == s
        @test get_state(dr) == s
        @test get_state(d) == s
        @test get_state(s) == s
        @test Manopt._get_state(s, Val(false)) == s

        @test Manopt._extract_val(Val(true))
        @test !Manopt._extract_val(Val(false))

        @test_throws ErrorException get_gradient(s)
        @test_throws ErrorException get_gradient(r)
        @test_throws ErrorException get_iterate(s)
        @test_throws ErrorException get_iterate(r)
        @test_throws ErrorException set_iterate!(s, M, 0)
        @test_throws ErrorException set_iterate!(r, M, 0)
    end
    @testset "Iteration and Gradient setters" begin
        M = Euclidean(3)
        s1 = NelderMeadState(M)
        s2 = GradientDescentState(M)
        p = 3.0 * ones(3)
        X = ones(3)
        d1 = DebugSolverState(s1, DebugIteration())
        set_iterate!(d1, M, p)
        @test d1.state.p == 3 * ones(3)
        @test_throws ErrorException set_gradient!(d1, M, p, X)

        d2 = DebugSolverState(s2, DebugIteration())
        set_iterate!(d2, M, p)
        @test d2.state.p == 3 * ones(3)
        set_gradient!(d2, M, p, X)
        @test d2.state.X == ones(3)
    end
end
