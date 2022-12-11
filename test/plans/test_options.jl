using Manifolds, Manopt, Test, ManifoldsBase

using Dates

struct TestProblem1{Teval<:AbstractEvaluationType} <: Problem{Teval} end
mutable struct TestOptions1 <: Options
    storage::Vector{Float64}
end

@testset "generic Options test" begin
    p = TestProblem1{MutatingEvaluation}()
    o = TestOptions1([1.0, 2.0])
    a = ArmijoLinesearch(Euclidean(3); initial_stepsize=1.0)
    @test get_last_stepsize(p, o, a) == 1.0
    @test get_initial_stepsize(a) == 1.0
end
@testset "Decresaing Stepsize" begin
    p = TestProblem1{MutatingEvaluation}()
    ds = DecreasingStepsize(; length=10.0, factor=1.0, subtrahend=0.0, exponent=1.0)
    @test get_initial_stepsize(ds) == 10.0
    @test ds(p, TestOptions1([1.0]), 1) == 10.0
    @test ds(p, TestOptions1([1.0]), 2) == 5.0
end

@testset "Decorator Options test" begin
    o = TestOptions1([1.0, 2.0])
    r = RecordOptions(o, RecordIteration())
    d = DebugOptions(o, DebugIteration())
    dr = DebugOptions(r, DebugIteration())

    @test has_record(o) == has_record(d)
    @test !has_record(o)
    @test has_record(r)
    @test has_record(r) == has_record(dr)

    @test is_options_decorator(r)
    @test is_options_decorator(dr)
    @test is_options_decorator(d)
    @test !is_options_decorator(o)

    @test dispatch_options_decorator(r) === Val(true)
    @test dispatch_options_decorator(dr) === Val(true)
    @test dispatch_options_decorator(d) === Val(true)
    @test dispatch_options_decorator(o) === Val(false)

    @test get_options(r) == o
    @test get_options(dr) == o
    @test get_options(d) == o
    @test get_options(o) == o
    @test get_options(o, Val(false)) == o

    @test Manopt._extract_val(Val(true))
    @test !Manopt._extract_val(Val(false))

    @test_throws ErrorException get_gradient(o)
    @test_throws ErrorException get_gradient(r)
    @test_throws ErrorException get_iterate(o)
    @test_throws ErrorException get_iterate(r)
    @test_throws ErrorException set_iterate!(o, 0)
    @test_throws ErrorException set_iterate!(r, 0)
end

@testset "FieldReference" begin
    p_mutating = TestProblem1{MutatingEvaluation}()
    p_allocating = TestProblem1{AllocatingEvaluation}()
    X = [10.0, 12.0]
    o = TestOptions1([1.0, 2.0])
    Teval = MutatingEvaluation
    fa = Manopt.@access_field o.storage
    @test fa === o.storage

    Teval = AllocatingEvaluation
    fa = Manopt.@access_field o.storage
    @test fa isa Manopt.FieldReference
    @test fa[] === o.storage
    fa[] = X
    @test fa[] === X
end
