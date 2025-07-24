using Test
using LinearAlgebra
using Manifolds
using Manopt
using JuMP

function _test_sphere_sum(model, obj_sign)
    @test MOI.get(unsafe_backend(model), MOI.ResultCount()) == 0
    optimize!(model)
    @test MOI.get(unsafe_backend(model), MOI.NumberOfVariables()) == 3
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test primal_status(model) == MOI.FEASIBLE_POINT
    @test primal_status(model) == MOI.FEASIBLE_POINT
    @test dual_status(model) == MOI.NO_SOLUTION
    @test objective_value(model) ≈ obj_sign * √3
    @test value.(model[:x]) ≈ obj_sign * inv(√3) * ones(3) rtol = 1.0e-2
    @test raw_status(model) isa String
    return @test raw_status(model)[end] != '\n'
end

function test_sphere()
    model = Model(Manopt.JuMP_Optimizer)
    start = normalize(1:3)
    @variable(model, x[i = 1:3] in Sphere(2), start = start[i])

    function eval_sum_cb(M, x)
        return sum(x)
    end
    function eval_grad_sum_cb(M, X)
        grad_f = ones(length(X))
        return Manopt.riemannian_gradient(M, X, grad_f)
    end

    objective = Manopt.ManifoldGradientObjective(eval_sum_cb, eval_grad_sum_cb)

    @testset "$obj_sense" for (obj_sense, obj_sign) in
        [(MOI.MIN_SENSE, -1), (MOI.MAX_SENSE, 1)]
        @testset "JuMP objective" begin
            @objective(model, obj_sense, sum(x))
            _test_sphere_sum(model, obj_sign)
        end

        @testset "Manopt objective" begin
            @objective(model, obj_sense, objective)
            _test_sphere_sum(model, obj_sign)
        end
    end
    @test contains(
        sprint(show, model),
        "Vector{VariableRef} in ManoptJuMPExt.VectorizedManifold{Sphere{ManifoldsBase.TypeParameter{Tuple{2}}, ℝ}}: 1",
    )
    @test contains(sprint(print, model), "[x[1], x[2], x[3]] in Sphere(2, ℝ)")
    @test contains(
        JuMP.model_string(MIME("text/latex"), model),
        "[x_{1}, x_{2}, x_{3}] \\in Sphere(2, ℝ)",
    )

    @objective(model, Min, sum(xi^4 for xi in x))
    set_start_value.(x, start)
    optimize!(model)
    @test objective_value(model) ≈ 1 / 3 rtol = 1.0e-4
    @test value.(x) ≈ inv(√3) * ones(3) rtol = 1.0e-2
    @test raw_status(model) isa String
    @test raw_status(model)[end] != '\n'

    set_objective_sense(model, MOI.FEASIBILITY_SENSE)
    optimize!(model)
    @test sum(value.(x) .^ 2) ≈ 1

    set_start_value(x[3], nothing)
    err = ErrorException("No starting value specified for `3`th variable.")
    @test_throws err optimize!(model)
    set_start_value(x[3], 1.0)

    @variable(model, [1:2, 1:2] in Stiefel(2, 2))
    return @test_throws MOI.AddConstraintNotAllowed optimize!(model)
end

function _test_stiefel(solver)
    A = [
        1 -1
        -1 1
    ]
    # Use `add_bridges = false` to test `copy_to`
    model = Model(Manopt.JuMP_Optimizer; add_bridges = false)
    dst = "descent_state_type"
    @test MOI.supports(unsafe_backend(model), MOI.RawOptimizerAttribute(dst))
    set_attribute(model, dst, solver)
    @test get_attribute(model, dst) == solver
    @variable(model, U[1:2, 1:2] in Stiefel(2, 2), start = 1.0)

    @objective(model, Min, sum((A - U) .^ 2))
    optimize!(model)
    @test objective_value(model) ≈ 2
    @test value.(U) ≈ [1 0; 0 1]
    return nothing
end

function test_stiefel()
    return _test_stiefel(Manopt.GradientDescentState)
end

@testset "JuMP tests" begin
    test_sphere()
    test_stiefel()
end

function test_runtests()
    optimizer = Manopt.JuMP_Optimizer()
    config = MOI.Test.Config(; exclude = Any[MOI.ListOfModelAttributesSet])
    return MOI.Test.runtests(
        optimizer,
        config;
        exclude = String[
            # See https://github.com/jump-dev/MathOptInterface.jl/pull/2195
            "test_model_copy_to_UnsupportedConstraint",
            "test_model_copy_to_UnsupportedAttribute",
            "test_model_ScalarFunctionConstantNotZero",
            # See https://github.com/jump-dev/MathOptInterface.jl/pull/2196/
            "test_objective_ScalarQuadraticFunction_in_ListOfModelAttributesSet",
            "test_objective_ScalarAffineFunction_in_ListOfModelAttributesSet",
            "test_objective_VariableIndex_in_ListOfModelAttributesSet",
            "test_objective_set_via_modify",
            "test_objective_ObjectiveSense_in_ListOfModelAttributesSet",
        ],
    )
end

@testset "MOI tests" begin
    test_runtests()
end
