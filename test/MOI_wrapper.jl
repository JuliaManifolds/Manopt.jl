using Test
using LinearAlgebra
using Manifolds
using Manopt
using JuMP

function _test_allocs(problem::Manopt.AbstractManoptProblem, x, g)
    Manopt.get_cost(problem, x) # Compilation
    @test 0 == @allocated Manopt.get_cost(problem, x)
    Manopt.get_gradient!(problem, g, x) # Compilation
    @test 0 == @allocated Manopt.get_gradient!(problem, g, x)
    return nothing
end

_test_allocs(optimizer, x, g) = _test_allocs(optimizer.problem, x, g)

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
    @test raw_status(model)[end] != '\n'
    _test_allocs(unsafe_backend(model), zeros(3), zeros(3))
    return nothing
end

function test_sphere()
    model = Model(Manopt.JuMP_Optimizer)
    start = normalize(1:3)
    @variable(model, x[i = 1:3] in Sphere(2), start = start[i])

    objective = let
        # We create `grad_f` here to avoid having an allocation in `eval_grad_sum_cb`
        # so that we can easily test that the rest is allocation-free by testing that
        # `@allocated` returns 0 the whole call to `get_gradient!`.
        # To avoid creating a closure capturing the `grad_f` object,
        # we use the `let` block trick detailed in:
        # https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-captured
        grad_f = ones(3)

        function eval_sum_cb(M, x)
            return sum(x)
        end

        function eval_grad_sum_cb(M, g, X)
            return Manopt.riemannian_gradient!(M, g, X, grad_f)
        end

        Manopt.ManifoldGradientObjective(
            eval_sum_cb, eval_grad_sum_cb; evaluation = Manopt.InplaceEvaluation()
        )
    end

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
        "Vector{VariableRef} in ManoptJuMPExt.ManifoldSet{Sphere{ManifoldsBase.TypeParameter{Tuple{2}}, ℝ}}: 1",
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

@testset "JuMP tests" begin
    test_sphere()
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
