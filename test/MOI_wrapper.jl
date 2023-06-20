using Test
using LinearAlgebra
using Manifolds
using Manopt
using JuMP

function test_sphere()
    model = Model(Manopt.Optimizer)
    start = normalize(1:3)
    @variable(model, x[i=1:3] in Sphere(2), start = start[i])

    @objective(model, Min, sum(x))
    optimize!(model)
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test primal_status(model) == MOI.FEASIBLE_POINT
    @test primal_status(model) == MOI.FEASIBLE_POINT
    @test dual_status(model) == MOI.NO_SOLUTION
    @test objective_value(model) ≈ -√3
    @test value.(x) ≈ -inv(√3) * ones(3) rtol = 1e-2
    @test raw_status(model) isa String
    @test raw_status(model)[end] != '\n'

    @objective(model, Max, sum(x))
    set_start_value.(x, start)
    optimize!(model)
    @test objective_value(model) ≈ -√3
    @test value.(x) ≈ inv(√3) * ones(3) rtol = 1e-2
    @test raw_status(model) isa String
    @test raw_status(model)[end] != '\n'

    # Creating a model directly with `@NLobjective` wouldn't work
    # because of https://github.com/jump-dev/MathOptInterface.jl/blob/32dbf6056b0b5fb9d44dc583ecc8249f6fd703ea/src/Utilities/copy.jl#L485-L500
    # so we need to wait for https://github.com/jump-dev/JuMP.jl/pull/3106 for NL objectives to work with `@objective`
    # Here, we are by-passing that because we are modifying the objective
    # so we don't call `copy_to`.
    @NLobjective(model, Min, sum(xi^4 for xi in x))
    set_start_value.(x, start)
    optimize!(model)
    @test objective_value(model) ≈ 1 / 3 rtol = 1e-4
    @test value.(x) ≈ inv(√3) * ones(3) rtol = 1e-2
    @test raw_status(model) isa String
    @test raw_status(model)[end] != '\n'
end

function test_rank()
    v = [
        1 -1
        -1 1
        1 1
    ]
    A = v * v'
    model = Model(Manopt.Optimizer)
    set_attribute(model, "stepsize", ConstantStepsize(1))
    @variable(model, U[1:3, 1:2] in FixedRankMatrices(3, 2, 2), start = 1.0)

    # We don't do the sum of the squares of the entry on purpose
    # to tests quadratic objective (and not quartic)
    @objective(model, Min, sum(A - U * U'))
    optimize!(model)
    @show objective_value(model)
    return nothing
end

@testset "JuMP tests" begin
    test_sphere()
    test_rank()
end

function test_runtests()
    optimizer = Manopt.Optimizer()
    config = MOI.Test.Config(; exclude=Any[MOI.ListOfModelAttributesSet])
    return MOI.Test.runtests(
        optimizer,
        config;
        exclude=String[
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
