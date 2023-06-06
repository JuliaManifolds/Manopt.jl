using Test
using LinearAlgebra
using Manifolds
using Manopt
using JuMP

model = Model(Manopt.Optimizer)
start = normalize(1:3)
@variable(model, x[i=1:3] in Sphere(2), start = start[i])

@objective(model, Min, sum(x))
optimize!(model)
@test termination_status(model) == MOI.LOCALLY_SOLVED
@test primal_status(model) == MOI.FEASIBLE_POINT
@test primal_status(model) == MOI.FEASIBLE_POINT
@test dual_status(model) == MOI.NO_SOLUTION
@test value.(x) ≈ -inv(√3) * ones(3) rtol=1e-2

@objective(model, Max, sum(x))
set_start_value.(x, start)
optimize!(model)
@test value.(x) ≈ inv(√3) * ones(3) rtol=1e-2

# Creating a model directly with `@NLobjective` wouldn't work
# because of https://github.com/jump-dev/MathOptInterface.jl/blob/32dbf6056b0b5fb9d44dc583ecc8249f6fd703ea/src/Utilities/copy.jl#L485-L500
# so we need to wait for https://github.com/jump-dev/JuMP.jl/pull/3106 for NL objectives to work with `@objective`
# Here, we are by-passing that because we are modifying the objective
# so we don't call `copy_to`.
@NLobjective(model, Min, sum(xi^4 for xi in x))
set_start_value.(x, start)
optimize!(model)
@test value.(x) ≈ inv(√3) * ones(3) rtol=1e-2
