using Manifolds
using Manopt
using JuMP

model = Model(Manopt.Optimizer)
@variable(model, x[1:3] in Sphere(2), start = 1/√2)

@objective(model, Min, x[1] + x[2])
optimize!(model)
println(solution_summary(model))
println(value.(x))

@NLobjective(model, Min, x[1]^4 + x[2]^4)
optimize!(model)
println(solution_summary(model))
println(value.(x))
