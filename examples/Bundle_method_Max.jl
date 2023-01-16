using Manopt, Manifolds, Random, QuadraticModels, RipQP

l = Int(1e4)
M = Stiefel(3,2)
#Random.seed!(42)
data = [rand(M; σ=0.4) for i in 1:l]

F(M, y) = sum(1 / (2 * length(data)) * distance.(Ref(M), data, Ref(y)) .^ 2)
gradF(M, y) = sum(1 / length(data) * grad_distance.(Ref(M), data, Ref(y)))
F2(M, y) = sum(1 / (2 * length(data)) * distance.(Ref(M), Ref(y), data))
gradF2(M, y) = sum(1 / (2 * length(data)) * grad_distance.(Ref(M), data, Ref(y), 1))

F3(M, y) = max(F(M, y), F2(M, y))
function gradF3(M, y)
    if F3(M, y) == F(M, y) && F3(M, y) != F2(M, y)
        return gradF(M, y)
    elseif F3(M, y) == F2(M, y) && F3(M, y) != F(M, y)
        return gradF2(M, y)
    else
        r = rand()
        return r * gradF(M, y) + (1 - r) * gradF2(M, y)
    end
end

G(M, y) = (F(M, y) - F2(M, y))^2
m = particle_swarm(M, G)

bundle_min = bundle_method(
    M,
    F3,
    gradF3,
    m;
    # stopping_criterion=StopAfterIteration(100),
    # debug=[:Iteration, :Cost, "\n"],
)

subgradient_min = subgradient_method(
    M,
    F3,
    gradF3,
    m;
    # stopping_criterion=StopAfterIteration(100),
    # debug=[:Iteration, :Cost, "\n"],
)

println("Distance between minima: $(distance(M, bundle_min, subgradient_min))")

#println("$(F3(M, maxfunc_optimum) == F2(M, maxfunc_optimum))")

# for j in 1:10
#     print("$j")
#     println("    $(F3(M,data[j]) == F(M,data[j]) ? "F3(x) = F(x)" : "F3(x) = F2(x)")")
#     println("    $(gradF3(M,data[j]) == gradF(M,data[j]) ? " ∇F3(x) = ∇F(x)" : " ∇F3(x) = ∇F2(x)")")

# println("    F3 = F: $(F3(M,maxfunc_optimum)==F(M,maxfunc_optimum) ? "Yes" : "No") \t F3 = F2: $(F3(M,maxfunc_optimum)==F2(M,maxfunc_optimum) ? "Yes" : "No")")
# println("F3($maxfunc_optimum) = $(F3(M, maxfunc_optimum))")
# end