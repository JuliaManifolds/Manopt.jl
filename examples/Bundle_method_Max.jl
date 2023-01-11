using Manopt, Manifolds, Random, QuadraticModels, RipQP

M = Hyperbolic(4)
#M = SymmetricPositiveDefinite(3)
Random.seed!(42)
data = [rand(M; σ=0.4) for i in 1:100]

F(M, y) = sum(1 / (2 * length(data)) * distance.(Ref(M), data, Ref(y)) .^ 2)
gradF(M, y) = sum(1 / length(data) * grad_distance.(Ref(M), data, Ref(y)))
F2(M, y) = sum(1 / (2 * length(data)) * distance.(Ref(M), Ref(y), data))
gradF2(M, y) = sum(1 / (2 * length(data)) * grad_distance.(Ref(M), data, Ref(y), 1))

F3(M, y) = max(F(M, y), F2(M, y))
function gradF3(M, y)
    if F3(M, y) == F(M, y)
        return gradF(M, y)
    else
        return gradF2(M, y)
    end
end

maxfunc_optimum = bundle_method(
    M, F3, gradF3, data[1]; stopping_criterion=StopAfterIteration(100)
)#, debug = [:Iteration, :Cost, "\n"])
println("$(F3(M, maxfunc_optimum) == F2(M, maxfunc_optimum))")

# for j in 1:10
#     print("$j")
#     println("    $(F3(M,data[j]) == F(M,data[j]) ? "F3(x) = F(x)" : "F3(x) = F2(x)")")
#     println("    $(gradF3(M,data[j]) == gradF(M,data[j]) ? " ∇F3(x) = ∇F(x)" : " ∇F3(x) = ∇F2(x)")")

# println("    F3 = F: $(F3(M,maxfunc_optimum)==F(M,maxfunc_optimum) ? "Yes" : "No") \t F3 = F2: $(F3(M,maxfunc_optimum)==F2(M,maxfunc_optimum) ? "Yes" : "No")")
# println("F3($maxfunc_optimum) = $(F3(M, maxfunc_optimum))")
# end
