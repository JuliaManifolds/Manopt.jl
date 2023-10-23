using Manopt, Manifolds, Random, QuadraticModels, RipQP, BenchmarkTools, Test

M = SymmetricPositiveDefinite(3)
f(M, y) = distance(M, y, x)
x = rand(M)
x0 = rand(M)
function ∂f(M, y)
    if distance(M, x, y) == 0
        return zero_vector(M, y)
    end
    return -2 * log(M, y, x) / max(10 * eps(Float64), distance(M, x, y))
end
x1 = convex_bundle_method(M, f, ∂f, x0)
@btime convex_bundle_method($M, $f, $∂f, $x0)

function ∂f!(M, X, y)
    d = distance(M, x, y)
    if d == 0
        return zero_vector!(M, X, y)
    end
    log!(M, X, y, x)
    X .*= -2 / max(10 * eps(Float64), d)
    return X
end
x2 = copy(x0)
convex_bundle_method!(M, f, ∂f!, x2; evaluation=InplaceEvaluation())
@btime convex_bundle_method!($M, $f, $∂f!, x3; evaluation=$(InplaceEvaluation())) setup = (
    x3 = deepcopy($x0)
)

@test x1 == x2
