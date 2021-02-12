using Manopt, ManifoldsBase, Manifolds, LinearAlgebra, BenchmarkTools, Test

M = Euclidean(2)
x = [4.0, 2.0]
x0 = [5.0, 2.0]
f(M, y) = distance(M, y, x)
function ∂f(M, y)
    if distance(M, x, y) == 0
        return zero_tangent_vector(M, y)
    end
    return -2 * log(M, y, x) / max(10 * eps(Float64), distance(M, x, y))
end
x1 = subgradient_method(M, f, ∂f, x0)
@btime subgradient_method($M, $f, $∂f, $x0)

function ∂f!(M, X, y)
    d = distance(M, x, y)
    if d == 0
        return zero_tangent_vector!(M, X, y)
    end
    log!(M, X, y, x)
    X .*= -2 / max(10 * eps(Float64), d)
    return X
end
x2 = copy(x0)
subgradient_method!(M, f, ∂f!, x2; evaluation=MutatingEvaluation())
@btime subgradient_method!($M, $f, $∂f!, x3; evaluation=$(MutatingEvaluation()))  setup = (x3 = deepcopy($x0))

@test x1==x2