# # [Illustration how to use mutating gradient functions](@id Mutations)
#
#
#
using Manopt, Manifolds, Random, BenchmarkTools, Test
Random.seed!(42)
m = 30
M = Sphere(m)
n = 800
σ = π / 8
x = zeros(Float64, m + 1)
x[end] = 1.0
data = [exp(M, x, random_tangent(M, x, Val(:Gaussian), σ)) for i in 1:n]
#
# Classical definition
#
F(y) = sum(1 / (2 * n) * distance.(Ref(M), Ref(y), data) .^ 2)
∇F(y) = sum(1 / n * ∇distance.(Ref(M), data, Ref(y)))

sc = StopWhenGradientNormLess(10.0^-10)
x0 = random_point(M)
m1 = gradient_descent(M, F, ∇F, x0; stopping_criterion=sc)
@btime gradient_descent(M, F, ∇F, x0; stopping_criterion=sc)

#
# Easy to use inplace variant
#
function ∇F!(X, y)
    return X .= sum(1 / n * ∇distance.(Ref(M), data, Ref(y)))
end
m2 = deepcopy(x0)
@btime gradient_descent!(
    M, F, ∇F!, m2; evaluation=MutatingEvaluation(), stopping_criterion=sc
)

#
# Memory and time efficient variant
#
struct cost{TM,TD}
    M::TM
    data::TD
end
(f::cost)(y) = sum(1 / (2 * n) * distance.(Ref(f.M), Ref(y), f.data) .^ 2)
struct grad!{TM,TD,TTMP}
    M::TM
    data::TD
    tmp::TTMP
end
function (∇f!::grad!)(X, y)
    fill!(X, 0)
    for di in ∇f!.data
        ∇distance!(∇f!.M, ∇f!.tmp, di, y)
        X .+= ∇f!.tmp
    end
    X ./= n
    return X
end

F2 = cost(M, data)
∇F2! = grad!(M, data, similar(data[1]))

m3 = deepcopy(x0)
@btime gradient_descent!(
    $M, $F2, $∇F2!, $m3; evaluation=$(MutatingEvaluation()), stopping_criterion=sc
)

#
# All yield the same result
#
@test distance(M, m1, m2) ≈ 0
@test distance(M, m1, m3) ≈ 0
