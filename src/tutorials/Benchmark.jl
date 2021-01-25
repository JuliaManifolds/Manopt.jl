# # [Illustration how to use mutating gradient functions](@id Mutations)
#
# When it comes to time critital operations, a main ingredient in Julia are
# mutating functions, i.e. those that compute inplace without additional Memory
# allocations. In the following the illustrate how to do this with `Manopt.jl`.
#
# Let's start with the same function as in [Get Started: Optimize!](@ref Optimize)
# and compute the mean of some points. Just that here we use the sphere ``\mathbb S^{30}``
# and `n=800` points.
#
# From the just mentioned example, the implementation looks like
using Manopt, Manifolds, Random, BenchmarkTools, Test
Random.seed!(42)
m = 30
M = Sphere(m)
n = 800
σ = π / 8
x = zeros(Float64, m + 1)
x[end] = 1.0
data = [exp(M, x, random_tangent(M, x, Val(:Gaussian), σ)) for i in 1:n];
#
# ## Classical definition
#
# The variant from the previous tutorial defines a cost ``F(x)`` and its gradient ``∇F(x)``
F(x) = sum(1 / (2 * n) * distance.(Ref(M), Ref(x), data) .^ 2)
∇F(x) = sum(1 / n * ∇distance.(Ref(M), data, Ref(x)))
nothing #hide
#
# we further set the stopping criterion to be a little more strict, then we obtain
#
sc = StopWhenGradientNormLess(1e-10)
x0 = random_point(M)
m1 = gradient_descent(M, F, ∇F, x0; stopping_criterion=sc)
@btime gradient_descent(M, F, ∇F, x0; stopping_criterion=sc)
nothing #hide
#
# ## Easy to use inplace variant
#
# A first optimization that – in this example – will just take less than  a 1/20th in time
# and less then 1/16th of memory allocations, is the idea to perform the gradient
# computation inplace. We define `∇F!(X,x)` thatz computes the gradient in place of `X`.
#
# We further use `gradient_descent!` which works in place of its initial value, here `m2`.
#
function ∇F!(X, y)
    return X .= sum(1 / n * ∇distance.(Ref(M), data, Ref(y)))
end
m2 = deepcopy(x0)
@btime gradient_descent!(
    M, F, ∇F!, m2; evaluation=MutatingEvaluation(), stopping_criterion=sc
)
nothing #hide
#
# ## A more involved and more efficient variant
#
# We can reduce the memory allocations even more, when we avoid using global variables
# encapsulated `∇F!`. We can do this by implementing the gradient as a [functor](https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects).
# The motivation is twofold: On the one hand, we want to avoid variables from global scope,
# for example the manifold `M` or the `data`, to be used within the function
# For more complicated cost functions it might also be worth considering to do the same.
#
# Here we store the data (as reference) and one temporary memory in order to avoid
# reallication of memory per `∇distance` computation. We have
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
#
# Then we just have to initialize the gradient and perform our final benchmark.
# Note that we also have to interpolate all variables passed to the benchmark with a `$`.
#
∇F2! = grad!(M, data, similar(data[1]))
m3 = deepcopy(x0)
@btime gradient_descent!(
    $M, $F, $∇F2!, $m3; evaluation=$(MutatingEvaluation()), stopping_criterion=$sc
)
nothing #hide
#
# Mote that all 3 results `m1, m2, m3` are of course the same.
#
@test distance(M, m1, m2) ≈ 0
@test distance(M, m1, m3) ≈ 0
