# # [Illustration how to use mutating gradient functions](@id Mutations)
#
# When it comes to time critital operations, a main ingredient in Julia are
# mutating functions, i.e. those that compute in place without additional Memory
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
x[2] = 1.0
data = [exp(M, x, random_tangent(M, x, Val(:Gaussian), σ)) for i in 1:n];
#
# ## Classical definition
#
# The variant from the previous tutorial defines a cost ``F(x)`` and its gradient ``gradF(x)``
F(M, x) = sum(1 / (2 * n) * distance.(Ref(M), Ref(x), data) .^ 2)
gradF(M, x) = sum(1 / n * grad_distance.(Ref(M), data, Ref(x)))
nothing #hide
#
# we further set the stopping criterion to be a little more strict, then we obtain
#
sc = StopWhenGradientNormLess(1e-10)
x0 = random_point(M)
m1 = gradient_descent(M, F, gradF, x0; stopping_criterion=sc)

@btime gradient_descent($M, $F, $gradF, $x0; stopping_criterion=$sc)
nothing #hide
#
# ## Inplace computation of the gradient
#
# We can reduce the memory allocations, by implementing the gradient as a [functor](https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects).
# The motivation is twofold: On the one hand, we want to avoid variables from global scope,
# for example the manifold `M` or the `data`, to be used within the function
# For more complicated cost functions it might also be worth considering to do the same.
#
# Here we store the data (as reference) and one temporary memory in order to avoid
# reallocation of memory per `grad_distance` computation. We have
struct grad!{TD,TTMP}
    data::TD
    tmp::TTMP
end
function (gradf!::grad!)(M, X, x)
    fill!(X, 0)
    for di in gradf!.data
        grad_distance!(M, gradf!.tmp, di, x)
        X .+= gradf!.tmp
    end
    X ./= length(gradf!.data)
    return X
end
#
# Then we just have to initialize the gradient and perform our final benchmark.
# Note that we also have to interpolate all variables passed to the benchmark with a `$`.
#
gradF2! = grad!(data, similar(data[1]))
m2 = deepcopy(x0)
gradient_descent!(M, F, gradF2!, m2; evaluation=MutatingEvaluation(), stopping_criterion=sc)

@btime gradient_descent!(
    $M, $F, $gradF2!, m2; evaluation=$(MutatingEvaluation()), stopping_criterion=$sc
) setup = (m2 = deepcopy($x0))
nothing #hide
#
# Mote that the results `m1`and `m2` are nearly the same.
#
@test distance(M, m1, m2) ≈ 0 atol = 10^-7
