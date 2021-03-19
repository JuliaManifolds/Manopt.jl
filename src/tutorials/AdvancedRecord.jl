# # [Advanced Recording Example](@id RecordingTutorial)
#
# The recording and debug possiblities make it possible to record nearly any data during the iterations.
# For fields of the [`Options`](@ref) this can be directly done using the [`RecordEntry`](@ref).
# For others, an own [`RecordAction`](@ref) has to be defined.
# This small tutorial illustrates, how to track cost function evaluations.
#
# We first define the cost function itself as a functor, that counts its own evaluations.
# We stick to the example from the introductionary [Get Started: Optimize!](@ref Optimize) and compute the Riemannian center of mass.
# 
using Manopt, Manifolds, Random

mutable struct MyCost{T}
    data::T
    count::Int
end
MyCost(data::T) where {T} = MyCost{T}(data, 0)
function (c::MyCost)(M, x)
    c.count += 1
    return sum(1 / (2 * length(c.data)) * distance.(Ref(M), Ref(x), c.data) .^ 2)
end
nothing #hide
#
# and we define the following RecordAction, 
#
mutable struct RecordCount <: RecordAction
    recorded_values::Vector{Int}
    RecordCount() = new(Vector{Int}())
end
function (r::RecordCount)(p::Problem, ::Options, i)
    if i > 0
        push!(r.recorded_values, p.cost.count)
    elseif i < 0 && i > typemin(Int) # reset if negative but smallest int (stop)
        r.recorded_values = Vector{Int}()
    end
end
nothing #hide
#
# The remainder we just set up as before.
#
Random.seed!(42)
m = 30
M = Sphere(m)
n = 800
σ = π / 8
x = zeros(Float64, m + 1)
x[2] = 1.0
data = [exp(M, x, random_tangent(M, x, Val(:Gaussian), σ)) for i in 1:n];
gradF(M, x) = sum(1 / n * grad_distance.(Ref(M), data, Ref(x)))
nothing #hide
#
# and now we can initialize the new cost and call the gradient descent.
# Note that in order to access the record we have to change the return value of the gradient descent
#
F = MyCost(data)
R = gradient_descent(M, F, gradF, data[1], record=[:Iteration, RecordCount(), :Cost], return_options=true)
get_record(R)