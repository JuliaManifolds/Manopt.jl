# # [Advanced Recording Example](@id RecordingTutorial)
#
# The recording and debug possiblities make it possible to record nearly any data during the iterations.
# This tutorial illustrates how to
# * record one value during the iterations
# * record multiple values during the iterations and access them afterwards
# * define an own [`RecordAction`](@ref) to perform individual recordings.
#
# Several predefined recordings exist, for example [`RecordCost`](@ref) or [`RecordGradient`](@ref), depending on the solver used.
# For fields of the [`Options`](@ref) this can be directly done using the [`RecordEntry`](@ref).
# For others, an own [`RecordAction`](@ref) can be defined.
#
# We illustrate these using the gradient descent used in the introductionary [Get Started: Optimize!](@ref Optimize) tutorial example of computing the Riemannian Center of mass and refer to that tutorial for the mathematical details.
using Manopt, Manifolds, Random
Random.seed!(42)
m = 30
M = Sphere(m)
n = 800
σ = π / 8
x = zeros(Float64, m + 1)
x[2] = 1.0
data = [exp(M, x, random_tangent(M, x, Val(:Gaussian), σ)) for i in 1:n];
F(M, y) = sum(1 / (2 * n) * distance.(Ref(M), Ref(y), data) .^ 2)
gradF(M, y) = sum(1 / n * grad_distance.(Ref(M), data, Ref(y)))
nothing #hide
# ## Plain examples
#
# For the high level interfaces of the solvers, like [`gradient_descent`](@ref) we have to set `return_options` to `true` to obtain the whole options structure and not only the resulting resulting minimizer.
#
R = gradient_descent(M, F, gradF, data[1]; record=:Cost, return_options=true)
#
# You can attach different recorders to some operations (`:Start`. `:Stop`, `:Iteration` at time of writing), where `:Iteration` is the default, so the following is the same as `get_record(R, :Iteration)`. We get
#
get_record(R)
#
# To record more than one value, you can pass a array of a mix of symbols and [`RecordAction`](@ref) which gets mapped to a [`RecordGroup`](@ref)
R = gradient_descent(M, F, gradF, data[1]; record=[:Iteration, :Cost], return_options=true)
#
# Here, the Symbol `:Cost` is mapped to using the [`RecordCost`](@ref) action. The same holds for `:Iteration` and `:Iterate` and any member field of the current [`Options`](@ref).
# To access these you can first extract the group of records (of the `:Iteration` action) and then access the `:Cost`
ra = get_record_action(R)[:Cost]
#
# Or similarly
#
get_record(R, :Iteration, :Cost)
#
# Note that the first symbol again refers to the point where we record (not to the thing we record).
# We can also pass a Tuple as second argument to have our own order (not that now the second `:Iteration` refers to the recorded iteratons)
get_record(R, :Iteration, (:Cost, :Iteration))
#
# ## A more complex example
#
# To illustrate a complicated example let's record
# * the iteration number, cost and gradient field, but only every sixth iteration
# * the iteration at which we stop
#
# We first generate the problem and the options
#
p = GradientProblem(M, F, gradF)
o = GradientDescentOptions(
    M,
    copy(data[1]);
    stopping_criterion=StopAfterIteration(200) | StopWhenGradientNormLess(10.0^-9),
)
#
# and now decorate these with [`RecordOptions`](@ref)
#
rI = RecordEvery(
    RecordGroup([
        :Iteration => RecordIteration(),
        :Cost => RecordCost(),
        :Gradient => RecordEntry(similar(data[1]), :gradient),
    ]),
    6,
)
sI = RecordIteration()
r = RecordOptions(o, Dict(:Iteration => rI, :Stop => sI))
r2 = solve(p, r)
#
# and we see
#
get_record(r2, :Stop)
#
# as well as
#
get_record(r2, :Iteration, (:Iteration, :Cost))
#
# Here it is interesting to see, that a meta-record like [`RecordEvery`](@ref) just passes the tuple further on, so we can again also do
#
get_record_action(r2, :Iteration)[:Gradient]
#
# ## Writing an own [`RecordAction`](@ref)s
#
# Let's investigate where we want to count the number of function evaluations, again just to illustrate, since for the gradient this is just one evaluation per iteration.
# We first define a cost, that counts it's own calls.
#
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
    elseif i < 0 # reset if negative
        r.recorded_values = Vector{Int}()
    end
end
nothing #hide
#
# And now we can initialize the new cost and call the gradient descent.
# Note that this illustrates also the last use case – you can pass symbol-Action pairs into the `record=`array.
#
F2 = MyCost(data)
R = gradient_descent(
    M,
    F2,
    gradF,
    data[1];
    record=[:Iteration, :Count => RecordCount(), :Cost],
    return_options=true,
)
get_record(R)
