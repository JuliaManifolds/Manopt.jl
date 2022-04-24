"""
    Problem{T}

Describe the problem that should be optimized by stating all properties, that do not change
during an optimization or that are dependent of a certain solver.

The parameter `T` can be used to distinguish problems with different representations
or implementations.
The default parameter [`AllocatingEvaluation`](@ref), which might be slower but easier to use.
The usually faster parameter value is [`MutatingEvaluation`](@ref)

See [`Options`](@ref) for the changing and solver dependent properties.
"""
abstract type Problem{T} end

"""
    AbstractEvaluationType

An abstract type to specify the kind of evaluation a [`Problem`](@ref) supports.
"""
abstract type AbstractEvaluationType end

"""
    AllocatingEvaluation <: AbstractEvaluationType

A parameter for a [`Problem`](@ref) indicating that the problem uses functions that
allocate memory for their result, i.e. they work out of place.
"""
struct AllocatingEvaluation <: AbstractEvaluationType end

"""
    MutatingEvaluation

A parameter for a [`Problem`](@ref) indicating that the problem uses functions that
do not allocate memory but work on their input, i.e. in place.
"""
struct MutatingEvaluation <: AbstractEvaluationType end

"""
    get_cost(p, x)

Evaluate the cost function stored within a [`Problem`](@ref) at the point `x`.
"""
function get_cost(p::Problem, x)
    return p.cost(p.M, x)
end

"""
    update_cost!(p::Problem, f)

Update the cost function in a [`Problem`](@ref) `p` to a new function `f = (M, x) -> ...`.

By default the cost function is assumed to be stored `p.cost`.
"""
function update_cost!(p::Problem, f)
    p.cost = f
    return p
end
