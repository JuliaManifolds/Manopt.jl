"""
    AbstractConstrainedFunction{T} <: AbstractManifoldFunction

A common supertype for functors that model constraint functions.

This supertype provides access to the fields `λ` and `μ`, the dual variables of
constraints of type `T`.
"""
abstract type AbstractConstrainedFunction{T} <: AbstractManifoldFunction end

function set_parameter!(acf::AbstractConstrainedFunction, ::Val{:μ}, μ)
    acf.μ = μ
    return acf
end
get_parameter(acf::AbstractConstrainedFunction, ::Val{:μ}) = acf.μ
function set_parameter!(acf::AbstractConstrainedFunction, ::Val{:λ}, λ)
    acf.λ = λ
    return acf
end
get_parameter(acf::AbstractConstrainedFunction, ::Val{:λ}) = acf.λ

"""
    AbstractConstrainedSlackFunction{T,R} <: AbstractManifoldFunction

A common supertype for functors that model constraint functions with slack.

This supertype additionally provides access to the fields

* `μ::T`: the dual for the inequality constraints
* `s::T`: the slack parameter
* `β::R`: the barrier parameter
"""
abstract type AbstractConstrainedSlackFunction{T, R} <: AbstractManifoldFunction end

function set_parameter!(acsf::AbstractConstrainedSlackFunction{T}, ::Val{:s}, s::T) where {T}
    acsf.s = s
    return acsf
end
get_parameter(acsf::AbstractConstrainedSlackFunction, ::Val{:s}) = acsf.s
function set_parameter!(acsf::AbstractConstrainedSlackFunction{T}, ::Val{:μ}, μ::T) where {T}
    acsf.μ = μ
    return acsf
end
get_parameter(acsf::AbstractConstrainedSlackFunction, ::Val{:μ}) = acsf.μ
function set_parameter!(acsf::AbstractConstrainedSlackFunction{T, R}, ::Val{:β}, β::R) where {T, R}
    acsf.β = β
    return acsf
end
get_parameter(acsf::AbstractConstrainedSlackFunction, ::Val{:β}) = acsf.β
