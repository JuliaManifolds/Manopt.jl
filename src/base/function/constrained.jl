"""
    AbstractConstrainedFunction{T}

A common supertype for functors that model constraint functions.

This supertype provides access for the fields ``λ`` and ``μ``, the dual variables of
constraints of type `T`.
"""
abstract type AbstractConstrainedFunction{T} <: AbstractManifoldFunction end

function set_parameter!(acf::AbstractConstrainedFunction{T}, ::Val{:μ}, μ::T) where {T}
    acf.μ = μ
    return acf
end
get_parameter(acf::AbstractConstrainedFunction, ::Val{:μ}) = acf.μ
function set_parameter!(acf::AbstractConstrainedFunction{T}, ::Val{:λ}, λ::T) where {T}
    acf.λ = λ
    return acf
end
get_parameter(acf::AbstractConstrainedFunction, ::Val{:λ}) = acf.λ

"""
    AbstractConstrainedSlackFunction{T,R}

A common supertype for functors that model constraint functions with slack.

This supertype additionally provides access for the fields
* `μ::T` the dual for the inequality constraints
* `s::T` the slack parameter, and
* `β::R` the  the barrier parameter
which is also of type `T`.
"""
abstract type AbstractConstrainedSlackFunction{T, R} end

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
function set_parameter!(
        acsf::AbstractConstrainedSlackFunction{T, R}, ::Val{:β}, β::R
    ) where {T, R}
    acsf.β = β
    return acsf
end
get_parameter(acsf::AbstractConstrainedSlackFunction, ::Val{:β}) = acsf.β
