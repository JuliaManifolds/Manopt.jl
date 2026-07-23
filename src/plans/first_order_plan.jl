#
# Records
#
@doc """
    RecordGradient <: RecordAction

record the gradient evaluated at the current iterate

# Constructors
    RecordGradient(ξ)

initialize the [`RecordAction`](@ref) to the corresponding type of the tangent vector.
"""
mutable struct RecordGradient{T} <: RecordAction
    recorded_values::Array{T, 1}
    RecordGradient{T}() where {T} = new(Array{T, 1}())
end
RecordGradient(::T) where {T} = RecordGradient{T}()
function (r::RecordGradient{T})(
        ::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int
    ) where {T}
    return record_or_reset!(r, get_gradient(s), k)
end
show(io::IO, ::RecordGradient{T}) where {T} = print(io, "RecordGradient($T)")
function status_summary(rg::RecordGradient; context::Symbol = :default)
    (context === :short) && return ":Gradient"
    return "A RecordAction to record the current gradient"
end
@doc """
    RecordGradientNorm{R<:Real} <: RecordAction

record the norm of the current gradient

## Constructor
    RecordGradientNorm(r::Type{<:Real}=Float64)
"""
mutable struct RecordGradientNorm{R <: Real} <: RecordAction
    recorded_values::Array{R, 1}
    RecordGradientNorm(r::Type{<:Real} = Float64) = new{r}(Array{r, 1}())
end
function (r::RecordGradientNorm)(
        mp::AbstractManoptProblem, ast::AbstractManoptSolverState, k::Int
    )
    M = get_manifold(mp)
    return record_or_reset!(r, norm(M, get_iterate(ast), get_gradient(ast)), k)
end
show(io::IO, ::RecordGradientNorm) = print(io, "RecordGradientNorm()")
function status_summary(rg::RecordGradientNorm; context::Symbol = :default)
    (context === :short) && return ":GradientNorm"
    return "A RecordAction to record the current gradient norm"
end

@doc """
    RecordStepsize <: RecordAction

record the step size.

## Constructor
    RecordStepsise(r::Type{<:Real}=Float64)
"""
mutable struct RecordStepsize{R <: Real} <: RecordAction
    recorded_values::Array{R, 1}
    RecordStepsize(r::Type{<:Real} = Float64) = new{r}(Array{r, 1}())
end
function (r::RecordStepsize)(p::AbstractManoptProblem, s::AbstractGradientSolverState, k)
    return record_or_reset!(r, get_last_stepsize(p, s, k), k)
end
show(io::IO, ::RecordStepsize{R}) where {R} = print(io, "RecordStepsize($R)")
function status_summary(rg::RecordStepsize{R}; context::Symbol = :default) where {R}
    (context === :short) && return ":Stepsize"
    return "A RecordAction to record the current stepsize (of type $R)"
end
