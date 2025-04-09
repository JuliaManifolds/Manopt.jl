"""
    struct RiemannianFunction{E<:AbstractEvaluationType, O<:Manopt.AbstractManifoldObjective{E} <: MOI.AbstractScalarFunction
        objective::O
    end

Use a [`AbstractManifoldObjective`](@ref) from [`Manopt.jl`](@ref) within [`JuMP`](@ref).
This wraps both the signature and evaluation mode transparently.

!!! note
    For this black box function, the input variable `x` is not in a vectorized format but in the format of points in the corresponding manifold.
"""
struct RiemannianFunction{
    E<:AbstractEvaluationType,O<:Manopt.AbstractManifoldObjective{E}
} <: MOI.AbstractScalarFunction
    objective::O
end