"""
    AbstractManoptFunction{E<:AbstractEvaluationType} <: Function

An abstract type to represent costs, gradients, and similarly in Manopt.jl
"""
abstract type AbstractManoptFunction{E<:AbstractEvaluationType} <: Function end

@doc raw"""
    GradientFunction{E,F} <: AbstractManoptFunction{E}

Represent the gradient ``\operatorname{grad} f: \mathcal M → T\mathcal M``,
``\operatorname{grad} f(p) ∈ T_p\mathcal M``.

The resulting type is a functor and can both be called as allocating or in-place variant.

# Constructor

    GradientFunction(grad_f; evaluation=AllocatingEvaluation)

Generate the `Gradientfunction` wrapper for `grad_f` with `evaluation` type.

TODO can we even “guess” the evaluation type by checking number of parameters of `grad_f`?
"""
struct GradientFunction{E<:AbstractEvaluationType,F}
    f::F
end
function GradientFunction(
    f::F; evaluation::E=AllocatingEvaluation()
) where {F,E<:AbstractEvaluationType}
    return GradientFunction{E,F}(f)
end
function (f::GradientFunction{AllocatingEvaluation})(M, p)
    return f.f(M, p)
end
function (f::GradientFunction{AllocatingEvaluation})(M, X, p)
    return copyto!(M, X, f.f(M, p))
end
function (f::GradientFunction{InplaceEvaluation})(M, p)
    X = zero_vector(M, p)
    return f.f(M, X, p)
end
function (f::GradientFunction{InplaceEvaluation})(M, X, p)
    return f.f(M, X, p)
end
