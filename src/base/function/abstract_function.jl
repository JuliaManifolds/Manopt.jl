"""
    AbstractManifoldFunction <: Function

An abstract function type to wrap functions defined on Riemannian manifolds.
Within `Manopt.jl`, most functions are assumed to work in-place, like gradient and Hessian functions.
Exceptions are those functions that return single values, e.g.
a cost function, the differential of a function mapping into the real or complex numbers
or a boolean function like a [`StoppingCriterion`](@ref).
"""
abstract type AbstractManifoldFunction <: Function end

"""
    AbstractDecoratedManifoldFunction{F} <: AbstractManifoldFunction

An abstract function type to indicate that another function is wrapped.
Often, the wrapped function of type `F` is itself an [`AbstractManifoldFunction`](@ref).
"""
abstract type AbstractDecoratedManifoldFunction{F} <: AbstractManifoldFunction end

"""
    set_parameter!(amf::AbstractManifoldFunction, element::Symbol, args...)

Set a certain `element` from the [`AbstractManifoldFunction`](@ref) `amf` to a value specified
by `args...`.
This function by default dispatches on the same function replacing the second argument by `Val(element)`.
"""
set_parameter!(amo::AbstractManifoldFunction, e::Symbol, args...)


@doc """
    AbstractApproximateHessianFunction <: AbstractManifoldFunction

An abstract supertype for approximate Hessian functions, declares them also to be functions.
"""
abstract type AbstractApproximateHessianFunction <: AbstractManifoldFunction end

_doc_ApproxHessian_formula = """
```math
$(_tex(:Hess))f(p)[X] ≈
$(_tex(:frac, _tex(:norm, "X"), "c"))$(_tex(:Bigl))(
  $(_math(:VectorTransport, "q", "p"))$(_tex(:bigl))($(_tex(:grad))f(q)$(_tex(:bigr)))
  - $(_tex(:grad))f(p)
$(_tex(:Bigr)))
```
"""
_doc_ApproxHessian_step = raw"\operatorname{retr}_p(\frac{c}{\lVert X \rVert_p}X)"
