@doc """
    AbstractApproximateHessianFunction <: Function

An abstract supertype for approximate Hessian functions, declares them also to be functions.
"""
abstract type AbstractApproximateHessianFunction <: Function end

_doc_ApproxHessian_formula = """
```math
$(_tex(:Hess))f(p)[X] ≈
$(_tex(:frac, "$(_tex(:norm, "X"))", "c"))$(_tex(:Bigl))(
  $(_math(:VectorTransport, "p", "q"))$(_tex(:bigl))( $(_tex(:grad))f(q)$(_tex(:bigr)) - $(_tex(:grad))f(p)
$(_tex(:Bigr)))
```
"""
_doc_ApproxHessian_step = raw"\operatorname{retr}_p(\frac{c}{\lVert X \rVert_p}X)"
