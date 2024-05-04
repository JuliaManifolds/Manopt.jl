@doc raw"""
    AbstractVectorialType

An abstract type for different representations of a vectorial function
    ``f: \mathcal → \mathbb R^m`` and its (component-wise) gradient/Jacobian
"""
abstract type AbstractVectorialType end

@doc raw"""
    CoefficientVectorialType{B<:AbstractBasis} <: AbstractVectorialType

"""
struct CoefficientVectorialType{B<:AbstractBasis} <: AbstractVectorialType
    basis::B
end

@doc raw"""
    ComponentVectorialType <: AbstractVectorialType

"""
struct ComponentVectorialType <: AbstractVectorialType end

@doc raw"""
    FunctionVectorialType <: AbstractVectorialType

"""
struct FunctionVectorialType <: AbstractVectorialType end

@doc raw"""
    PowerManifoldVectorialType <: AbstractVectorialType

"""
struct PowerManifoldVectorialType <: AbstractVectorialType end

@doc raw"""
    VectorialGradientObjective{E, F, FT, J, FT, I} <: <: AbstractManifoldObjective{E}

Represent an objective ``f:\mathcal M → \mathbb R^n``, that is a function on a manifold,
that maps into a vector space.

# Representations of ``f``

There are two different representations of ``f``, which might be benefictial in one or
the other situation: the [`FunctionVectorialType`](@ref) and the [`ComponentVectorialType`](@ref).

For the [`ComponentVectorialType`](@ref) imagine that ``f`` could also be written
using its component functions,

```math
f(p) = \bigl( f_1(p), f_2(p), \ldots, f_n(p) \bigr)^{\mathrm{T}}
```

In this representation `f` is given as a vector `[f1(M,p), f2(M,p), ..., fn(M,p)]`
of its component functions.
An advantage is that the single components can be evaluated and from this representation
one even can directly read of the number `n`. A disadvantage might be, that one has to
implement a lot of individual (component) functions.

For the  [`FunctionVectorialType`](@ref) ``f`` is implemented as a single function
`f(M, p)`, that returns an `AbstractArray`.
And advantage here is, that this is a single function. A disadvantage might be,
that if this is expensive even to compute a single component, all of `f` has to be evaluated

# Representations of the Jacobian ``J_f`` or vectorial ``\operatorname{grad} f`` or its and its jacobian

For the [`ComponentVectorialType`](@ref) of `f`, each of the component functions
is a (classical) objective and hence has a gradient ``\oepratorname{grad} f_i(p) ∈ T_p\mathcal M``.
Putting these gradients into a vector the same way as the functions, we obtain a
[`ComponentVectorialType`](@ref)

```math
\operatorname{grad} f(p) = \Bigl( \operatorname{grad} f_1(p), \operatorname{grad} f_2(p), …, \operatorname{grad} f_n(p) \Bigr)^{\mathrm{T}}
∈ (T_p\mathcal M)^n
```

And advantage here is, that again the single components can be evaluated individually

For the [`FunctionVectorialType`](@ref) ``f`` there are three different variants possible.

1. similar to the previous case, returning a [`ComponentVectorialType`](@ref) ``\operatorname{grad} f(p) \in (T_p\mathcal M)^n``
  is also possible when implementing a single function `grad_f(M, p)` returning such a vector of tangent vectors.
2. Implementing a [`PowerManifoldVectorialType`](@ref), the single function `grad_f(M, p)` returns a tangent vector
  on the power manifolds tangent space ``\operatorname{grad} f(p) ∈ T_P\mathcal M^n``, where ``P = (p,…,p)^{\mathrm{T}} \in \mathcal M^n``.
  This representation can use any representation of tangent vectors on the power manifold,
  where the [`NestedPowerRepresentation`](@extref) yields the same as in the first case, but
  one could as well use the [`ArrayPowerRepresentation`](@extref), that is, a single array.
3. the [`CoefficientVectorialType`](@ref) would return a matrix ``J ∈ \mathbb R^{d\times n}``
  with respect to an [`AbstractBasis`](@ref) on the tangent space ``T_p\mathcal M``.
  While the necessity for a basis might be a disadvantage, this might be the least memory usage
  one and comes closest to a “classical” Jacobian for a function between vector spaces.

# Fields

* `cost`:          the cost function ``f``, which can take different formats
* `cost_type`:     indicating the type of `f`
* `jacobian!!:     the jacobian of ``f``
* `jacobian_type`: JT
* `num_components: the nunmber `n` from above, that is the size of the vector ``f`` returns.

"""
struct VectorialGradientObjective{
    E<:AbstractEvaluationType,
    F,
    FT<:AbstractVectorialType,
    J,
    JT<:AbstractVectorialType,
    I<:Integer,
} <: AbstractManifoldObjective{E}
    cost::F
    cost_type::FT
    jacobian!!::J
    jacobian_type::JT
    num_components::I
end
