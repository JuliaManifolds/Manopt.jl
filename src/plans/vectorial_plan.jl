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
    VectorGradientFunction{E, FT, JT, F, J, I} <: <: AbstractManifoldObjective{E}

Represent a function ``f:\mathcal M → \mathbb R^n`` including its gradient.

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

For the [`FunctionVectorialType`](@ref) ``f`` there are two different variants possible.

1. Implementing a [`PowerManifoldVectorialType`](@ref), the single function `grad_f(M, p)` returns a tangent vector
   on the power manifolds tangent space ``\operatorname{grad} f(p) ∈ T_P\mathcal M^n``, where ``P = (p,…,p)^{\mathrm{T}} \in \mathcal M^n``.
   This representation can use any representation of tangent vectors on the power manifold,
   where the [`NestedPowerRepresentation`](@extref) yields the same as in the first case, but
   one could as well use the [`ArrayPowerRepresentation`](@extref), that is, a single array.
3. the [`CoefficientVectorialType`](@ref) would return a matrix ``J ∈ \mathbb R^{d\times n}``
   with respect to an [`AbstractBasis`](@ref) on the tangent space ``T_p\mathcal M``.
   While the necessity for a basis might be a disadvantage, this might be the least memory usage
   one and comes closest to a “classical” Jacobian for a function between vector spaces.

!!! note
    Since this is not a classical objective mappint into the real numbers, the struct
    has a name distinguishing it from the objective. Nevertheless, it behaves in several
    aspects very similar to an objective.
    A major difference is, that the “cost” is vectorial and hence can also follow the
    [`AbstractEvaluationType`](@ref) rules, cf [`get_costs`](@ref) and that the singular
    function [`get_cost`](@ref get_cost(::AbstractManifold, ::VectorGradientFunction, ::Any, ::Any))
    requires the index if the component as well

# Fields

* `costs!!`:          the cost function ``f``, which can take different formats
* `cost_type`:     indicating / string data for the type of `f`
* `jacobian!!:     the jacobian of ``f``
* `jacobian_type`: indicating / stpring data for the type of ``J_f``
* `parameters`:    the nunmber `n` from above, that is the size of the vector ``f`` returns.

# Constructor

VectorGradientFunction(f, Jf, vector_dimension;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    function_type::AbstractVectorialType=FunctionVectorialType(),
    jacobian_type::AbstractVectorialType=FunctionVectorialType(),
)
"""
struct VectorGradientFunction{
    E<:AbstractEvaluationType,
    FT<:AbstractVectorialType,
    JT<:AbstractVectorialType,
    F,
    J,
    I<:Integer,
} # <: AbstractManifoldObjective{E} # maybe not?
    costs!!::F
    costs_type::FT
    jacobian!!::J
    jacobian_type::JT
    vector_dimension::I
end

function VectorGradientFunction(
    f::F,
    Jf::J,
    vector_dimension::I;
    evaluation::E=AllocatingEvaluation(),
    function_type::FT=FunctionVectorialType(),
    jacobian_type::JT=FunctionVectorialType(),
) where {
    I,F,J,E<:AbstractEvaluationType,FT<:AbstractVectorialType,JT<:AbstractVectorialType
}
    return VectorGradientFunction{E,F,FT,J,JT,I}(
        f, function_type, Jf, jacobian_type, vector_dimension
    )
end

function get_costs end

@doc raw"""
    get_costs(M, vfg::VectorGradientFunction, p)
    get_costs!(M, x, vfg::VectorGradientFunction, p)

Evaluate the function ``f: \mathcal M → \mathbb R^n`` stored in the [`VectorGradientFunction`](@ref) `vgf` at `p`.
This can also be done in place of `x`.
"""
get_costs(M::AbstractManifold, vfg::VectorGradientFunction, p)

function get_costs(
    M::AbstractManifold,
    vfg::VectorGradientFunction{<:AllocatingEvaluation,<:FunctionVectorialType},
    p,
)
    return vfg.costs!!(M, p)
end
function get_costs(
    M, vfg::VectorGradientFunction{<:AllocatingEvaluation,<:ComponentVectorialType}, p
)
    return [fi(M, p) for fi in vfg.costs!!]
end
function get_costs(M::AbstractManifold, vfg::VectorGradientFunction{<:InplaceEvaluation}, p)
    x = zeros(vfg.vector_dimension)
    return get_costs(M, x, vgf, p)
end

function get_costs!(
    M::AbstractManifold,
    x,
    vfg::VectorGradientFunction{<:AllocatingEvaluation,<:FunctionVectorialType},
    p,
)
    x .= vfg.costs!!(M, p)
    return x
end
function get_costs!(
    M, x, vfg::VectorGradientFunction{<:AllocatingEvaluation,<:ComponentVectorialType}, p
)
    for (xi, fi) in zip(x, bgf.costs)
        xi = fi(M, p)
    end
    return x
end
function get_costs!(
    M::AbstractManifold,
    x,
    vfg::VectorGradientFunction{<:InplaceEvaluation,<:FunctionVectorialType},
    p,
)
    return vgs.costs(M, x, p)
end
# does vfg::VectorGradientFunction{<:AllocatingEvaluation,<:ComponentVectorialType}
# make sense at all, the components would return integers so probably not

@doc raw"""
    get_cost(M::AbstractManifold, vgf::VectorGradientFunction, p, i)

Evaluate the ``i``th component of the cost.
Note that for some types, this might still mean, that the whole vector os costs ahs to be evaluated.
"""
get_cost(M::AbstractManifold, vfg::VectorGradientFunction, p, i)
function get_cost(
    M::AbstractManifold,
    vfg::VectorGradientFunction{<:AllocatingEvaluation,<:FunctionVectorialType},
    p,
    i,
)
    return vfg.costs!!(M, p)[i]
end
function get_cost(
    M, vfg::VectorGradientFunction{<:AllocatingEvaluation,<:ComponentVectorialType}, p
)
    return vfg.costs!![i](M, p)
end
function get_cost(
    M::AbstractManifold,
    vfg::VectorGradientFunction{<:InplaceEvaluation,<:FunctionVectorialType},
    p,
)
    x = zeros(vfg.vector_dimension)
    get_costs(M, x, vgf, p)
    return x[i]
end
