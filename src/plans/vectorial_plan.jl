@doc raw"""
    AbstractVectorialType

An abstract type for different representations of a vectorial function
    ``f: \mathcal → \mathbb R^m`` and its (component-wise) gradient/Jacobian
"""
abstract type AbstractVectorialType end

@doc raw"""
    CoefficientVectorialType{B<:AbstractBasis} <: AbstractVectorialType

    A type to indicate that gradient of the constraints is implemented as a
Jacobian matrix with respect to a certain basis, that is if we have constraints
``g: \mathcal M → ℝ^m`` and a basis ``\mathcal B`` of ``T_p\mathcal M``, at ``p∈ \mathcal M``

We have ``J_g(p) = (c_1^{\mathrm{T)},…,c_m^{\mathrm{T)})^{\mathrm{T)} \in ℝ^{m,d}``, that is,
every row ``c_i`` of this matrix is a set of coefficients such that
`get_coefficients(M, p, c, B)` is the tangent vector ``\oepratorname{grad} g_i(p)``

for example ``g_i(p) ∈ ℝ^m`` or ``\operatorname{grad} g_(p) ∈ T_p\mathcal M``,
    ``i=1,…,m``.

# Fields

* `basis` an [`AbstractBasis`](@extref) to indicate the default representation.
"""
struct CoefficientVectorialType{B<:AbstractBasis} <: AbstractVectorialType
    basis::B
end
@doc raw"""
    ComponentVectorialType <: AbstractVectorialType

A type to indicate that constraints are implemented as component functions,
for example ``g_i(p) ∈ ℝ^m`` or ``\operatorname{grad} g_(p) ∈ T_p\mathcal M``, ``i=1,…,m``.
"""
struct ComponentVectorialType <: AbstractVectorialType end

@doc raw"""
    FunctionVectorialType <: AbstractVectorialType

 A type to indicate that constraints are implemented one whole functions,
for example ``g(p) ∈ ℝ^m`` or ``\operatorname{grad} g(p) ∈ (T_p\mathcal M)^m``.
"""
struct FunctionVectorialType <: AbstractVectorialType end

@doc raw"""
    VectorGradientFunction{E, FT, JT, F, J, I} <: <: AbstractManifoldObjective{E}

Represent a function ``f:\mathcal M → ℝ^n`` including it first derivatie,
either as a vector of gradients of a Jacobian

# Representations of ``f``

There are htree different representations of ``f``, which might be benefictial in one or
the other situation:
* the [`FunctionVectorialType`](@ref),
* the [`ComponentVectorialType`](@ref),
* the [`CoefficientVectorialType`](@ref) with respect to a specific basis of the tangent space.

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

For the [`ComponentVectorialType`](@ref) of `f`, each of the component functions
is a (classical) objective and hence has a gradient ``\oepratorname{grad} f_i(p) ∈ T_p\mathcal M``.
Putting these gradients into a vector the same way as the functions, we obtain a
[`ComponentVectorialType`](@ref)

```math
\operatorname{grad} f(p) = \Bigl( \operatorname{grad} f_1(p), \operatorname{grad} f_2(p), …, \operatorname{grad} f_n(p) \Bigr)^{\mathrm{T}}
∈ (T_p\mathcal M)^n
```

And advantage here is, that again the single components can be evaluated individually

# Fields

* `costs!!`:          the cost function ``f``, which can take different formats
* `cost_type`:     indicating / string data for the type of `f`
* `jacobian!!:     the jacobian of ``f``
* `jacobian_type`: indicating / storing data for the type of ``J_f``
* `parameters`:    the nunmber `n` from above, that is the size of the vector ``f`` returns.

# Constructor

VectorGradientFunction(f, Jf, range_dimension;
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
    jacobian!!::J
    range_dimension::I
end

function VectorGradientFunction(
    f::F,
    Jf::J,
    range_dimension::I;
    evaluation::E=AllocatingEvaluation(),
    function_type::FT=FunctionVectorialType(),
    jacobian_type::JT=FunctionVectorialType(),
) where {
    I,F,J,E<:AbstractEvaluationType,FT<:AbstractVectorialType,JT<:AbstractVectorialType
}
    return VectorGradientFunction{E,F,FT,J,JT,I}(f, Jf, range_dimension)
end

@doc raw"""
    get_cost(M::AbstractManifold, vgf::VectorGradientFunction, p, i)

Evaluate the ``i``th component of the cost.
Note that for some types, this might still mean, that the whole vector os costs ahs to be evaluated.
"""
get_cost(M::AbstractManifold, vgf::VectorGradientFunction, p, i)
function get_cost(
    M::AbstractManifold,
    vgf::VectorGradientFunction{<:AllocatingEvaluation,<:FunctionVectorialType},
    p,
    i,
)
    return vgf.costs!!(M, p)[i]
end
function get_cost(
    M, vgf::VectorGradientFunction{<:AllocatingEvaluation,<:ComponentVectorialType}, p
)
    return vgf.costs!![i](M, p)
end
function get_cost(
    M::AbstractManifold,
    vgf::VectorGradientFunction{<:InplaceEvaluation,<:FunctionVectorialType},
    p,
)
    x = zeros(vgf.range_dimension)
    get_costs(M, x, vgf, p)
    return x[i]
end

function get_costs end

@doc raw"""
    get_costs(M, vgf::VectorGradientFunction, p, range=nothing)
    get_costs!(M, x, vgf::VectorGradientFunction, p, range=nothing)

Evaluate the function ``f: \mathcal M → \mathbb R^n`` stored in the [`VectorGradientFunction`](@ref) `vgf` at `p`.
This can also be done in place of `x`.

The last optional argument `range` is provided for consistency with the other access
functions of the [`VectorGradientFunction`](@ref). For now the only assumed
range is a vector.
"""
get_costs(M::AbstractManifold, vgf::VectorGradientFunction, p, range=nothing)

function get_costs(
    M::AbstractManifold,
    vgf::VectorGradientFunction{<:AllocatingEvaluation,<:FunctionVectorialType},
    p,
    range=nothing,
)
    return vgf.costs!!(M, p)
end
function get_costs(
    M,
    vgf::VectorGradientFunction{<:AllocatingEvaluation,<:ComponentVectorialType},
    p,
    range=nothing,
)
    return [fi(M, p) for fi in vgf.costs!!]
end
function get_costs(
    M::AbstractManifold, vgf::VectorGradientFunction{<:InplaceEvaluation}, p, range=nothing
)
    x = zeros(vgf.range_dimension)
    return get_costs!(M, x, vgf, p, range)
end

function get_costs!(
    M::AbstractManifold,
    x,
    vgf::VectorGradientFunction{<:AllocatingEvaluation,<:FunctionVectorialType},
    p,
    range=nothing,
)
    x .= vgf.costs!!(M, p)
    return x
end
function get_costs!(
    M,
    x,
    vgf::VectorGradientFunction{<:AllocatingEvaluation,<:ComponentVectorialType},
    p,
    range=nothing,
)
    for (xi, fi) in zip(x, vgf.costs)
        xi = fi(M, p)
    end
    return x
end
function get_costs!(
    M::AbstractManifold,
    x,
    vgf::VectorGradientFunction{<:InplaceEvaluation,<:FunctionVectorialType},
    p,
    range=nothing,
)
    return vgf.costs(M, x, p)
end
# does vgf::VectorGradientFunction{<:AllocatingEvaluation,<:ComponentVectorialType}
# make sense at all, the components would return floats (immutables) so probably not

@doc raw"""
    get_costs_function(vgf::VectorGradientFunction, recursive=false)

return the internally stored costs function.
Note the additional _s_ compared to the usual [`get_cost_function`](@ref).
"""
function get_costs_function(vgf::VectorGradientFunction, recursive=false)
    return vgf.costs!!
end

function get_gradients end

@doc raw"""
    get_gradients(
        M::AbstractManifold,
        vgf::VectorGradientFunction,
        p,
        range=PowerManifold(M, NestedPowerRepresentation(), vgf.range_dimension...)
        )

Evaluate and return the gradients of the components of ``f``, by interpreting its
components as objectives and returning their gradients as an element on.
Here `range` denotes the (power) manifold, used to define the tangent space
the gradients seen as a single tangent vector lie in. The default corresponds
tp

!!! note
    The power manifold is only created if necessary;
    it is usually recommended to pass it to the function to
    either increase performance and/or change the default representation
"""
get_gradients(M::AbstractManifold, vgf::VectorGradientFunction, p)

function get_gradients(
    M::AbstractManifold,
    vgf::VectorGradientFunction{<:AllocatingEvaluation,FT,<:CoefficientVectorialType},
    p,
    range=PowerManifold(M, NestedPowerRepresentation(), vgf.range_dimension...),
) where {FT}
    P = fill(range, p)
    X = zero_vector(range, P)
    return get_gradients!(M, X, vgf, p, range)
end
function get_gradients(
    M::AbstractManifold,
    vgf::VectorGradientFunction{<:AllocatingEvaluation,FT,<:FunctionVectorialType},
    p,
    range=nothing,
) where {FT}
    return vgf.jacobian!!(M, p) # Returns an element on the power manifold anyways
end
function get_gradients(
    M::AbstractManifold,
    vgf::VectorGradientFunction{<:InplaceEvaluation,FT,<:ComponentVectorialType},
    p,
    range=PowerManifold(M, NestedPowerRepresentation(), vgf.range_dimension...),
) where {FT}
    P = fill(range, p)
    X = zero_vector(range, P)
    return get_gradients!(M, X, vgf, p, range)
end

function get_gradients!(
    M::AbstractManifold,
    X,
    vgf::VectorGradientFunction{<:AllocatingEvaluation,FT,<:CoefficientVectorialType},
    p,
    range=PowerManifold(M, NestedPowerRepresentation(), vgf.range_dimension...),
) where {FT}
    JF = vgf.jacobian!!(M, p)
    for i in ManifoldsBase.get_iterator(range)
        get_vector!(M, X[M, i], p, JF[i..., :]) #convert rows to gradients
    end
    return X
end
function get_gradients!(
    M::AbstractManifold,
    X,
    vgf::VectorGradientFunction{<:AllocatingEvaluation,FT,<:FunctionVectorialType},
    p,
    range=PowerManifold(M, NestedPowerRepresentation(), vgf.range_dimension...),
) where {FT}
    copyto!(range, X, vgf.jacobian!!(M, p))
    return X
end

function get_gradients!(
    M::AbstractManifold,
    X,
    vgf::VectorGradientFunction{<:InplaceEvaluation,FT,<:CoefficientVectorialType},
    p,
    range=PowerManifold(M, NestedPowerRepresentation(), vgf.range_dimension...),
) where {FT}
    # A bit tricky since we might have to allocate an mxd matrix
    return error("TODO")
end

function get_gradients!(
    M::AbstractManifold,
    X,
    vgf::VectorGradientFunction{<:InplaceEvaluation,FT,<:ComponentVectorialType},
    p,
    range=PowerManifold(M, NestedPowerRepresentation(), vgf.range_dimension...),
) where {FT}
    for i in ManifoldsBase.get_iterator(range)
        vgf.jacobian!!(M, X[M, i], p)
    end
    return X
end

function get_gradients!(
    M::AbstractManifold,
    X,
    vgf::VectorGradientFunction{<:InplaceEvaluation,FT,<:FunctionVectorialType},
    p,
    range=nothing,
) where {FT}
    return vgf.jacobian!!(M, X, p)
end

@doc raw"""
    get_gradient(M::AbstractManifold, vgf::VectorGradientFunction, p, i)

Evaluate and return the `ì``th gradient of the component of ``f``, that is of the
``i``th component function.
"""
get_gradient(M::AbstractManifold, vgf::VectorGradientFunction, p, i)

#TODO Check all cases and compose vectors in case of coefficient representation

function get_jacobian end

@doc raw"""
    get_jacobian(M::AbstractManifold, vgf::VectorGradientFunction, p, B)

Evaluate and return the Jacobian with respect to a basis of the tangent space.
"""
get_jacobian(M::AbstractManifold, vgf::VectorGradientFunction, p, B)

#TODO Check all cases and decompose vectors in basis B to get Jacobian
# For jacobian representation check to maybe do a change of basis if the indernal basis does not agree with B? Or error?

function get_jacobian_function(vgf::VectorGradientFunction, recursive=false)
    return vgf.jacobian!!
end
