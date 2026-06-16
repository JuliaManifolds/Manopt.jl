@doc """
    ManifoldNonlinearLeastSquaresObjective{E<:AbstractEvaluationType} <: AbstractManifoldObjective{E}

An objective to model the robustified nonlinear least squares problem

$(_problem(:NonLinearLeastSquares))

# Fields

* `objective`: a vector of [`AbstractFirstOrderVectorFunction`](@ref)`{E}`s, one for each
  block component cost function ``F_i``, which might internally also be a vector of component costs ``(F_i)_j``,
  as well as their Jacobian ``J_{F_i}`` or a vector of gradients ``$(_tex(:grad)) (F_i)_j``
  depending on the specified [`AbstractVectorialType`](@ref)s.
* `robustifier`: a vector of [`AbstractRobustifierFunction`](@ref)`s`, one for each
  block component cost function ``F_i``.
* `value_cache::AbstractVector` and internal cache to store the result of evaluating the cost functions

# Constructors

    ManifoldNonlinearLeastSquaresObjective(f, jacobian, range_dimension::Integer, robustifier=IdentityRobustifier(); kwargs...)

Create a nonlinear least squares objective for a single vectorial function `f` and its `jacobian`,
where `range_dimension` is the dimension of the vector space `f` maps into. These three are internally
wrapped into a [`VectorGradientFunction`](@ref) and calls the following constructor.

    ManifoldNonlinearLeastSquaresObjective(vf::AbstractFirstOrderVectorFunction, robustifier::AbstractRobustifierFunction=IdentityRobustifier())

Create a nonlinear least squares objective for a given vectorial function.
Note that for this constructor the `robustifier` is applied componentwise to each component of `vf`,
i.e. wrapped in a [`ComponentwiseRobustifierFunction`](@ref).
Internally this wraps both `vf` and `robustifier` in an array and calls the next constructor.
Hence to not use the componentwise robustifier but a global one, pass `[vf,]` and `[robustifier,]` instead.

    ManifoldNonlinearLeastSquaresObjective(fs::Vector{<:AbstractFirstOrderVectorFunction}, robustifiers::Vector{<:AbstractRobustifierFunction}=fill(IdentityRobustifier(), length(fs)))

Given a vector of [`AbstractFirstOrderVectorFunction`](@ref)`s to represent the single blocks
and a vector of robustifiers, one for each block, create the corresponding nonlinear least squares objective.

# Keyword arguments

The first constructor allows to pass the following keyword arguments, that are passed on to
the corresponding
the constructor of the As well as for the first variant of having a single block

$(_kwargs(:evaluation))
* `function_type::`[`AbstractVectorialType`](@ref)`=`[`FunctionVectorialType`](@ref)`()`: specify
  the format the residuals are given in. By default a function returning a vector.
* `jacobian_tangent_basis::AbstractBasis=DefaultOrthonormalBasis()`; shortcut to specify
  the basis the Jacobian matrix is build with.
* `jacobian_type::`[`AbstractVectorialType`](@ref)`=`[`CoefficientVectorialType`](@ref)`(jacobian_tangent_basis)`:
  specify the format the Jacobian is given in. By default a matrix of the differential with
  respect to a certain basis of the tangent space.

# See also

[`LevenbergMarquardt`](@ref), [`LevenbergMarquardtState`](@ref)
"""
struct ManifoldNonlinearLeastSquaresObjective{
        E <: AbstractEvaluationType, VF <: AbstractFirstOrderVectorFunction{E},
        RF <: AbstractRobustifierFunction, TVC <: AbstractVector,
    } <: AbstractManifoldFirstOrderObjective{E, Vector{VF}}
    objective::Vector{VF}
    robustifier::Vector{RF}
    value_cache::TVC
    # block components case constructor
    function ManifoldNonlinearLeastSquaresObjective(
            fs::Vector{VF},
            robustifiers::Vector{RV} = fill(IdentityRobustifier(), length(fs)),
            value_cache::TVC = zeros(sum(length(f) for f in fs)),
        ) where {
            E <: AbstractEvaluationType,
            VF <: AbstractFirstOrderVectorFunction{E},
            RV <: AbstractRobustifierFunction, TVC <: AbstractVector,
        }
        # we need to check that the lengths match
        (length(fs) != length(robustifiers)) && throw(
            ArgumentError(
                "Number of functions ($(length(fs))) does not match number of robustifiers ($(length(robustifiers)))",
            ),
        )
        return new{E, VF, RV, TVC}(fs, robustifiers, value_cache)
    end
    # single component case constructor
    function ManifoldNonlinearLeastSquaresObjective(
            f::F,
            robustifier::R = IdentityRobustifier(),
            value_cache::TVC = zeros(length(f)),
        ) where {E <: AbstractEvaluationType, F <: AbstractFirstOrderVectorFunction{E}, R <: AbstractRobustifierFunction, TVC <: AbstractVector}
        cr = ComponentwiseRobustifierFunction(robustifier)
        return new{E, F, typeof(cr), TVC}([f], [cr], value_cache)
    end
end
function ManifoldNonlinearLeastSquaresObjective(
        f, jacobian, range_dimension::Integer,
        robustifier::AbstractRobustifierFunction = IdentityRobustifier();
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        jacobian_tangent_basis::AbstractBasis = DefaultOrthonormalBasis(),
        jacobian_type::AbstractVectorialType = CoefficientVectorialType(jacobian_tangent_basis),
        function_type::AbstractVectorialType = FunctionVectorialType(),
    )
    vgf = VectorGradientFunction(
        f, jacobian, range_dimension;
        evaluation = evaluation, jacobian_type = jacobian_type, function_type = function_type,
    )
    return ManifoldNonlinearLeastSquaresObjective(vgf, robustifier)
end

"""
    residuals_count(nlso::ManifoldNonlinearLeastSquaresObjective)

Return the total number of residuals in [`ManifoldNonlinearLeastSquaresObjective`](@ref) `nlso`,
which is the sum of the single block components lengths.
"""
function residuals_count(nlso::ManifoldNonlinearLeastSquaresObjective)
    return sum(length(o) for o in nlso.objective)
end
residuals_count(admo::AbstractDecoratedManifoldObjective) = residuals_count(get_objective(admo, false))

"""
    get_cost(M::AbstractManifold, nlso::ManifoldNonLinearLeastSquaresObjective, p)

Compute the cost of the least squares objective, i.e.

```math
$(_tex(:frac, "1", "2")) $(_tex(:sum, "i=1", "m")) ρ_i $(_tex(:bigl))( $(_tex(:norm, "F_i(p)"))^2 $(_tex(:bigr))),
```

where ``F_i: $(_math(:Manifold)) → ℝ^{n_i}`` is the ``i``th block component of length ``n_i > 0``
and each ``ρ_i: ℝ → ℝ`` is a [* R robustifier function, cf. [`AbstractRobustifierFunction`](@ref),
for each such a block component.
"""
function get_cost(M::AbstractManifold, nlso::ManifoldNonlinearLeastSquaresObjective, p)
    v = 0.0
    start = 0
    get_residuals!(M, nlso.value_cache, nlso, p)
    for (o, r) in zip(nlso.objective, nlso.robustifier)
        len = length(o)
        value_cache = view(nlso.value_cache, (start + 1):(start + len))
        v += _get_cost(M, o, r, p; value_cache = value_cache)
        start += len
    end
    v /= 2
    return v
end
# For a single block – or one summand in the docs of the previous function
function _get_cost(
        M, vgf::AbstractFirstOrderVectorFunction, r::AbstractRobustifierFunction, p;
        value_cache = get_value(M, vgf, p)
    )
    vi = sum(abs2, value_cache)
    (a, _, _) = get_robustifier_values(r, vi)
    return a
end
# For a single vectorial function where the robustifier is applied to every in dex separately.
function _get_cost(
        M, vgf::AbstractFirstOrderVectorFunction, cr::ComponentwiseRobustifierFunction, p;
        value_cache = get_value(M, vgf, p)
    )
    v = abs2.(value_cache)
    # componentwise robustify
    (a, _, _) = get_robustifier_values(cr, v)
    return sum(a)
end

_doc_get_gradient_nlso = """
    get_gradient(M::AbstractManifold, nlso::ManifoldNonlinearLeastSquaresObjective, p; kwargs...)
    get_gradient!(M::AbstractManifold, X, nlso::ManifoldNonlinearLeastSquaresObjective, p; kwargs...)

Compute the gradient for the [`ManifoldNonlinearLeastSquaresObjective`](@ref) `nlso` at the point ``p ∈ M``,
i.e.

```math
$(_tex(:grad)) f(p) = $(_tex(:sum, "i=1", "m")) ρ'_i$(_tex(:bigl))($(_tex(:norm, "F_i(p)"; index = "2"))^2$(_tex(:bigr)))
$(_tex(:sum, "j=1", "n_i")) f_{i,j}(p) $(_tex(:grad)) f_{i,j}(p)
```

where ``F_i(p) ∈ ℝ^{n_i}`` is the vector of residuals for the `i`-th block component cost function
and ``f_{i,j}(p)`` its `j`-th component function.

# Keyword arguments
* `value_cache=nothing`: if provided, this vector is used to store the residuals ``F(p)``
  internally to avoid re-computations.
* `jacobian_cache=fill(nothing, length(nlso.objective))`: if provided, this is used to store
  the Jacobians of the component functions.
"""
@doc "$(_doc_get_gradient_nlso)"
function get_gradient(
        M::AbstractManifold, nlso::ManifoldNonlinearLeastSquaresObjective, p; kwargs...,
    )
    X = zero_vector(M, p)
    return get_gradient!(M, X, nlso, p; kwargs...)
end
function get_gradient!(
        M::AbstractManifold, X, nlso::ManifoldNonlinearLeastSquaresObjective, p;
        value_cache = nothing, jacobian_cache = fill(nothing, length(nlso.objective)),
    )
    zero_vector!(M, X, p)
    start = 0
    for (o, r, jb) in zip(nlso.objective, nlso.robustifier, jacobian_cache) # for every block
        len = length(o)
        Fi = isnothing(value_cache) ? get_value(M, o, p) : view(value_cache, (start + 1):(start + len))
        _add_gradient!(M, X, o, r, p; value_cache = Fi, jacobian_cache = jb)
        start += len
    end
    return X
end
# Gradient for a single summand from above, that is a single (robustified) block
function _add_gradient!(
        M, X, vgf::AbstractFirstOrderVectorFunction, r::AbstractRobustifierFunction, p;
        value_cache = get_value(M, vgf, p), jacobian_cache = nothing
    )
    # get gradients for every component
    len = length(vgf)

    # compute robustifier derivative
    (_, b, _) = get_robustifier_values(r, sum(abs2, value_cache))
    if isnothing(jacobian_cache)
        Y = allocate(M, X)
        for j in 1:len
            get_gradient!(M, Y, vgf, p, j) # gradient of f_{i,j}
            X .+= (b * value_cache[j]) .* Y
        end
    else
        Jc = jacobian_cache' * value_cache
        Jc .*= b
        add_vector!(M, X, p, Jc, vgf.jacobian_type.basis)
    end
    return X
end
# Gradient for a single summand from above, that is a single (robustified) block where the
# robustifier is applied to every component / coordinate
function _add_gradient!(
        M, X, vgf::AbstractFirstOrderVectorFunction, cr::ComponentwiseRobustifierFunction, p;
        value_cache = get_value(M, vgf, p), jacobian_cache = nothing,
    )
    # get gradients for every component
    len = length(vgf)
    r = cr.robustifier
    zero_vector!(M, X, p)
    Y = copy(M, p, X)
    for j in 1:len
        get_gradient!(M, Y, vgf, p, j) # gradient of f_{i,j}
        (_, b, _) = get_robustifier_values(r, abs(value_cache[j])^2)
        # compute robustifier derivative
        X .+= (b * value_cache[j]) .* Y
    end
    return X
end

# --- Residuals
_doc_get_residuals_nlso = """
    get_residuals(M::AbstractManifold, nlso::ManifoldNonlinearLeastSquaresObjective, p)
    get_residuals!(M::AbstractManifold, v, nlso::ManifoldNonlinearLeastSquaresObjective, p)

Compute the vector of residuals ``F(p) ∈ ℝ^n``, ``n = $(_tex(:sum, "1", "m")) n_i``.
In other words this is the concatenation of the residual vectors ``F_i(p)``, ``i=1,…,m``
of the components of the the [`ManifoldNonlinearLeastSquaresObjective`](@ref) `nlso`
at the current point ``p`` on `M`.

This can be computed in-place of `v`.

Note that even in the presence of [`RobustifierFunction`](@ref)s, these are not applied here,
this function computes the “pure” residuals.
"""

@doc "$(_doc_get_residuals_nlso)"
function get_residuals(
        M::AbstractManifold, o::AbstractManifoldObjective, p; kwargs...
    )
    v = zeros(residuals_count(o))
    return get_residuals!(M, v, o, p; kwargs...)
end

@doc "$(_doc_get_residuals_nlso)"
function get_residuals!(
        M::AbstractManifold, v, nlso::ManifoldNonlinearLeastSquaresObjective, p; kwargs...,
    )
    start = 0
    for o in nlso.objective # for every block
        len = length(o)
        view_v = view(v, (start + 1):(start + len))
        get_value!(M, view_v, o, p)
        start += len
    end
    return v
end
function get_residuals!(M::AbstractManifold, v, admo::AbstractDecoratedManifoldObjective, p; kwargs...)
    return get_residuals!(M, v, get_objective(admo, false), p; kwargs...)
end

function Base.show(io::IO, nlso::ManifoldNonlinearLeastSquaresObjective)
    print(io, "ManifoldNonlinearLeastSquaresObjective(")
    print(io, nlso.objective, ", ", nlso.robustifier, ", ", nlso.value_cache)
    return print(io, ")")
end
function status_summary(nlso::ManifoldNonlinearLeastSquaresObjective; context::Symbol = :default)
    (context === :short) && (return repr(nlso))
    n = length(nlso.objective)
    return ("A nonlinear least squares objective $(n) vectorial block$(n > 1 ? "s" : "")")
end
