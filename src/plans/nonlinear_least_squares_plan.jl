
@doc """
    NonlinearLeastSquaresObjective{E<:AbstractEvaluationType} <: AbstractManifoldObjective{T}

An objective to model the nonlinear least squares problem

$(_problem(:NonLinearLeastSquares))

Specify a nonlinear least squares problem

# Fields

* `objective`: a [`AbstractVectorGradientFunction`](@ref)`{E}` containing both the vector of cost functions ``f_i`` as well as their gradients ``$(_tex(:grad)) f_i```
* `smoothing`: a [`ManifoldHessianObjective`](@ref) or a [`Vector of a smoothing function ``s: ℝ → ℝ``, hence including its first and second derivatives ``s'`` and ``s''``.

This `NonlinearLeastSquaresObjective` then has the same [`AbstractEvaluationType`](@ref) `T`
as the (inner) `objective`.
The smoothing is expected to be be of [`AllocatingEvaluation`](@ref) type,
since it works on a one-dimensional vector space ``ℝ`` or each of its components does.

# Constructors

    NonlinearLeastSquaresObjective(f, jacobian, range_dimension; kwargs...)
    NonlinearLeastSquaresObjective(vf::AbstractVectorGradientFunction, smoothing = :Identity)

# Arguments

* `f` the vectorial cost function ``f: $(_math(:M)) → ℝ^m``
* `jacobian` the Jacobian, might also be a vector of gradients of the component functions of `f`
* `range_dimension` the number of dimensions `m` the function `f` maps into

These three can also be passed as a [`AbstractVectorGradientFunction`](@ref) `vf` already.
For this second constructor, the `smoothing=` keyword is the second positional argument.

# Keyword arguments

$(_var(:Keyword, :evaluation))
* `function_type::`[`AbstractVectorialType`](@ref)`=`[`FunctionVectorialType`](@ref)`()`: format the
* `jacobian_tangent_basis::AbstractBasis=DefaultOrthonormalBasis()`; shortcut to specify the basis the Jacobian matrix is build with.
* `jacobian_type::`[`AbstractVectorialType`](@ref)`=`[`CoordinateVectorialType`](@ref)`(jacobian_tangent_basis)`: specify the format the Jacobian is given in. By default a matrix of the differential with respect to a certain basis of the tangent space.
* `smoothing::Union{`[`ManifoldHessianObjective`](@ref),[`VectorHessianFunction`](@ref)`,Symbol}=:Identity`: specify the type of smoothing,
  either applied to all components, a vector for each component or a symbol passed to the [`smoothing_factory`](@ref) of predefined smoothing types.

# See also

[`LevenbergMarquardt`](@ref), [`LevenbergMarquardtState`](@ref)
"""
struct NonlinearLeastSquaresObjective{
    E<:AbstractEvaluationType,
    F<:AbstractVectorGradientFunction{E},
    R<:Union{AbstractManifoldHessianObjective,VectorHessianFunction},
} <: AbstractManifoldGradientObjective{E,F,F}
    objective::F
    smoothing::R
end

function NonlinearLeastSquaresObjective(
    f,
    jacobian,
    range_dimension;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    jacobian_tangent_basis::AbstractBasis=DefaultOrthonormalBasis(),
    jacobian_type::AbstractVectorialType=CoordinateVectorialType(jacobian_tangent_basis),
    function_type::AbstractVectorialType=FunctionVectorialType(),
    kwargs...,
)
    vgf = VectorGradientFunction(
        f,
        jacobian,
        range_dimension;
        evaluation=evaluation,
        jacobian_type=jacobian_type,
        function_type=function_type,
    )
    return NonlinearLeastSquaresObjective(vgf; kwargs...)
end

function NonlinearLeastSquaresObjective(
    vgf::F; smoothing=:Identity
) where {F<:AbstractVectorGradientFunction}
    s = smoothing_factory(smoothing)
    return NonlinearLeastSquaresObjective(vgf, s)
end

# Cost
# (a) with for a single smoothing function
function get_cost(
    M::AbstractManifold,
    nlso::NonlinearLeastSquaresObjective{
        E,<:AbstractVectorFunction{E,<:ComponentVectorialType},H
    },
    p;
    vector_space=Rn,
    kwargs...,
) where {E<:AbstractEvaluationType,H<:AbstractManifoldHessianObjective}
    v = 0.0
    for i in 1:length(nlso.objective)
        v += get_cost(
            vector_space(1), nlso.smoothing, abs(get_value(M, nlso.objective, p, i))^2
        )
    end
    v *= 0.5
    return v
end
function get_cost(
    M::AbstractManifold,
    nlso::NonlinearLeastSquaresObjective{
        E,<:AbstractVectorFunction{E,<:FunctionVectorialType},H
    },
    p;
    vector_space=Rn,
    value_cache=get_value(M, nlso.objective, p),
) where {E<:AbstractEvaluationType,H<:AbstractManifoldHessianObjective}
    return 0.5 * sum(
        get_cost(vector_space(1), nlso.smoothing, abs(value_cache[i])^2) for
        i in 1:length(value_cache)
    )
end
# (b) vectorial ρ
function get_cost(
    M::AbstractManifold,
    nlso::NonlinearLeastSquaresObjective{
        E,<:AbstractVectorFunction{E,<:ComponentVectorialType},<:AbstractVectorFunction
    },
    p;
    vector_space=Rn,
    kwargs...,
) where {E<:AbstractEvaluationType}
    v = 0.0
    for i in 1:length(nlso.objective)
        v += get_value(
            vector_space(1), nlso.smoothing, abs(get_value(M, nlso.objective, p, i))^2, i
        )
    end
    v *= 0.5
    return v
end
function get_cost(
    M::AbstractManifold,
    nlso::NonlinearLeastSquaresObjective{
        E,<:AbstractVectorFunction{E,<:FunctionVectorialType},<:AbstractVectorFunction
    },
    p;
    vector_space=Rn,
    value_cache=get_value(M, nlso.objective, p),
) where {E<:AbstractEvaluationType}
    return 0.5 * sum(
        get_value(vector_space(1), nlso.smoothing, abs(value_cache[i])^2, i) for
        i in 1:length(value_cache)
    )
end

function get_jacobian(
    M::AbstractManifold, nlso::NonlinearLeastSquaresObjective, p; kwargs...
)
    J = zeros(length(nlso.objective), manifold_dimension(M))
    get_jacobian!(M, J, nlso, p; kwargs...)
    return J
end
# Cases: (a) single smoothing function
function get_jacobian!(
    M::AbstractManifold,
    J,
    nlso::NonlinearLeastSquaresObjective{E,AHVF,<:AbstractManifoldGradientObjective},
    p;
    vector_space=Rn,
    value_cache=get_value(M, nlso.objective, p),
    kwargs...,
) where {E,AHVF}
    get_jacobian!(M, J, nlso.objective, p; kwargs...)
    for i in 1:length(nlso.objective) # s'(f_i(p)) * f_i'(p)
        J[i, :] .*= get_gradient(vector_space(1), nlso.smoothing, value_cache[i])
    end
    return J
end
# Cases: (b) vectorial smoothing function
function get_jacobian!(
    M::AbstractManifold,
    J,
    nlso::NonlinearLeastSquaresObjective{E,AHVF,<:AbstractVectorGradientFunction},
    p;
    vector_space=Rn,
    basis::AbstractBasis=get_basis(nlso.objective.jacobian_type),
    value_cache=get_value(M, nlso.objective, p),
) where {E,AHVF}
    get_jacobian!(M, J, nlso.objective, p; basis=basis)
    for i in 1:length(nlso.objective) # s_i'(f_i(p)) * f_i'(p)
        J[i, :] .*= get_gradient(vector_space(1), nlso.smoothing, value_cache[i], i)
    end
    return J
end

function get_gradient(
    M::AbstractManifold, nlso::NonlinearLeastSquaresObjective, p; kwargs...
)
    X = zero_vector(M, p)
    return get_gradient!(M, X, nlso, p; kwargs...)
end
function get_gradient!(
    M::AbstractManifold,
    X,
    nlso::NonlinearLeastSquaresObjective,
    p;
    basis=get_basis(nlso.objective.jacobian_type),
    jacobian_cache=get_jacobian(M, nlso, p; basis=basis),
    value_cache=get_residuals(M, nlso, p),
)
    return get_vector!(M, X, p, transpose(jacobian_cache) * value_cache, basis)
end

#
#
# --- Residuals
_doc_get_residuals_nlso = """
    get_residuals(M::AbstractManifold, nlso::NonlinearLeastSquaresObjective, p)
    get_residuals!(M::AbstractManifold, V, nlso::NonlinearLeastSquaresObjective, p)

Compute the vector of residuals ``f_i(p)``, ``i=1,…,n`` given the manifold `M`,
the [`NonlinearLeastSquaresObjective`](@ref) `nlso` and a current point ``p`` on `M`.

# Keyword arguments

* `vector_space=`[`Rn`](@ref)`: a vector space to use for evaluating the single
  smoothing functions ``s_i`` on.
* `value_cache=`[`get_value`](@ref)`(M, nlso.objective, p)`: a cache to provide the
  function evaltuation vector of the ``f_i(p)``, ``i=1,…,n`` in.
"""

@doc "$(_doc_get_residuals_nlso)"
get_residuals(M::AbstractManifold, nlso::NonlinearLeastSquaresObjective, p; kwargs...)

# (a) with for a single smoothing function
function get_residuals(
    M::AbstractManifold, nlso::NonlinearLeastSquaresObjective, p; kwargs...
)
    V = zeros(length(nlso.objective))
    return get_residuals!(M, V, nlso, p; kwargs...)
end

@doc "$(_doc_get_residuals_nlso)"
get_residuals!(M::AbstractManifold, V, nlso::NonlinearLeastSquaresObjective, p; kwargs...)

function get_residuals!(
    M::AbstractManifold,
    V,
    nlso::NonlinearLeastSquaresObjective{
        E,<:AbstractVectorFunction{E,<:ComponentVectorialType},H
    },
    p;
    kwargs...,
) where {E<:AbstractEvaluationType,H<:AbstractManifoldHessianObjective}
    for i in 1:length(nlso.objective)
        V[i] = get_value(M, nlso.objective, p, i)
    end
    return V
end
function get_residuals!(
    M::AbstractManifold,
    V,
    nlso::NonlinearLeastSquaresObjective{
        E,<:AbstractVectorFunction{E,<:FunctionVectorialType},H
    },
    p;
    value_cache=get_value(M, nlso.objective, p),
) where {E<:AbstractEvaluationType,H<:AbstractManifoldHessianObjective}
    for i in 1:length(value_cache)
        V[i] = value_cache[i]
    end
    return V
end
# (b) vectorial ρ
function get_residuals!(
    M::AbstractManifold,
    V,
    nlso::NonlinearLeastSquaresObjective{
        E,<:AbstractVectorFunction{E,<:ComponentVectorialType},<:AbstractVectorFunction
    },
    p;
    kwargs...,
) where {E<:AbstractEvaluationType}
    for i in 1:length(nlso.objective)
        V[i] = get_value(M, nlso.objective, p, i)
    end
    return V
end
function get_residuals!(
    M::AbstractManifold,
    V,
    nlso::NonlinearLeastSquaresObjective{
        E,<:AbstractVectorFunction{E,<:FunctionVectorialType},<:AbstractVectorFunction
    },
    p;
    value_cache=get_value(M, nlso.objective, p),
) where {E<:AbstractEvaluationType}
    for i in 1:length(value_cache)
        V[i] = value_cache[i]
    end
    return V
end

@doc """
    LevenbergMarquardtState{P,T} <: AbstractGradientSolverState

Describes a Gradient based descent algorithm, with

# Fields

A default value is given in brackets if a parameter can be left out in initialization.

$(_var(:Field, :p; add=[:as_Iterate]))
$(_var(:Field, :retraction_method))
* `residual_values`:      value of ``F`` calculated in the solver setup or the previous iteration
* `residual_values_temp`: value of ``F`` for the current proposal point
$(_var(:Field, :stopping_criterion, "stop"))
* `jacobian`:                 the current Jacobian of ``F``
* `gradient`:             the current gradient of ``F``
* `step_vector`:          the tangent vector at `x` that is used to move to the next point
* `last_stepsize`:        length of `step_vector`
* `η`:                    Scaling factor for the sufficient cost decrease threshold required
  to accept new proposal points. Allowed range: `0 < η < 1`.
* `damping_term`:         current value of the damping term
* `damping_term_min`:     initial (and also minimal) value of the damping term
* `β`:                    parameter by which the damping term is multiplied when the current
  new point is rejected
* `expect_zero_residual`: if true, the algorithm expects that the value of
  the residual (objective) at minimum is equal to 0.

# Constructor

    LevenbergMarquardtState(M, initial_residual_values, initial_jacobian; kwargs...)

Generate the Levenberg-Marquardt solver state.

# Keyword arguments

The following fields are keyword arguments

* `β=5.0`
* `damping_term_min=0.1`
* `η=0.2`,
* `expect_zero_residual=false`
* `initial_gradient=`$(_link(:zero_vector))
$(_var(:Keyword, :retraction_method))
$(_var(:Keyword, :stopping_criterion; default="[`StopAfterIteration`](@ref)`(200)`$(_sc(:Any))[`StopWhenGradientNormLess`](@ref)`(1e-12)`$(_sc(:Any))[`StopWhenStepsizeLess`](@ref)`(1e-12)`"))

# See also

[`gradient_descent`](@ref), [`LevenbergMarquardt`](@ref)
"""
mutable struct LevenbergMarquardtState{
    P,
    TStop<:StoppingCriterion,
    TRTM<:AbstractRetractionMethod,
    Tresidual_values,
    TJac,
    TGrad,
    Tparams<:Real,
} <: AbstractGradientSolverState
    p::P
    stop::TStop
    retraction_method::TRTM
    residual_values::Tresidual_values
    candidate_residual_values::Tresidual_values
    jacobian::TJac
    X::TGrad
    step_vector::TGrad
    last_stepsize::Tparams
    η::Tparams
    damping_term::Tparams
    damping_term_min::Tparams
    β::Tparams
    expect_zero_residual::Bool
    last_step_successful::Bool
    function LevenbergMarquardtState(
        M::AbstractManifold,
        initial_residual_values::Tresidual_values,
        initial_jacobian::TJac;
        p::P=rand(M),
        X::TGrad=zero_vector(M, p),
        stopping_criterion::StoppingCriterion=StopAfterIteration(200) |
                                              StopWhenGradientNormLess(1e-12) |
                                              StopWhenStepsizeLess(1e-12),
        retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
        η::Real=0.2,
        damping_term_min::Real=0.1,
        β::Real=5.0,
        expect_zero_residual::Bool=false,
    ) where {P,Tresidual_values,TJac,TGrad}
        if η <= 0 || η >= 1
            throw(ArgumentError("Value of η must be strictly between 0 and 1, received $η"))
        end
        if damping_term_min <= 0
            throw(
                ArgumentError(
                    "Value of damping_term_min must be strictly above 0, received $damping_term_min",
                ),
            )
        end
        if β <= 1
            throw(ArgumentError("Value of β must be strictly above 1, received $β"))
        end
        Tparams = promote_type(typeof(η), typeof(damping_term_min), typeof(β))
        return new{
            P,
            typeof(stopping_criterion),
            typeof(retraction_method),
            Tresidual_values,
            TJac,
            TGrad,
            Tparams,
        }(
            p,
            stopping_criterion,
            retraction_method,
            initial_residual_values,
            copy(initial_residual_values),
            initial_jacobian,
            X,
            allocate(M, X),
            zero(Tparams),
            η,
            damping_term_min,
            damping_term_min,
            β,
            expect_zero_residual,
            true,
        )
    end
end

function smoothing_factory end
"""
    smoothing_factory(s::Symbol=:Identity)
    smoothing_factory((s,α)::Tuple{Union{Symbol, ManifoldHessianObjective,<:Real})
    smoothing_factory((s,k)::Tuple{Union{Symbol, ManifoldHessianObjective,<:Int})
    smoothing_factory(o::ManifoldHessianObjective)
    smoothing_factory(o::VectorHessianFunction)
    smoothing_factory(s...)

Create a smoothing function from a symbol `s`.

For a single symbol `s`, the corresponding smoothing function is returned as a [`ManifoldHessianObjective`](@ref)
If the argument already is a [`ManifoldHessianObjective`](@ref), it is returned unchanged.

For a tuple `(s, α)`, the smoothing function is scaled by `α` as ``s_α(x) = α s$(_tex(:bigl))($(_tex(:frac, "x", "α^2"))$(_tex(:bigr)))``,
which yields ``s_α'(x) = s'$(_tex(:bigl))($(_tex(:frac, "x", "α^2"))$(_tex(:bigr)))`` and ``s_α''(x)[X] = $(_tex(:bigl))($(_tex(:frac, "1", "α^2"))$(_tex(:bigr)))s''$(_tex(:bigl))($(_tex(:frac, "x", "α^2"))$(_tex(:bigr)))[X]``.

For a tuple `(s, k)`, a [`VectorHessianFunction`](@ref) is returned, where every component is the smooting function indicated by `s`
If the argument already is a [`VectorHessianFunction`](@ref), it is returned unchanged.

Finally passing a sequence of these different cases as `s...`, a [`VectorHessianFunction`](@ref) is returned,
containing all smoothing functions with their repetitions mentioned as a vectorial function.

# Examples

* `smoothing_factory(:Identity)`: returns the identity function as a single smoothing function
* `smoothing_factory(:Identity, 2)`: returns a [`VectorHessianFunction`](@ref) with two identity functions
* `smoothing_factory(:Identity, 2.0)`: returns a [`VectorHessianFunction`](@ref) with the identity ``s(x) = x`` function scaled to ``s_2(x)`` as described above
* `smoothing_factory(mho, 0.5)`: returns a [`ManifoldHessianObjective`](@ref) with the scaled variant of the given `mho`, for example the one returned in the first example
* `smoothing_factory( ( (:Identity, 2), (:Huber, 3) ))`: returns a [`VectorHessianFunction`](@ref) with 5 components, the first 2 `:Identity` the last 3 `:Huber`

# Currently available smoothing functions

| `Symbol` | ``s(x)`` | ``s'(x)`` | ``s''(x)`` | Comment |
|:------ |:-----:|:-----:|:-----:|:--------- |
| `:Arctan` | ``$(_tex(:rm, "arctan"))(x)`` | ``$(_tex(:displaystyle))$(_tex(:frac, "1", "x^2+1"))`` | ``-$(_tex(:displaystyle))$(_tex(:frac, "2x", "(x^2+1)^2"))`` | |
| `:Cauchy` | ``$(_tex(:log))(1+x)`` | ``$(_tex(:displaystyle))$(_tex(:frac, "1", "1+x"))`` | ``-$(_tex(:displaystyle))$(_tex(:frac, "1", "(1+x)^2"))`` | |
| `:Huber` | ``$(_tex(:cases, "x, &  x ≤ 1,", "2$(_tex(:sqrt, "x")) - 1, &  x > 1."))`` | ``$(_tex(:cases, "1, &  x ≤ 1,", "$(_tex(:frac, "1", _tex(:sqrt, "x"))), &  x > 1."))`` | ``$(_tex(:cases, "0, &  x ≤ 1,", "-$(_tex(:frac, "1", "x^{3/2}")), &  x > 1."))`` | |
| `:Identity` | ``x`` | ``1`` | ``0`` | default (no smoothing) |
| `:SoftL1` | ``2$(_tex(:sqrt, "1+x")) - 1`` | ``$(_tex(:displaystyle))$(_tex(:frac, "1", _tex(:sqrt, "1+x")))`` | ``-$(_tex(:displaystyle))$(_tex(:frac, "1", "2(1+x)^{3/2}"))`` | |
| `:Tukey` | ``$(_tex(:cases, "$(_tex(:frac, "1", "3"))-$(_tex(:frac, "1", "3"))(1-x)^3, &  x ≤ 1,", "$(_tex(:frac, "1", "3")), &  x > 1."))`` | ``$(_tex(:cases, "(1-x)^2, &  x ≤ 1,", "0, &  x > 1."))`` | ``$(_tex(:cases, "x-2, &  x ≤ 1,", "0, &  x > 1."))`` | |

Note that in the implementation the second derivative follows the general scheme of hessians
and actually implements s''(x)[X] = s''(x)X``.
"""
smoothing_factory(s)

smoothing_factory() = smoothing_factory(:Identity)
smoothing_factory(o::ManifoldHessianObjective) = o
smoothing_factory(o::VectorHessianFunction) = o
function smoothing_factory(s::Symbol)
    return ManifoldHessianObjective(_smoothing_factory(Val(s))...)
end
function _smoothing_factory(s::Symbol)
    return _smoothing_factory(Val(s))
end
function smoothing_factory((s, α)::Tuple{Symbol,<:Real})
    s, s_p, s_pp = _smoothing_factory(s, α)
    return ManifoldHessianObjective(s, s_p, s_pp)
end
function smoothing_factory((o, α)::Tuple{ManifoldHessianObjective,<:Real})
    s, s_p, s_pp = _smoothing_factory(o, α)
    return ManifoldHessianObjective(s, s_p, s_pp)
end
function smoothing_factory((s, k)::Tuple{Symbol,<:Int})
    s, s_p, s_pp = _smoothing_factory(s, k)
    return VectorHessianFunction(
        s,
        s_p,
        s_pp,
        k;
        function_type=ComponentVectorialType(),
        jacobian_type=ComponentVectorialType(),
        hessian_type=ComponentVectorialType(),
    )
end
function smoothing_factory((o, k)::Tuple{ManifoldHessianObjective,<:Int})
    s, s_p, s_pp = _smoothing_factory(o, k)
    return VectorHessianFunction(
        s,
        s_p,
        s_pp,
        k;
        function_type=ComponentVectorialType(),
        jacobian_type=ComponentVectorialType(),
        hessian_type=ComponentVectorialType(),
    )
end
function smoothing_factory(S...)
    s = Function[]
    s_p = Function[]
    s_pp = Function[]
    # collect all functions including their copies into a large vector
    for t in S
        _s, _s_p, _s_pp = t isa Tuple ? _smoothing_factory(t...) : _smoothing_factory(t)
        if _s isa Array
            push!(s, _s...)
            push!(s_p, _s_p...)
            push!(s_pp, _s_pp...)
        else
            push!(s, _s)
            push!(s_p, _s_p)
            push!(s_pp, _s_pp)
        end
    end
    k = length(s)
    return VectorHessianFunction(
        s,
        s_p,
        s_pp,
        k;
        function_type=ComponentVectorialType(),
        jacobian_type=ComponentVectorialType(),
        hessian_type=ComponentVectorialType(),
    )
end
# Inner functions that split any smoothing function into its  ρ, ρ' and  ρ'' parts
function _smoothing_factory(o::ManifoldHessianObjective)
    return (E, x) -> get_cost(E, o, x),
    (E, x) -> get_gradient(E, o, x),
    (E, x, X) -> get_hessian(E, o, x, X)
end
function _smoothing_factory(o::VectorHessianFunction)
    return o.value!!, o.jacobian!!, o.hessians!!
end

function _smoothing_factory(o::ManifoldHessianObjective, α::Real)
    return (E, x) -> α^2 * get_cost(E, o, x / α^2),
    (E, x) -> get_gradient(E, o, x / α^2),
    (E, x, X) -> get_hessian(E, o, x / α^2, X) / α^2
end
function _smoothing_factory(s::Symbol, α::Real)
    s, s_p, s_pp = _smoothing_factory(Val(s))
    return (E, x) -> α^2 * s(E, x / α^2),
    (E, x) -> s_p(E, x / α^2),
    (E, x, X) -> s_pp(E, x / α^2, X) / α^2
end
function _smoothing_factory(o::ManifoldHessianObjective, k::Int)
    return fill((E, x) -> get_cost(E, o, x), k),
    fill((E, x) -> get_gradient(E, o, x), k),
    fill((E, x, X) -> get_hessian(E, o, x, X), k)
end
function _smoothing_factory(s::Symbol, k::Int)
    s, s_p, s_pp = _smoothing_factory(Val(s))
    return fill(s, k), fill(s_p, k), fill(s_pp, k)
end
# Library
function _smoothing_factory(::Val{:Arctan})
    return (E, x) -> arctan(x),
    (E, x) -> x <= 1 / (x^2 + 1),
    (E, x, X) -> 2 * x / (x^2 + 1)^2
end
function _smoothing_factory(::Val{:Cauchy})
    return (E, x) -> log(1 + x), (E, x) -> x <= 1 / (1 + x), (E, x, X) -> -1 / (1 + x)^2
end
function _smoothing_factory(::Val{:Huber})
    return (E, x) -> x <= 1 ? x : 2 * sqrt(x) - 1,
    (E, x) -> x <= 1 ? 1 : 1 / sqrt(x),
    (E, x, X) -> (x <= 1 ? 0 : -1 / (2x^(3 / 2))) * X
end
# Default
function _smoothing_factory(::Val{:Identity})
    return (E, x) -> x, (E, x) -> one(x), (E, x, X) -> zero(X)
end
function _smoothing_factory(::Val{:SoftL1})
    return (E, x) -> 2 * sqrt(1 + x) - 1,
    (E, x) -> x <= 1 / sqrt(1 + x),
    (E, x, X) -> -1 / (2 * (1 + x)^(3 / 2))
end
function _smoothing_factory(::Val{:Tukey})
    return (E, x) -> x <= 1 ? 1 / 3 * (1 - (1 - x)^3) : 1 / 3,
    (E, x) -> x <= 1 ? (1 - s)^2 : 0,
    (E, x, X) -> (x <= 1 ? x - 2 : 0) * X
end

function show(io::IO, lms::LevenbergMarquardtState)
    i = get_count(lms, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(lms.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Levenberg Marquardt Algorithm
    $Iter
    ## Parameters
    * β: $(lms.β)
    * damping term_ $(lms.damping_term) (min: $(lms.damping_term_min))
    * η: $(lms.η)
    * expect zero residual: $(lms.expect_zero_residual)
    * retraction method: $(lms.retraction_method)

    ## Stopping criterion

    $(status_summary(lms.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end
