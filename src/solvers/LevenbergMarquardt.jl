@doc raw"""
    LevenbergMarquardt(M, f, jacobian_f, p, num_components=-1)

Solve an optimization problem of the form

```math
\operatorname{arg\,min}_{p ∈ \mathcal M} \frac{1}{2} \lVert f(p) \rVert^2,
```

where ``f\colon\mathcal M \to ℝ^d`` is a continuously differentiable function,
using the Riemannian Levenberg-Marquardt algorithm [^Peeters1993].
The implementation follows Algorithm 1[^Adachi2022].

# Input
* `M` – a manifold ``\mathcal M``
* `f` – a cost function ``F: \mathcal M→ℝ^d``
* `jacobian_f` – the Jacobian of ``f``. The Jacobian `jacF` is supposed to accept a keyword argument
  `basis_domain` which specifies basis of the tangent space at a given point in which the
  Jacobian is to be calculated. By default it should be the `DefaultOrthonormalBasis`.
* `p` – an initial value ``p ∈ \mathcal M``
* `num_components` -- length of the vector returned by the cost function (`d`).
  By default its value is -1 which means that it will be determined automatically by
  calling `F` one additional time. Only possible when `evaluation` is `AllocatingEvaluation`,
  for mutating evaluation this must be explicitly specified.

These can also be passed as a [`NonlinearLeastSquaresObjective`](@ref),
then the keyword `jacobian_tangent_basis` below is ignored

# Optional

* `evaluation` – ([`AllocatingEvaluation`](@ref)) specify whether the gradient works by allocation (default) form `gradF(M, x)`
  or [`InplaceEvaluation`](@ref) in place, i.e. is of the form `gradF!(M, X, x)`.
* `retraction_method` – (`default_retraction_method(M, typeof(p))`) a `retraction(M,x,ξ)` to use.
* `stopping_criterion` – ([`StopWhenAny`](@ref)`(`[`StopAfterIteration`](@ref)`(200), `[`StopWhenGradientNormLess`](@ref)`(1e-12))`)
  a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop.
* `expect_zero_residual` – (`false`) whether or not the algorithm might expect that the value of
  residual (objective) at mimimum is equal to 0.

All other keyword arguments are passed to [`decorate_state!`](@ref) for decorators or
[`decorate_objective!`](@ref), respectively.
If you provide the [`ManifoldGradientObjective`](@ref) directly, these decorations can still be specified


# Output

the obtained (approximate) minimizer ``p^*``, see [`get_solver_return`](@ref) for details

# References

[^Adachi2022]:
    > S. Adachi, T. Okuno, and A. Takeda, “Riemannian Levenberg-Marquardt Method with Global
    > and Local Convergence Properties.” arXiv, Oct. 01, 2022.
    > arXiv: [2210.00253](https://doi.org/10.48550/arXiv.2210.00253).
[^Peeters1993]:
    > R. L. M. Peeters, “On a Riemannian version of the Levenberg-Marquardt algorithm,”
    > VU University Amsterdam, Faculty of Economics, Business Administration and Econometrics,
    > Serie Research Memoranda 0011, 1993.
    > link: [https://econpapers.repec.org/paper/vuawpaper/1993-11.htm](https://econpapers.repec.org/paper/vuawpaper/1993-11.htm).
"""
LevenbergMarquardt(M::AbstractManifold, args...; kwargs...)
function LevenbergMarquardt(
    M::AbstractManifold,
    f,
    jacobian_f,
    num_components::Int=-1;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
)
    return LevenbergMarquardt(
        M, f, jacobian_f, rand(M), num_components; evaluation=evaluation, kwargs...
    )
end
function LevenbergMarquardt(
    M::AbstractManifold,
    f,
    jacobian_f,
    p,
    num_components::Int=-1;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    jacB=nothing,
    jacobian_tangent_basis::AbstractBasis=if isnothing(jacB)
        DefaultOrthonormalBasis()
    else
        jacB
    end,
    kwargs...,
)
    !isnothing(jacB) &&
        (@warn "The keyword `jacB` is deprecated, use `jacobian_tangent_basis` instead.")
    if num_components == -1
        if evaluation === AllocatingEvaluation()
            num_components = length(f(M, p))
        else
            throw(
                ArgumentError(
                    "For mutating evaluation num_components needs to be explicitly specified",
                ),
            )
        end
    end
    nlso = NonlinearLeastSquaresObjective(
        f,
        jacobian_f,
        num_components;
        evaluation=evaluation,
        jacobian_tangent_basis=jacobian_tangent_basis,
    )
    return LevenbergMarquardt(M, nlso, p; evaluation=evaluation, kwargs...)
end
function LevenbergMarquardt(
    M::AbstractManifold, nlso::O, p; kwargs...
) where {O<:Union{NonlinearLeastSquaresObjective,AbstractDecoratedManifoldObjective}}
    q = copy(M, p)
    return LevenbergMarquardt!(M, nlso, q; kwargs...)
end

@doc raw"""
    LevenbergMarquardt!(M, f, jacobian_f, p, num_components=-1; kwargs...)

For more options see [`LevenbergMarquardt`](@ref).
"""
LevenbergMarquardt!(M::AbstractManifold, args...; kwargs...)
function LevenbergMarquardt!(
    M::AbstractManifold,
    f,
    jacobian_f,
    p,
    num_components::Int=-1;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    jacB=nothing,
    jacobian_tangent_basis::AbstractBasis=if isnothing(jacB)
        DefaultOrthonormalBasis()
    else
        jacB
    end,
    kwargs...,
)
    !isnothing(jacB) &&
        (@warn "The keyword `jacB` is deprecated, use `jacobian_tangent_basis` instead.")
    if num_components == -1
        if evaluation === AllocatingEvaluation()
            num_components = length(f(M, p))
        else
            throw(
                ArgumentError(
                    "For mutating evaluation num_components needs to be explicitly specified",
                ),
            )
        end
    end
    nlso = NonlinearLeastSquaresObjective(
        f,
        jacobian_f,
        num_components;
        evaluation=evaluation,
        jacobian_tangent_basis=jacobian_tangent_basis,
    )
    return LevenbergMarquardt!(M, nlso, p; evaluation=evaluation, kwargs...)
end
function LevenbergMarquardt!(
    M::AbstractManifold,
    nlso::O,
    p;
    retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
    stopping_criterion::StoppingCriterion=StopAfterIteration(200) |
                                          StopWhenGradientNormLess(1e-12) |
                                          StopWhenStepsizeLess(1e-12),
    debug=[DebugWarnIfCostIncreases()],
    expect_zero_residual::Bool=false,
    initial_residual_values=similar(p, get_objective(nlso).num_components),
    initial_jacF=similar(p, get_objective(nlso).num_components, manifold_dimension(M)),
    kwargs..., #collect rest
) where {O<:Union{NonlinearLeastSquaresObjective,AbstractDecoratedManifoldObjective}}
    i_nlso = get_objective(nlso) # undeecorate – for safety
    dnlso = decorate_objective!(M, nlso; kwargs...)
    nlsp = DefaultManoptProblem(M, dnlso)
    lms = LevenbergMarquardtState(
        M,
        p,
        initial_residual_values,
        initial_jacF;
        stopping_criterion=stopping_criterion,
        retraction_method=retraction_method,
        expect_zero_residual=expect_zero_residual,
    )
    dlms = decorate_state!(lms; debug=debug, kwargs...)
    solve!(nlsp, dlms)
    return get_solver_return(get_objective(nlsp), dlms)
end
#
# Solver functions
#
function initialize_solver!(
    dmp::DefaultManoptProblem{mT,<:NonlinearLeastSquaresObjective{AllocatingEvaluation}},
    lms::LevenbergMarquardtState,
) where {mT<:AbstractManifold}
    M = get_manifold(dmp)
    lms.residual_values = get_objective(dmp).f(M, lms.p)
    lms.X = get_gradient(dmp, lms.p)
    return lms
end
function initialize_solver!(
    dmp::DefaultManoptProblem{mT,<:NonlinearLeastSquaresObjective{InplaceEvaluation}},
    lms::LevenbergMarquardtState,
) where {mT<:AbstractManifold}
    get_objective(dmp).f(get_manifold(dmp), lms.residual_values, lms.p)
    lms.X = get_gradient(dmp, lms.p)
    return lms
end

function _maybe_get_basis(M::AbstractManifold, p, B::AbstractBasis)
    if requires_caching(B)
        return get_basis(M, p, B)
    else
        return B
    end
end

function get_jacobian!(
    dmp::DefaultManoptProblem{mT,<:NonlinearLeastSquaresObjective{AllocatingEvaluation}},
    jacF,
    p,
    basis_domain::AbstractBasis,
) where {mT}
    nlso = get_objective(dmp)
    return copyto!(jacF, nlso.jacobian!!(get_manifold(dmp), p; basis_domain=basis_domain))
end
function get_jacobian!(
    dmp::DefaultManoptProblem{mT,<:NonlinearLeastSquaresObjective{InplaceEvaluation}},
    jacF,
    p,
    basis_domain::AbstractBasis,
) where {mT}
    nlso = get_objective(dmp)
    return nlso.jacobian!!(get_manifold(dmp), jacF, p; basis_domain=basis_domain)
end

function get_residuals!(
    dmp::DefaultManoptProblem{mT,<:NonlinearLeastSquaresObjective{AllocatingEvaluation}},
    residuals,
    p,
) where {mT}
    return copyto!(residuals, get_objective(dmp).f(get_manifold(dmp), p))
end
function get_residuals!(
    dmp::DefaultManoptProblem{mT,<:NonlinearLeastSquaresObjective{InplaceEvaluation}},
    residuals,
    p,
) where {mT}
    return get_objective(dmp).f(get_manifold(dmp), residuals, p)
end

function step_solver!(
    dmp::DefaultManoptProblem{mT,<:NonlinearLeastSquaresObjective},
    lms::LevenbergMarquardtState,
    i::Integer,
) where {mT<:AbstractManifold}
    # o.residual_values is either initialized by initialize_solver! or taken from the previous iteraion
    M = get_manifold(dmp)
    nlso = get_objective(dmp)
    basis_ox = _maybe_get_basis(M, lms.p, nlso.jacobian_tangent_basis)
    get_jacobian!(dmp, lms.jacF, lms.p, basis_ox)
    λk = lms.damping_term * norm(lms.residual_values)

    JJ = transpose(lms.jacF) * lms.jacF + λk * I
    # `cholesky` is technically not necessary but it's the fastest method to solve the
    # problem because JJ is symmetric positive definite
    grad_f_c = transpose(lms.jacF) * lms.residual_values
    sk = cholesky(JJ) \ -grad_f_c
    get_vector!(M, lms.X, lms.p, grad_f_c, basis_ox)

    get_vector!(M, lms.step_vector, lms.p, sk, basis_ox)
    lms.last_stepsize = norm(M, lms.p, lms.step_vector)
    temp_x = retract(M, lms.p, lms.step_vector, lms.retraction_method)

    normFk2 = norm(lms.residual_values)^2
    get_residuals!(dmp, lms.candidate_residual_values, temp_x)

    ρk =
        2 * (normFk2 - norm(lms.candidate_residual_values)^2) / (
            -2 * inner(M, lms.p, lms.X, lms.step_vector) - norm(lms.jacF * sk)^2 -
            λk * norm(sk)
        )
    if ρk >= lms.η
        copyto!(M, lms.p, temp_x)
        copyto!(lms.residual_values, lms.candidate_residual_values)
        if lms.expect_zero_residual
            lms.damping_term = max(lms.damping_term_min, lms.damping_term / lms.β)
        end
    else
        lms.damping_term *= lms.β
    end
    return lms
end

function _get_last_stepsize(
    ::AbstractManoptProblem, lms::LevenbergMarquardtState, ::Val{false}, vars...
)
    return lms.last_stepsize
end
