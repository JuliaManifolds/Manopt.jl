@doc raw"""
    LevenbergMarquardt(M, F, jacF, x, num_components=-1)

Solve an optimization problem of the form

```math
\operatorname{arg\,min}_{p ∈ \mathcal M} \frac{1}{2} \lVert F(p) \rVert^2,
```

where ``F\colon\mathcal M \to ℝ^d`` is a continuously differentiable function,
using the Riemannian Levenberg-Marquardt algorithm [^Peeters1993].
The implementation follows Algorithm 1[^Adachi2022].

# Input
* `M` – a manifold ``\mathcal M``
* `F` – a cost function ``F: \mathcal M→ℝ^d``
* `jacF` – the Jacobian of ``F``. `jacF` is supposed to accept a keyword argument
  `basis_domain` which specifies basis of the tangent space at a given point in which the
  Jacobian is to be calculated. By default it should be the `DefaultOrthonormalBasis`.
* `x` – an initial value ``x ∈ \mathcal M``
* `num_components` -- length of the vector returned by the cost function (`d`).
  By default its value is -1 which means that it will be determined automatically by
  calling `F` one additional time. Only possible when `evaluation` is `AllocatingEvaluation`,
  for mutating evaluation this must be explicitly specified.

# Optional
* `evaluation` – ([`AllocatingEvaluation`](@ref)) specify whether the gradient works by allocation (default) form `gradF(M, x)`
  or [`InplaceEvaluation`](@ref) in place, i.e. is of the form `gradF!(M, X, x)`.
* `retraction_method` – (`default_retraction_method(M)`) a `retraction(M,x,ξ)` to use.
* `stopping_criterion` – ([`StopWhenAny`](@ref)`(`[`StopAfterIteration`](@ref)`(200), `[`StopWhenGradientNormLess`](@ref)`(1e-12))`)
  a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop.
...
and the ones that are passed to [`decorate_options`](@ref) for decorators.

# Output

the obtained (approximate) minimizer ``x^*``, see [`get_solver_return`](@ref) for details

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
function LevenbergMarquardt(
    M::AbstractManifold, F::TF, jacF::TDF, p, num_components::Int=-1; kwargs...
) where {TF,TDF}
    q = copy(M, p)
    return LevenbergMarquardt!(M, F, jacF, q, num_components; kwargs...)
end

@doc raw"""
    LevenbergMarquardt!(M, F, jacF, x, num_components; kwargs...)

For more options see [`LevenbergMarquardt`](@ref).
"""
function LevenbergMarquardt!(
    M::AbstractManifold,
    F::TF,
    jacF::TDF,
    p,
    num_components::Int=-1;
    retraction_method::AbstractRetractionMethod=default_retraction_method(M),
    stopping_criterion::StoppingCriterion=StopAfterIteration(200) |
                                          StopWhenGradientNormLess(1e-12) |
                                          StopWhenStepsizeLess(1e-12),
    debug=[DebugWarnIfCostIncreases()],
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    expect_zero_residual::Bool=false,
    jacB::AbstractBasis=DefaultOrthonormalBasis(),
    kwargs..., #collect rest
) where {TF,TDF}
    if num_components == -1
        if evaluation === AllocatingEvaluation()
            num_components = length(F(M, p))
        else
            throw(
                ArgumentError(
                    "For mutating evaluation num_components needs to be explicitly specified",
                ),
            )
        end
    end
    nlso = NonlinearLeastSquaresObjective(
        F, jacF, num_components; evaluation=evaluation, jacB=jacB
    )
    nlsp = DefaultManoptProblem(M, nlso)
    lms = LevenbergMarquardtState(
        M,
        p,
        similar(p, num_components),
        similar(p, num_components, manifold_dimension(M));
        stopping_criterion=stopping_criterion,
        retraction_method=retraction_method,
        expect_zero_residual=expect_zero_residual,
    )
    lms = decorate_state(lms; debug=debug, kwargs...)
    return get_solver_return(solve!(nlsp, lms))
end
#
# Solver functions
#
function initialize_solver!(
    dmp::DefaultManoptProblem{mT,<:NonlinearLeastSquaresObjective{AllocatingEvaluation}},
    lms::LevenbergMarquardtState,
) where {mT<:AbstractManifold}
    M = get_manifold(dmp)
    lms.residual_values = get_objective(dmp).F(M, lms.p)
    lms.X = get_gradient(dmp, lms.p)
    return lms
end
function initialize_solver!(
    dmp::DefaultManoptProblem{mT,<:NonlinearLeastSquaresObjective{InplaceEvaluation}},
    lms::LevenbergMarquardtState,
) where {mT<:AbstractManifold}
    get_objective(dmp).F(get_manifold(M), lms.residual_values, lms.p)
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
    return copyto!(residuals, get_objective(dmp).F(get_manifold(dmp), p))
end
function get_residuals!(
    dmp::DefaultManoptProblem{mT,<:NonlinearLeastSquaresObjective{InplaceEvaluation}},
    residuals,
    p,
) where {mT}
    return get_objective(dmp).F(get_manifold(dmp), residuals, p)
end

function step_solver!(
    dmp::DefaultManoptProblem{mT,<:NonlinearLeastSquaresObjective},
    lms::LevenbergMarquardtState,
    i::Integer,
) where {mT<:AbstractManifold}
    # o.residual_values is either initialized by initialize_solver! or taken from the previous iteraion
    M = get_manifold(dmp)
    nlso = get_objective(dmp)
    basis_ox = _maybe_get_basis(M, lms.p, nlso.jacB)
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
