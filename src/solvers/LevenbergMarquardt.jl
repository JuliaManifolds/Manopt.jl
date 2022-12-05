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
  or [`MutatingEvaluation`](@ref) in place, i.e. is of the form `gradF!(M, X, x)`.
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
    M::AbstractManifold, F::TF, gradF::TDF, x, num_components::Int=-1; kwargs...
) where {TF,TDF}
    x_res = copy(M, x)
    return LevenbergMarquardt!(M, F, gradF, x_res, num_components; kwargs...)
end

@doc raw"""
    LevenbergMarquardt!(M, F, jacF, x, num_components; kwargs...)

For more options see [`LevenbergMarquardt`](@ref).
"""
function LevenbergMarquardt!(
    M::AbstractManifold,
    F::TF,
    jacF::TDF,
    x,
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
            num_components = length(F(M, x))
        else
            throw(
                ArgumentError(
                    "For mutating evaluation num_components needs to be explicitly specified",
                ),
            )
        end
    end
    p = NonlinearLeastSquaresProblem(
        M, F, jacF, num_components; evaluation=evaluation, jacB=jacB
    )
    o = LevenbergMarquardtState(
        M,
        x,
        similar(x, num_components),
        similar(x, num_components, manifold_dimension(M));
        stopping_criterion=stopping_criterion,
        retraction_method=retraction_method,
        expect_zero_residual=expect_zero_residual,
    )
    o = decorate_options(o; debug=debug, kwargs...)
    return get_solver_return(solve(p, o))
end
#
# Solver functions
#
function initialize_solver!(
    p::AbstractManoptProblem{mT,NonlinearLeastSquaresObjective{AllocatingEvaluation}},
    s::LevenbergMarquardtState,
) where {mT}
    M = get_manifold(p)
    s.residual_values = get_objective(p).F(M, s.x)
    s.gradient = get_gradient(p, s.x)
    return s
end
function initialize_solver!(
    p::AbstractManoptProblem{mT,NonlinearLeastSquaresObjective{MutatingEvaluation}},
    o::LevenbergMarquardtState,
) where {mT}
    get_objective(p).F(get_manifold(M), o.residual_values, o.x)
    o.gradient = get_gradient(p, o.x)
    return o
end

function _maybe_get_basis(M::AbstractManifold, p, B::AbstractBasis)
    if requires_caching(B)
        return get_basis(M, p, B)
    else
        return B
    end
end

function get_jacobian!(
    p::AbstractManoptProblem{M,NonlinearLeastSquaresObjective{AllocatingEvaluation}},
    jacF::FieldReference,
    x,
    basis_domain::AbstractBasis,
)
    return jacF[] = p.jacobian!!(get_manifold(p), x; basis_domain=basis_domain)
end
function get_jacobian!(
    p::AbstractManoptProblem{M,NonlinearLeastSquaresObjective{AllocatingEvaluation}},
    jacF,
    x,
    basis_domain::AbstractBasis,
)
    return p.jacobian!!(p.M, jacF, x; basis_domain=basis_domain)
end

function get_residuals!(
    p::AbstractManoptProblem{M,NonlinearLeastSquaresObjective{AllocatingEvaluation}},
    residuals::FieldReference,
    x,
)
    return residuals[] = get_objective(p).F(p.M, x)
end
function get_residuals!(
    p::AbstractManoptProblem{M,NonlinearLeastSquaresObjective{AllocatingEvaluation}},
    residuals,
    x,
)
    return get_objective(p).F(p.M, residuals, x)
end

function step_solver!(
    p::AbstractManoptProblem{M,NonlinearLeastSquaresObjective{Teval}},
    o::LevenbergMarquardtState,
    i::Integer,
) where {Teval<:AbstractEvaluationType}
    # o.residual_values is either initialized by initialize_solver! or taken from the previous iteraion

    M = get_manifold(p)
    basis_ox = _maybe_get_basis(M, o.x, p.jacB)
    get_jacobian!(p, (@access_field o.jacF), o.x, basis_ox)
    λk = o.damping_term * norm(o.residual_values)

    JJ = transpose(o.jacF) * o.jacF + λk * I
    # `cholesky` is technically not necessary but it's the fastest method to solve the
    # problem because JJ is symmetric positive definite
    grad_f_c = transpose(o.jacF) * o.residual_values
    sk = cholesky(JJ) \ -grad_f_c
    get_vector!(M, o.gradient, o.x, grad_f_c, basis_ox)

    get_vector!(M, o.step_vector, o.x, sk, basis_ox)
    o.last_stepsize = norm(M, o.x, o.step_vector)
    temp_x = retract(M, o.x, o.step_vector, o.retraction_method)

    normFk2 = norm(o.residual_values)^2
    get_residuals!(p, (@access_field o.candidate_residual_values), temp_x)

    ρk =
        2 * (normFk2 - norm(o.candidate_residual_values)^2) / (
            -2 * inner(M, o.x, o.gradient, o.step_vector) - norm(o.jacF * sk)^2 -
            λk * norm(sk)
        )
    if ρk >= o.η
        copyto!(M, o.x, temp_x)
        copyto!(o.residual_values, o.candidate_residual_values)
        if o.expect_zero_residual
            o.damping_term = max(o.damping_term_min, o.damping_term / o.β)
        end
    else
        o.damping_term *= o.β
    end
    return o
end

function _get_last_stepsize(::Problem, o::LevenbergMarquardtState, ::Val{false}, vars...)
    return o.last_stepsize
end
