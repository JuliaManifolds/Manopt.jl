@doc raw"""
    LevenbergMarquardt(M, F, jacF, x)

Solve an optimization problem of the form

```math
\operatorname{minimize}_{x ∈ \mathcal M} \frac{1}{2} \lVert F(x) \rVert^2,
```

where ``F: \mathcal M \to ℝ^d`` is a continuously differentiable function,
using the Riemannian Levenberg-Marquardt algorithm [^Peeters1993].
The implementation follows [^Adachi2022].

# Input
* `M` – a manifold ``\mathcal M``
* `F` – a cost function ``F: \mathcal M→ℝ^d``
* `jacF` – the Jacobian of ``F``
* `x` – an initial value ``x ∈ \mathcal M``
* `num_components` -- length of the vector returned by the cost function (`d`).

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
    > doi: [10.48550/arXiv.2210.00253](https://doi.org/10.48550/arXiv.2210.00253).
[^Peeters1993]:
    > R. L. M. Peeters, “On a Riemannian version of the Levenberg-Marquardt algorithm,”
    > VU University Amsterdam, Faculty of Economics, Business Administration and Econometrics,
    > Serie Research Memoranda 0011, 1993.
    > link: [https://econpapers.repec.org/paper/vuawpaper/1993-11.htm](https://econpapers.repec.org/paper/vuawpaper/1993-11.htm).
"""
function LevenbergMarquardt(
    M::AbstractManifold, F::TF, gradF::TDF, x, num_components::Int; kwargs...
) where {TF,TDF}
    x_res = copy(M, x)
    return LevenbergMarquardt!(M, F, gradF, x_res, num_components; kwargs...)
end

@doc raw"""
    LevenbergMarquardt!(M, F, jacF, x)


For more options see [`LevenbergMarquardt`](@ref).
"""
function LevenbergMarquardt!(
    M::AbstractManifold,
    F::TF,
    jacF::TDF,
    x,
    num_components::Int;
    retraction_method::AbstractRetractionMethod=default_retraction_method(M),
    stopping_criterion::StoppingCriterion=StopAfterIteration(200) |
                                          StopWhenGradientNormLess(1e-12) |
                                          StopWhenStepsizeLess(1e-12),
    debug=[DebugWarnIfCostIncreases()],
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    flagnz::Bool=true,
    kwargs..., #collect rest
) where {TF,TDF}
    p = NonlinearLeastSquaresProblem(M, F, jacF, num_components; evaluation=evaluation)
    o = LevenbergMarquardtOptions(
        M,
        x,
        similar(x, num_components), # TODO: rethink this?
        similar(x, num_components, manifold_dimension(M)); # TODO: rethink this?
        stopping_criterion=stopping_criterion,
        retraction_method=retraction_method,
        flagnz=flagnz,
    )
    o = decorate_options(o; debug=debug, kwargs...)
    return get_solver_return(solve(p, o))
end
#
# Solver functions
#
function initialize_solver!(
    p::NonlinearLeastSquaresProblem{AllocatingEvaluation}, o::LevenbergMarquardtOptions
)
    o.Fval = p.F(p.M, o.x)
    o.gradient = get_gradient(p, o.x)
    return o
end
function initialize_solver!(
    p::NonlinearLeastSquaresProblem{MutatingEvaluation}, o::LevenbergMarquardtOptions
)
    p.F(p.M, o.Fval, o.x)
    o.gradient = get_gradient(p, o.x)
    return o
end
function step_solver!(
    p::NonlinearLeastSquaresProblem{Teval}, o::LevenbergMarquardtOptions, iter::Integer
) where {Teval<:AbstractEvaluationType}
    # o.Fval is either initialized by initialize_solver! or taken from the previous iteraion
    if Teval === AllocatingEvaluation
        o.jacF = p.jacobian!!(p.M, o.x; B_dom=p.jacB)
    else
        p.jacobian!!(p.M, o.jacF, o.x; B_dom=p.jacB)
    end
    λk = o.damping_term * norm(o.Fval)

    JJ = transpose(o.jacF) * o.jacF + λk * I
    # `cholesky` is technically not necessary but it's the fastest method to solve the
    # problem because JJ is symmetric positive definite
    grad_f_c = transpose(o.jacF) * o.Fval
    sk = cholesky(JJ) \ -grad_f_c
    get_vector!(p.M, o.gradient, o.x, grad_f_c, p.jacB)

    get_vector!(p.M, o.step_vector, o.x, sk, p.jacB)
    o.last_stepsize = norm(p.M, o.x, o.step_vector)
    temp_x = retract(p.M, o.x, o.step_vector, o.retraction_method)

    normFk2 = norm(o.Fval)^2
    if Teval === AllocatingEvaluation
        o.Fval_temp = p.F(p.M, temp_x)
    else
        p.F(p.M, o.Fval_temp, temp_x)
    end

    ρk =
        2 * (normFk2 - norm(o.Fval_temp)^2) / (
            -2 * inner(p.M, o.x, o.gradient, o.step_vector) - norm(o.jacF * sk)^2 -
            λk * norm(sk)
        )
    if ρk >= o.η
        copyto!(p.M, o.x, temp_x)
        copyto!(o.Fval, o.Fval_temp)
        if !o.flagnz
            o.damping_term = max(o.damping_term_min, o.damping_term / o.β)
        end
    else
        o.damping_term *= o.β
    end
    return o
end

function _get_last_stepsize(::Problem, o::LevenbergMarquardtOptions, ::Val{false}, vars...)
    return o.last_stepsize
end
