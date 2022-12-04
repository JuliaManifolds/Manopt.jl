@doc raw"""
    Frank_Wolfe_method(M, F, grad_F, p)

Perform the Frank-Wolfe algorithm to compute for ``\mathcal C \subset \mathcal M``

```math
    \operatorname*{arg\,min}_{p∈\mathcal C} f(p)
```

Where the main step is a constrained optimisation is within the algorithm,
that is the sub problem (Oracle)

```math
    q_k = \operatorname{arg\,min}_{q \in C} ⟨\operatorname{grad} F(p_k), \log_{p_k}q⟩.
```

for every iterate ``p_k`` together with a stepsize ``s_k≤1``, by default ``s_k = \frac{2}{k+2}``.

The next iterate is then given by ``p_{k+1} = γ_{p_k,q_k}(s_k)``,
where by default ``γ`` is the shortest geodesic between the two points but can also be changed to
use a retraction and its inverse.

## Keyword Arguments

* `evaluation` ([`AllocatingEvaluation`](@ref)) whether `grad_F` is an inplace or allocating (default) funtion
* `initial_vector=zero_vector` (`zero_vectoir(M,p)`) how to initialize the inner gradient tangent vector
* `stopping_criterion` – [`StopAfterIteration`](@ref)`(500) | `[`StopWhenGradientNormLess`](@ref)`(1.0e-6)`
* `subtask` specify the oracle, can either be a closed form solution (in place function `oracle(M, q, p, X)`
  or a subsolver, e.g. (by default) a [`GradientProblem`](@ref) with [`GradientDescentState`](@ref)
  using the [`FrankWolfeCost`](@ref) and [`FrankWolfeGradient`](@ref).
* `stepsize` ([`DecreasingStepsize`](@ref)`(; length=2.0, shift=2)`
  a [`Stepsize`](@ref) to use; but it has to be always less than 1. The default is the one proposed by Frank & Wolfe:
  ``s_k = \frac{2}{k+2}``.

all further keywords are passed down to [`decorate_state`](@ref), e.g. `debug`.

# Output

the obtained (approximate) minimizer ``x^*``, see [`get_solver_return`](@ref) for details
"""
function Frank_Wolfe_method(M::AbstractManifold, F, grad_F, p; kwargs...)
    q = copy(M, p)
    return Frank_Wolfe_method!(M, F, grad_F, q; kwargs...)
end

@doc raw"""
    Frank_Wolfe_method!(M, F, grad_F, q; kwargs...)
"""
function Frank_Wolfe_method!(
    M::AbstractManifold,
    F,
    grad_F,
    p;
    initial_vector=zero_vector(M, p),
    subtask=(
        GradientProblem(
            M, FrankWolfeCost(p, initial_vector), FrankWolfeGradient(p, initial_vector)
        ),
        GradientDescentState(M, copy(M, p)),
    ),
    evaluation=AllocatingEvaluation(),
    stopping_criterion::TStop=StopAfterIteration(200) |
                              StopWhenGradientNormLess(1.0e-8) |
                              StopWhenChangeLess(1.0e-8),
    stepsize::TStep=DecreasingStepsize(; length=2.0, shift=2),
    kwargs..., #collect rest
) where {TStop<:StoppingCriterion,TStep<:Stepsize}
    P = GradientProblem(M, F, grad_F; evaluation=evaluation)
    O = FrankWolfeState(
        M,
        p,
        subtask;
        initial_vector=initial_vector,
        stepsize=stepsize,
        stopping_criterion=stopping_criterion,
        evaluation=evaluation,
    )
    O = decorate_state(O; kwargs...)
    return get_solver_return(solve!(P, O))
end
function initialize_solver!(P::AbstractManoptProblem, O::FrankWolfeState)
    get_gradient!(P, O.X, O.p)
    return O
end
function step_solver!(
    P::AbstractManoptProblem,
    O::FrankWolfeState{<:Tuple{<:AbstractManoptProblem,<:AbstractManoptSolverState}},
    i,
)
    # update gradient
    get_gradient!(P, O.X, O.p) # evaluate grad F(p), store the result in O.X
    # solve subtask
    solve!(O.subtask[1], O.subtask[2]) # call the subsolver
    q = get_solver_result(O.subtask[2])
    s = O.stepsize(P, O, i)
    # step along the geodesic
    retract!(
        P.M,
        O.p,
        O.p,
        s .* inverse_retract(P.M, O.p, q, O.inverse_retraction_method),
        O.retraction_method,
    )
    return O
end
#
# Variant II: subtask is a mutating function providing a closed form soltuion
#
function step_solver!(
    P::AbstractManoptProblem, O::FrankWolfeState{<:Tuple{S,<:InplaceEvaluation}}, i
) where {S}
    get_gradient!(P, O.X, O.p) # evaluate grad F in place for O.X
    q = copy(P.M, O.p)
    O.subtask[1](P.M, q, O.p, O.X) # evaluate the closed form solution and store the result in q
    s = O.stepsize(P, O, i)
    # step along the geodesic
    retract!(
        P.M,
        O.p,
        O.p,
        s .* inverse_retract(P.M, O.p, q, O.inverse_retraction_method),
        O.retraction_method,
    )
    return O
end
#
# Variant II: subtask is an allocating function providing a closed form soltuion
#
function step_solver!(
    P::AbstractManoptProblem, O::FrankWolfeState{<:Tuple{S,<:AllocatingEvaluation}}, i
) where {S}
    get_gradient!(P, O.X, O.p) # evaluate grad F in place for O.X
    q = O.subtask[1](P.M, O.p, O.X) # evaluate the closed form solution and store the result in O.p
    s = O.stepsize(P, O, i)
    # step along the geodesic
    retract!(
        P.M,
        O.p,
        O.p,
        s .* inverse_retract(P.M, O.p, q, O.inverse_retraction_method),
        O.retraction_method,
    )
    return O
end
