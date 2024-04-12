@doc raw"""
    QuasiNewtonState <: AbstractManoptSolverState

These Quasi Newton [`AbstractManoptSolverState`](@ref) represent any quasi-Newton based method and can be
used with any update rule for the direction.

# Fields

* `p`                             the current iterate, a point on a manifold
* `X`                             the current gradient
* `sk`                            the current step
* `yk`                            the current gradient difference
* `direction_update`              an [`AbstractQuasiNewtonDirectionUpdate`](@ref) rule.
* `nondescent_direction_behavior` a `Symbol` to specify how to handle direction that are not descent ones.
* `retraction_method`             an `AbstractRetractionMethod`
* `stepsize`                      a [`Stepsize`](@ref)
* `stop`                          a [`StoppingCriterion`](@ref)

as well as for internal use

* `p_old`                      the last iterate
* `η`                          the current update direction
* `X_old`                      the last gradient
* `nondescent_direction_value` the value from the last inner product check for descent directions

# Constructor

    QuasiNewtonState(
        M::AbstractManifold,
        x;
        initial_vector=zero_vector(M,x),
        direction_update::D=QuasiNewtonLimitedMemoryDirectionUpdate(M, x, InverseBFGS(), 20;
            vector_transport_method=vector_transport_method,
        )
        stopping_criterion=StopAfterIteration(1000) | StopWhenGradientNormLess(1e-6),
        retraction_method::RM=default_retraction_method(M, typeof(p)),
        vector_transport_method::VTM=default_vector_transport_method(M, typeof(p)),
        stepsize=default_stepsize(M; QuasiNewtonState)
    )

# See also

[`quasi_Newton`](@ref)
"""
mutable struct QuasiNewtonState{
    P,
    T,
    D<:AbstractQuasiNewtonDirectionUpdate,
    SC<:StoppingCriterion,
    S<:Stepsize,
    RTR<:AbstractRetractionMethod,
    VT<:AbstractVectorTransportMethod,
    R,
} <: AbstractGradientSolverState
    p::P
    p_old::P
    η::T
    X::T
    sk::T
    yk::T
    direction_update::D
    retraction_method::RTR
    stepsize::S
    stop::SC
    X_old::T
    vector_transport_method::VT
    nondescent_direction_behavior::Symbol
    nondescent_direction_value::R
end
function QuasiNewtonState(
    M::AbstractManifold,
    p::P;
    initial_vector::T=zero_vector(M, p),
    vector_transport_method::VTM=default_vector_transport_method(M, typeof(p)),
    direction_update::D=QuasiNewtonLimitedMemoryDirectionUpdate(
        M, p, InverseBFGS(), 20; vector_transport_method=vector_transport_method
    ),
    stopping_criterion::SC=StopAfterIteration(1000) | StopWhenGradientNormLess(1e-6),
    retraction_method::RM=default_retraction_method(M, typeof(p)),
    stepsize::S=default_stepsize(
        M,
        QuasiNewtonState;
        retraction_method=retraction_method,
        vector_transport_method=vector_transport_method,
    ),
    nondescent_direction_behavior::Symbol=:step_towards_negative_gradient,
    kwargs..., # collect but ignore rest to be more tolerant
) where {
    P,
    T,
    D<:AbstractQuasiNewtonDirectionUpdate,
    SC<:StoppingCriterion,
    S<:Stepsize,
    RM<:AbstractRetractionMethod,
    VTM<:AbstractVectorTransportMethod,
}
    sk_init = zero_vector(M, p)
    return QuasiNewtonState{P,typeof(sk_init),D,SC,typeof(stepsize),RM,VTM,Float64}(
        p,
        copy(M, p),
        copy(M, p, initial_vector),
        initial_vector,
        sk_init,
        copy(M, sk_init),
        direction_update,
        retraction_method,
        stepsize,
        stopping_criterion,
        copy(M, p, initial_vector),
        vector_transport_method,
        nondescent_direction_behavior,
        1.0,
    )
end
function get_message(qns::QuasiNewtonState)
    # collect messages from
    # (1) direction update or the
    # (2) the step size and combine them
    # (3) the nondescent behaviour check message
    msg1 = get_message(qns.direction_update)
    msg2 = get_message(qns.stepsize)
    msg3 = ""
    if qns.nondescent_direction_value > 0
        msg3 = "Computed direction is not a descent direction. The inner product evaluated to $(qns.nondescent_direction_value)."
        if qns.nondescent_direction_behavior === :step_towards_negative_gradient
            msg3 = "$(msg3) Resetting to negative gradient."
        end
    end
    d = "$(msg1)"
    d = "$(length(d) > 0 ? "\n" : "")$(msg2)"
    d = "$(length(d) > 0 ? "\n" : "")$(msg3)"
    return d
end
function show(io::IO, qns::QuasiNewtonState)
    i = get_count(qns, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(qns.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Quasi Newton Method
    $Iter
    ## Parameters
    * direction update:        $(status_summary(qns.direction_update))
    * retraction method:       $(qns.retraction_method)
    * vector transport method: $(qns.vector_transport_method)

    ## Stepsize
    $(qns.stepsize)

    ## Stopping criterion

    $(status_summary(qns.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end
get_iterate(qns::QuasiNewtonState) = qns.p
function set_iterate!(qns::QuasiNewtonState, M, p)
    copyto!(M, qns.p, p)
    return qns
end
get_gradient(qns::QuasiNewtonState) = qns.X
function set_gradient!(qns::QuasiNewtonState, M, p, X)
    copyto!(M, qns.X, p, X)
    return qns
end
function default_stepsize(
    M::AbstractManifold,
    ::Type{QuasiNewtonState};
    vector_transport_method=default_vector_transport_method(M),
    retraction_method=default_retraction_method(M),
)
    return WolfePowellLinesearch(
        M;
        retraction_method=retraction_method,
        vector_transport_method=vector_transport_method,
        linesearch_stopsize=1e-10,
    )
end
@doc raw"""
    quasi_Newton(M, f, grad_f, p)

Perform a quasi Newton iteration for `f` on the manifold `M` starting
in the point `p`.

The ``k``th iteration consists of

1. Compute the search direction ``η_k = -\mathcal{B}_k [\operatorname{grad}f (p_k)]`` or solve ``\mathcal{H}_k [η_k] = -\operatorname{grad}f (p_k)]``.
2. Determine a suitable stepsize ``α_k`` along the curve ``\gamma(α) = R_{p_k}(α η_k)``, usually by using [`WolfePowellLinesearch`](@ref).
3. Compute `p_{k+1} = R_{p_k}(α_k η_k)``.
4. Define ``s_k = T_{p_k, α_k η_k}(α_k η_k)`` and ``y_k = \operatorname{grad}f(p_{k+1}) - T_{p_k, α_k η_k}(\operatorname{grad}f(p_k))``.
5. Compute the new approximate Hessian ``H_{k+1}`` or its inverse ``B_k``.

# Input

* `M`      a manifold ``\mathcal{M}``.
* `f`      a cost function ``F : \mathcal{M} →ℝ`` to minimize.
* `grad_f` the gradient ``\operatorname{grad}F : \mathcal{M} →T_x\mathcal M`` of ``F``.
* `p`      an initial value ``p ∈ \mathcal{M}``.

# Optional

* `basis`:                   (`DefaultOrthonormalBasis()`) basis within the tangent spaces
 to represent the Hessian (inverse).
* `cautious_update`:         (`false`) whether or not to use
  a [`QuasiNewtonCautiousDirectionUpdate`](@ref)
* `cautious_function`:       (`(x) -> x*10^(-4)`) a monotone increasing function that is zero
  at 0 and strictly increasing at 0 for the cautious update.
* `direction_update`:        ([`InverseBFGS`](@ref)`()`) the update rule to use.
* `evaluation`:              ([`AllocatingEvaluation`](@ref)) specify whether the gradient works by
   allocation (default) form `gradF(M, x)` or [`InplaceEvaluation`](@ref) in place of form `gradF!(M, X, x)`.
* `initial_operator`:        (`Matrix{Float64}(I,n,n)`) initial matrix to use die the
  approximation, where `n=manifold_dimension(M)`, see also `scale_initial_operator`.
* `memory_size`:             (`20`) limited memory, number of ``s_k, y_k`` to store. Set to a negative
  value to use a full memory representation
* `retraction_method`:       (`default_retraction_method(M, typeof(p))`) a retraction method to use
* `scale_initial_operator`:  (`true`) scale initial operator with
  ``\frac{⟨s_k,y_k⟩_{p_k}}{\lVert y_k\rVert_{p_k}}`` in the computation
* `stabilize`:               (`true`) stabilize the method numerically by projecting computed (Newton-)
  directions to the tangent space to reduce numerical errors
* `stepsize`:                ([`WolfePowellLinesearch`](@ref)`(retraction_method, vector_transport_method)`)
  specify a [`Stepsize`](@ref).
* `stopping_criterion`:      ([`StopAfterIteration`](@ref)`(max(1000, memory_size)) | `[`StopWhenGradientNormLess`](@ref)`(1e-6)`)
  specify a [`StoppingCriterion`](@ref)
* `vector_transport_method`: (`default_vector_transport_method(M, typeof(p))`) a vector transport to use.
* `nondescent_direction_behavior`: (`:step_towards_negative_gradient`) specify how non-descent direction is handled.
  This can be
  * ``:step_towards_negative_gradient` – the direction is replaced with negative gradient, a message is stored.
  * `:ignore` – the check is not performed, so any computed direction is accepted. No message is stored.
  * any other value performs the check, keeps the direction but stores a message.
  A stored message can be displayed using [`DebugMessages`](@ref).

# Output

the obtained (approximate) minimizer ``p^*``, see [`get_solver_return`](@ref) for details.
"""
function quasi_Newton(
    M::AbstractManifold,
    f::TF,
    grad_f::TDF,
    p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
) where {TF,TDF}
    mgo = ManifoldGradientObjective(f, grad_f; evaluation=evaluation)
    return quasi_Newton(M, mgo, p; kwargs...)
end
function quasi_Newton(
    M::AbstractManifold,
    f::TF,
    grad_f::TDF,
    p::Number;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
) where {TF,TDF}
    # redefine initial point
    q = [p]
    f_(M, p) = f(M, p[])
    grad_f_ = _to_mutating_gradient(grad_f, evaluation)
    rs = quasi_Newton(M, f_, grad_f_, q; evaluation=AllocatingEvaluation(), kwargs...)
    #return just a number if  the return type is the same as the type of q
    return (typeof(q) == typeof(rs)) ? rs[] : rs
end
function quasi_Newton(
    M::AbstractManifold, mgo::O, p; kwargs...
) where {O<:Union{ManifoldGradientObjective,AbstractDecoratedManifoldObjective}}
    q = copy(M, p)
    return quasi_Newton!(M, mgo, q; kwargs...)
end
@doc raw"""
    quasi_Newton!(M, F, gradF, x; options...)

Perform a quasi Newton iteration for `F` on the manifold `M` starting
in the point `x` using a retraction ``R`` and a vector transport ``T``.

# Input
* `M`     a manifold ``\mathcal{M}``.
* `F`     a cost function ``F: \mathcal{M} →ℝ`` to minimize.
* `gradF` the gradient ``\operatorname{grad}F : \mathcal{M} → T_x\mathcal M`` of ``F`` implemented as `gradF(M,p)`.
* `x`     an initial value ``x ∈ \mathcal{M}``.

For all optional parameters, see [`quasi_Newton`](@ref).
"""
quasi_Newton!(M::AbstractManifold, params...; kwargs...)
function quasi_Newton!(
    M::AbstractManifold,
    f::TF,
    grad_f::TDF,
    p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
) where {TF,TDF}
    mgo = ManifoldGradientObjective(f, grad_f; evaluation=evaluation)
    return quasi_Newton!(M, mgo, p; kwargs...)
end
function quasi_Newton!(
    M::AbstractManifold,
    mgo::O,
    p;
    cautious_update::Bool=false,
    cautious_function::Function=x -> x * 1e-4,
    debug=is_tutorial_mode() ? [DebugWarnIfGradientNormTooLarge()] : [],
    retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
    vector_transport_method::AbstractVectorTransportMethod=default_vector_transport_method(
        M, typeof(p)
    ),
    basis::AbstractBasis=DefaultOrthonormalBasis(),
    direction_update::AbstractQuasiNewtonUpdateRule=InverseBFGS(),
    memory_size::Int=min(manifold_dimension(M), 20),
    stabilize::Bool=true,
    initial_operator::AbstractMatrix=(
        if memory_size >= 0
            fill(1.0, 0, 0) # don't allocate `initial_operator` for limited memory operation
        else
            Matrix{Float64}(I, manifold_dimension(M), manifold_dimension(M))
        end
    ),
    scale_initial_operator::Bool=true,
    stepsize::Stepsize=default_stepsize(
        M,
        QuasiNewtonState;
        retraction_method=retraction_method,
        vector_transport_method=vector_transport_method,
    ),
    stopping_criterion::StoppingCriterion=StopAfterIteration(max(1000, memory_size)) |
                                          StopWhenGradientNormLess(1e-6),
    kwargs...,
) where {O<:Union{ManifoldGradientObjective,AbstractDecoratedManifoldObjective}}
    if memory_size >= 0
        local_dir_upd = QuasiNewtonLimitedMemoryDirectionUpdate(
            M,
            p,
            direction_update,
            memory_size;
            scale=scale_initial_operator,
            project=stabilize,
            vector_transport_method=vector_transport_method,
        )
    else
        local_dir_upd = QuasiNewtonMatrixDirectionUpdate(
            M,
            direction_update,
            basis,
            initial_operator;
            scale=scale_initial_operator,
            vector_transport_method=vector_transport_method,
        )
    end
    if cautious_update
        local_dir_upd = QuasiNewtonCautiousDirectionUpdate(
            local_dir_upd; θ=cautious_function
        )
    end
    dmgo = decorate_objective!(M, mgo; debug=debug, kwargs...)
    mp = DefaultManoptProblem(M, dmgo)
    qns = QuasiNewtonState(
        M,
        p;
        initial_vector=get_gradient(mp, p),
        direction_update=local_dir_upd,
        stopping_criterion=stopping_criterion,
        stepsize=stepsize,
        retraction_method=retraction_method,
        vector_transport_method=vector_transport_method,
    )
    dqns = decorate_state!(qns; debug=debug, kwargs...)
    solve!(mp, dqns)
    return get_solver_return(get_objective(mp), dqns)
end
function initialize_solver!(amp::AbstractManoptProblem, qns::QuasiNewtonState)
    M = get_manifold(amp)
    get_gradient!(amp, qns.X, qns.p)
    copyto!(M, qns.sk, qns.p, qns.X)
    copyto!(M, qns.yk, qns.p, qns.X)
    return qns
end
function step_solver!(mp::AbstractManoptProblem, qns::QuasiNewtonState, iter)
    M = get_manifold(mp)
    get_gradient!(mp, qns.X, qns.p)
    qns.direction_update(qns.η, mp, qns)
    if !(qns.nondescent_direction_behavior === :ignore)
        qns.nondescent_direction_value = real(inner(M, qns.p, qns.η, qns.X))
        if qns.nondescent_direction_value > 0
            if qns.nondescent_direction_behavior === :step_towards_negative_gradient
                copyto!(M, qns.η, qns.X)
                qns.η .*= -1
            end
        end
    end
    α = qns.stepsize(mp, qns, iter, qns.η)
    copyto!(M, qns.p_old, get_iterate(qns))
    retract!(M, qns.p, qns.p, qns.η, α, qns.retraction_method)
    qns.η .*= α
    # qns.yk update fails if α is equal to 0 because then β is NaN
    β = ifelse(
        iszero(α),
        one(α),
        locking_condition_scale(
            M, qns.direction_update, qns.p_old, qns.η, qns.p, qns.vector_transport_method
        ),
    )
    vector_transport_to!(
        M,
        qns.sk,
        qns.p_old,
        qns.η,
        qns.p,
        get_update_vector_transport(qns.direction_update),
    )
    vector_transport_to!(
        M, qns.X, qns.p_old, qns.X, qns.p, get_update_vector_transport(qns.direction_update)
    )
    copyto!(M, qns.X_old, qns.p, qns.X)
    get_gradient!(mp, qns.X, qns.p)
    qns.yk .= qns.X ./ β .- qns.X_old
    update_hessian!(qns.direction_update, mp, qns, qns.p_old, iter)
    return qns
end

function locking_condition_scale(
    M::AbstractManifold, ::AbstractQuasiNewtonDirectionUpdate, p_old, X, p, vtm
)
    return norm(M, p_old, X) / norm(M, p, vector_transport_to(M, p_old, X, p, vtm))
end

@doc raw"""
    update_hessian!(d, amp, st, p_old, iter)

update the Hessian within the [`QuasiNewtonState`](@ref) `o` given a [`AbstractManoptProblem`](@ref) `amp`
as well as the an [`AbstractQuasiNewtonDirectionUpdate`](@ref) `d` and the last iterate `p_old`.
Note that the current (`iter`th) iterate is already stored in `o.x`.

See also [`AbstractQuasiNewtonUpdateRule`](@ref) for the different rules that are available
within `d`.
"""
update_hessian!(d::AbstractQuasiNewtonDirectionUpdate, ::Any, ::Any, ::Any, ::Any)

function update_hessian!(
    d::QuasiNewtonMatrixDirectionUpdate{InverseBFGS},
    mp::AbstractManoptProblem,
    st::AbstractManoptSolverState,
    p_old,
    iter,
)
    M = get_manifold(mp)
    p = get_iterate(st)
    update_basis!(d.basis, M, p_old, p, d.vector_transport_method)
    yk_c = get_coordinates(M, p, st.yk, d.basis)
    sk_c = get_coordinates(M, p, st.sk, d.basis)
    skyk_c = inner(M, p, st.sk, st.yk)
    if iter == 1 && d.scale == true
        d.matrix = skyk_c / inner(M, p, st.yk, st.yk) * d.matrix
    end
    d.matrix =
        (I - sk_c * yk_c' / skyk_c) * d.matrix * (I - yk_c * sk_c' / skyk_c) +
        sk_c * sk_c' / skyk_c
    return d
end

# BFGS update
function update_hessian!(d::QuasiNewtonMatrixDirectionUpdate{BFGS}, mp::AbstractManoptProblem, st, p_old, iter)
    M = get_manifold(mp)
    p = get_iterate(st)
    update_basis!(d.basis, M, p_old, p, d.vector_transport_method)
    yk_c = get_coordinates(M, p, st.yk, d.basis)
    sk_c = get_coordinates(M, p, st.sk, d.basis)
    skyk_c = inner(M, p, st.sk, st.yk)
    if iter == 1 && d.scale == true
        d.matrix = inner(M, p, st.yk, st.yk) / skyk_c * d.matrix
    end
    d.matrix =
        d.matrix + yk_c * yk_c' / skyk_c -
        d.matrix * sk_c * sk_c' * d.matrix / dot(sk_c, d.matrix * sk_c)
    return d
end

# Inverse DFP update
function update_hessian!(
    d::QuasiNewtonMatrixDirectionUpdate{InverseDFP},
    mp::AbstractManoptProblem,
    st::AbstractManoptSolverState,
    p_old,
    iter,
)
    M = get_manifold(mp)
    p = get_iterate(st)
    update_basis!(d.basis, M, p_old, p, d.vector_transport_method)
    yk_c = get_coordinates(M, p, st.yk, d.basis)
    sk_c = get_coordinates(M, p, st.sk, d.basis)
    skyk_c = inner(M, p, st.sk, st.yk)
    if iter == 1 && d.scale == true
        d.matrix = inner(M, p, st.sk, st.sk) / skyk_c * d.matrix
    end
    d.matrix =
        d.matrix + sk_c * sk_c' / skyk_c -
        d.matrix * yk_c * yk_c' * d.matrix / dot(yk_c, d.matrix * yk_c)
    return d
end

# DFP update
function update_hessian!(
    d::QuasiNewtonMatrixDirectionUpdate{DFP},
    mp::AbstractManoptProblem,
    st::AbstractManoptSolverState,
    p_old,
    iter,
)
    M = get_manifold(mp)
    p = get_iterate(st)
    update_basis!(d.basis, M, p_old, p, d.vector_transport_method)
    yk_c = get_coordinates(M, p, st.yk, d.basis)
    sk_c = get_coordinates(M, p, st.sk, d.basis)
    skyk_c = inner(M, p, st.sk, st.yk)
    if iter == 1 && d.scale == true
        d.matrix = skyk_c / inner(M, p, st.sk, st.sk) * d.matrix
    end
    d.matrix =
        (I - yk_c * sk_c' / skyk_c) * d.matrix * (I - sk_c * yk_c' / skyk_c) +
        yk_c * yk_c' / skyk_c
    return d
end

# Inverse SR-1 update
function update_hessian!(
    d::QuasiNewtonMatrixDirectionUpdate{InverseSR1}, mp::AbstractManoptProblem, st::AbstractManoptSolverState, p_old, ::Int
)
    M = get_manifold(mp)
    p = get_iterate(st)
    update_basis!(d.basis, M, p_old, p, d.vector_transport_method)
    yk_c = get_coordinates(M, p, st.yk, d.basis)
    sk_c = get_coordinates(M, p, st.sk, d.basis)

    # computing the new matrix which represents the approximating operator in the next iteration
    srvec = sk_c - d.matrix * yk_c
    if d.update.r < 0 || abs(dot(srvec, yk_c)) >= d.update.r * norm(srvec) * norm(yk_c)
        d.matrix = d.matrix + srvec * srvec' / (srvec' * yk_c)
    end
    return d
end

# SR-1 update
function update_hessian!(
    d::QuasiNewtonMatrixDirectionUpdate{SR1},
    mp::AbstractManoptProblem,
    st::AbstractManoptSolverState,
    p_old,
    ::Int,
)
    M = get_manifold(mp)
    p = get_iterate(st)
    update_basis!(d.basis, M, p_old, p, d.vector_transport_method)
    yk_c = get_coordinates(M, p, st.yk, d.basis)
    sk_c = get_coordinates(M, p, st.sk, d.basis)

    # computing the new matrix which represents the approximating operator in the next iteration
    srvec = yk_c - d.matrix * sk_c
    if d.update.r < 0 || abs(dot(srvec, sk_c)) >= d.update.r * norm(srvec) * norm(sk_c)
        d.matrix = d.matrix + srvec * srvec' / (srvec' * sk_c)
    end
    return d
end

# Inverse Broyden update
function update_hessian!(
    d::QuasiNewtonMatrixDirectionUpdate{InverseBroyden},
    mp::AbstractManoptProblem,
    st::AbstractManoptSolverState,
    p_old,
    ::Int,
)
    M = get_manifold(mp)
    p = get_iterate(st)
    update_basis!(d.basis, M, p_old, p, d.vector_transport_method)
    yk_c = get_coordinates(M, p, st.yk, d.basis)
    sk_c = get_coordinates(M, p, st.sk, d.basis)
    skyk_c = inner(M, p, st.sk, st.yk)
    ykBkyk_c = yk_c' * d.matrix * yk_c

    φ = update_broyden_factor!(d, sk_c, yk_c, skyk_c, ykBkyk_c, d.update.update_rule)
    # computing the new matrix which represents the approximating operator in the next iteration
    d.matrix =
        d.matrix - (d.matrix * yk_c * yk_c' * d.matrix) / ykBkyk_c +
        (sk_c * sk_c') / skyk_c +
        φ *
        ykBkyk_c *
        (sk_c / skyk_c - (d.matrix * yk_c) / ykBkyk_c) *
        (sk_c / skyk_c - (d.matrix * yk_c) / ykBkyk_c)'
    return d
end

# Broyden update
function update_hessian!(
    d::QuasiNewtonMatrixDirectionUpdate{Broyden},
    mp::AbstractManoptProblem,
    st::AbstractManoptSolverState,
    p_old,
    ::Int,
)
    M = get_manifold(mp)
    p = get_iterate(st)
    update_basis!(d.basis, M, p_old, p, d.vector_transport_method)
    yk_c = get_coordinates(M, p, st.yk, d.basis)
    sk_c = get_coordinates(M, p, st.sk, d.basis)
    skyk_c = inner(M, p, st.sk, st.yk)
    skHksk_c = sk_c' * d.matrix * sk_c

    φ = update_broyden_factor!(d, sk_c, yk_c, skyk_c, skHksk_c, d.update.update_rule)
    # computing the new matrix which represents the approximating operator in the next iteration
    d.matrix =
        d.matrix - (d.matrix * sk_c * sk_c' * d.matrix) / skHksk_c +
        (yk_c * yk_c') / skyk_c +
        φ *
        skHksk_c *
        (yk_c / skyk_c - (d.matrix * sk_c) / skHksk_c) *
        (yk_c / skyk_c - (d.matrix * sk_c) / skHksk_c)'
    return d
end

function update_broyden_factor!(d, sk_c, yk_c, skyk_c, skHksk_c, s::Symbol)
    return update_broyden_factor!(d, sk_c, yk_c, skyk_c, skHksk_c, Val(s))
end

function update_broyden_factor!(d, ::Any, ::Any, ::Any, ::Any, ::Val{:constant})
    return d.update.φ
end

function update_broyden_factor!(d, ::Any, yk_c, skyk_c, skHksk_c, ::Val{:Davidon})
    yk_c_c = d.matrix \ yk_c
    ykyk_c_c = yk_c' * yk_c_c
    u = skyk_c <= 2 * (skHksk_c * ykyk_c_c) / (skHksk_c + ykyk_c_c)
    u && (d.update.φ = (skyk_c * (ykyk_c_c - skyk_c)) / (skHksk_c * ykyk_c_c - skyk_c^2))
    (!u) && (d.update.φ = skyk_c / (skyk_c - skHksk_c))
    return d.update.φ
end

function update_broyden_factor!(d, sk_c, ::Any, skyk_c, ykBkyk_c, ::Val{:InverseDavidon})
    sk_c_c = d.matrix \ sk_c
    sksk_c_c = sk_c' * sk_c_c
    u = skyk_c <= 2 * (ykBkyk_c * sksk_c_c) / (ykBkyk_c + sksk_c_c)
    u && (d.update.φ = (skyk_c * (sksk_c_c - skyk_c)) / (ykBkyk_c * sksk_c_c - skyk_c^2))
    (!u) && (d.update.φ = skyk_c / (skyk_c - ykBkyk_c))
    return d.update.φ
end

function update_basis!(
    b::AbstractBasis, ::AbstractManifold, ::P, ::P, ::AbstractVectorTransportMethod
) where {P}
    return b
end

function update_basis!(
    b::CachedBasis, M::AbstractManifold, x::P, y::P, m::AbstractVectorTransportMethod
) where {P}
    # transport all basis tangent vectors in the tangent space of the next iterate
    for v in b.data
        vector_transport_to!(M, v, x, v, y, m)
    end
    return b
end

# Cautious update
function update_hessian!(
    d::QuasiNewtonCautiousDirectionUpdate{U},
    mp::AbstractManoptProblem,
    st::AbstractManoptSolverState,
    p_old,
    iter,
) where {U<:AbstractQuasiNewtonDirectionUpdate}
    M = get_manifold(mp)
    p = get_iterate(st)
    X = get_gradient(st)
    # computing the bound used in the decision rule
    bound = d.θ(norm(M, p, X))
    sk_normsq = norm(M, p, st.sk)^2
    if sk_normsq != 0 && (inner(M, p, st.sk, st.yk) / sk_normsq) >= bound
        update_hessian!(d.update, mp, st, p_old, iter)
    end
    return d
end

# Limited-memory update
function update_hessian!(
    d::QuasiNewtonLimitedMemoryDirectionUpdate{U},
    mp::AbstractManoptProblem,
    st::AbstractManoptSolverState,
    p_old,
    ::Int,
) where {U<:InverseBFGS}
    (capacity(d.memory_s) == 0) && return d
    # only transport the first if it does not get overwritten at the end
    start = length(d.memory_s) == capacity(d.memory_s) ? 2 : 1
    M = get_manifold(mp)
    p = get_iterate(st)
    for i in start:length(d.memory_s)
        # transport all stored tangent vectors in the tangent space of the next iterate
        vector_transport_to!(
            M, d.memory_s[i], p_old, d.memory_s[i], p, d.vector_transport_method
        )
        vector_transport_to!(
            M, d.memory_y[i], p_old, d.memory_y[i], p, d.vector_transport_method
        )
    end

    # add newest
    # reuse old memory if buffer is full or allocate a copy if it is not
    if isfull(d.memory_s)
        old_sk = popfirst!(d.memory_s)
        copyto!(M, old_sk, st.sk)
        push!(d.memory_s, old_sk)
    else
        push!(d.memory_s, copy(M, st.sk))
    end
    if isfull(d.memory_y)
        old_yk = popfirst!(d.memory_y)
        copyto!(M, old_yk, st.yk)
        push!(d.memory_y, old_yk)
    else
        push!(d.memory_y, copy(M, st.yk))
    end
    return d
end

# all Cautious Limited Memory
function update_hessian!(
    d::QuasiNewtonCautiousDirectionUpdate{<:QuasiNewtonLimitedMemoryDirectionUpdate},
    mp::AbstractManoptProblem,
    st::AbstractManoptSolverState,
    p_old,
    iter,
)
    # computing the bound used in the decision rule
    M = get_manifold(mp)
    p = get_iterate(st)
    bound = d.θ(norm(M, p_old, get_gradient(mp, p_old)))
    sk_normsq = norm(M, p, st.sk)^2

    # if the decision rule is fulfilled, the new `sk` and `yk` are added
    if sk_normsq != 0 && real(inner(M, p, st.sk, st.yk) / sk_normsq) >= bound
        update_hessian!(d.update, mp, st, p_old, iter)
    else
        # the stored vectors are just transported to the new tangent space; `sk` and `yk` are not added
        for i in 1:length(d.update.memory_s)
            vector_transport_to!(
                M,
                d.update.memory_s[i],
                p_old,
                d.update.memory_s[i],
                p,
                d.update.vector_transport_method,
            )
            vector_transport_to!(
                M,
                d.update.memory_y[i],
                p_old,
                d.update.memory_y[i],
                p,
                d.update.vector_transport_method,
            )
        end
    end
    return d
end
