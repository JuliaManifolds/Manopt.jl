@doc """
    QuasiNewtonState <: AbstractManoptSolverState

The [`AbstractManoptSolverState`](@ref) represent any quasi-Newton based method and stores
all necessary fields.

# Fields

* `direction_update`:              an [`AbstractQuasiNewtonDirectionUpdate`](@ref) rule.
* `η`:                             the current update direction
* `nondescent_direction_behavior`: a `Symbol` to specify how to handle direction that are not descent ones.
* `nondescent_direction_value`:    the value from the last inner product from checking for descent directions
* $(_field_p)
* `p_old`:                         the last iterate
* `sk`:                            the current step
* `yk`:                            the current gradient difference
* $(_field_retr)
* $(_field_step)
* $(_field_stop)
* $(_field_gradient)
* `X_old`:                         the last gradient


# Constructor

    QuasiNewtonState(M::AbstractManifold, p; kwargs...)

Generate the Quasi Newton state on the manifold `M` with start point `p`.

## Keyword arguments

* `direction_update=`[`QuasiNewtonLimitedMemoryDirectionUpdate`](@ref)`(M, p, InverseBFGS(), 20; vector_transport_method=vector_transport_method)`
* `stopping_criterion=`[`StopAfterIteration`9(@ref)`(1000)`$(_sc_any)[`StopWhenGradientNormLess`](@ref)`(1e-6)`
* $(_kw_retraction_method_default): $(_kw_retraction_method)
* `stepsize=default_stepsize(M; QuasiNewtonState)`: $(_kw_stepsize)
  The default here is the [`WolfePowellLinesearch`](@ref) using the keywords `retraction_method` and `vector_transport_method`
* $(_kw_vector_transport_method_default): $(_kw_vector_transport_method)
* $(_kw_X_default)

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
    initial_vector::T=zero_vector(M, p), # deprecated
    X::T=initial_vector,
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
    nondescent_direction_behavior::Symbol=:reinitialize_direction_update,
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
    return QuasiNewtonState{P,T,D,SC,S,RM,VTM,Float64}(
        p,
        copy(M, p),
        copy(M, p, X),
        X,
        copy(M, p, X),
        copy(M, p, X),
        direction_update,
        retraction_method,
        stepsize,
        stopping_criterion,
        copy(M, p, X),
        vector_transport_method,
        nondescent_direction_behavior,
        1.0,
    )
end
function get_message(qns::QuasiNewtonState)
    # collect messages from
    # (1) direction update or the
    # (2) the step size and combine them
    # (3) the non-descent behaviour verification message
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
        stop_when_stepsize_less=1e-10,
    )
end
_doc_QN_init_scaling = raw"``\frac{⟨s_k,y_k⟩_{p_k}}{\lVert y_k\rVert_{p_k}}``"
_doc_QN = """
    quasi_Newton(M, f, grad_f, p; kwargs...)
    quasi_Newton!(M, f, grad_f, p; kwargs...)

Perform a quasi Newton iteration to solve

$(_problem_default)

with start point `p`. The iterations can be done in-place of `p```=p^{(0)}``.
The ``k``th iteration consists of

1. Compute the search direction ``η^{(k)} = -$(_l_cal("B"))_k [$(_l_grad)f (p^{(k)})]`` or solve ``$(_l_cal("H"))_k [η^{(k)}] = -$(_l_grad)f (p^{(k)})]``.
2. Determine a suitable stepsize ``α_k`` along the curve ``γ(α) = R_{p^{(k)}}(α η^{(k)})``, usually by using [`WolfePowellLinesearch`](@ref).
3. Compute ``p^{(k+1)} = R_{p^{(k)}}(α_k η^{(k)})``.
4. Define ``s_k = $(_l_cal("T"))_{p^{(k)}, α_k η^{(k)}}(α_k η^{(k)})`` and ``y_k = $(_l_grad)f(p^{(k+1)}) - $(_l_cal("T"))_{p^{(k)}, α_k η^{(k)}}($(_l_grad)f(p^{(k)}))``, where ``$(_l_cal("T"))`` denotes a vector transport.
5. Compute the new approximate Hessian ``H_{k+1}`` or its inverse ``B_{k+1}``.

# Input

$(_arg_M)
$(_arg_f)
$(_arg_grad_f)
$(_arg_p)

# Keyword arguments

* `basis=`[`DefaultOrthonormalBasis`](@extref ManifoldsBase.DefaultOrthonormalBasis)`()`:
  basis to use within each of the the tangent spaces to represent
  the Hessian (inverse) for the cases where it is stored in full (matrix) form.
* `cautious_update=false`:
   whether or not to use the [`QuasiNewtonCautiousDirectionUpdate`](@ref)
   which wraps the `direction_upate`.
* `cautious_function=(x) -> x * 1e-4`:
  a monotone increasing function for the cautious update that is zero at ``x=0``
  and strictly increasing at ``0``
* `direction_update=`[`InverseBFGS`](@ref)`()`:
  the [`AbstractQuasiNewtonUpdateRule`](@ref) to use.
* $(_kw_evaluation_default):
  $(_kw_evaluation)
  $(_kw_evaluation_example)
* `initial_operator=Matrix{Float64}(I, n, n)`:
   initial matrix to use in case the Hessian (inverse) approximation is stored as a full matrix,
   that is `n=manifold_dimension(M)`. This matrix is only allocated for the full matrix case.
   See also `scale_initial_operator`.
* `memory_size=20`: limited memory, number of ``s_k, y_k`` to store.
   Set to a negative value to use a full memory (matrix) representation
* `nondescent_direction_behavior=:reinitialize_direction_update`:
  specify how non-descent direction is handled. This can be
  * `:step_towards_negative_gradient`: the direction is replaced with negative gradient, a message is stored.
  * `:ignore`: the verification is not performed, so any computed direction is accepted. No message is stored.
  * `:reinitialize_direction_update`: discards operator state stored in direction update rules.
  * any other value performs the verification, keeps the direction but stores a message.
  A stored message can be displayed using [`DebugMessages`](@ref).
* `project!=copyto!`: for numerical stability it is possible to project onto the tangent space after every iteration.
  the function has to work inplace of `Y`, that is `(M, Y, p, X) -> Y`, where `X` and `Y` can be the same memory.
* $(_kw_retraction_method_default): $(_kw_retraction_method)
* `scale_initial_operator=true`: scale initial operator with $(_doc_QN_init_scaling) in the computation
* `stepsize=`[`WolfePowellLinesearch`](@ref)`(retraction_method, vector_transport_method)`:
  $(_kw_stepsize)
* `stopping_criterion=`[`StopAfterIteration`](@ref)`(max(1000, memory_size))`$(_sc_any)[`StopWhenGradientNormLess`](@ref)`(1e-6)`:
  $(_kw_stopping_criterion)
* $(_kw_vector_transport_method_default): $(_kw_vector_transport_method)

$(_kw_others)

$(_doc_sec_output)
"""

@doc "$(_doc_QN)"
function quasi_Newton(
    M::AbstractManifold,
    f::TF,
    grad_f::TDF,
    p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
) where {TF,TDF}
    p_ = _ensure_mutating_variable(p)
    f_ = _ensure_mutating_cost(f, p)
    grad_f_ = _ensure_mutating_gradient(grad_f, p, evaluation)
    mgo = ManifoldGradientObjective(f_, grad_f_; evaluation=evaluation)
    rs = quasi_Newton(M, mgo, p_; kwargs...)
    return _ensure_matching_output(p, rs)
end
function quasi_Newton(
    M::AbstractManifold, mgo::O, p; kwargs...
) where {O<:Union{ManifoldGradientObjective,AbstractDecoratedManifoldObjective}}
    q = copy(M, p)
    return quasi_Newton!(M, mgo, q; kwargs...)
end

@doc "$(_doc_QN)"
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
    (project!)=copyto!,
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
            (project!)=project!,
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
    initialize_update!(qns.direction_update)
    return qns
end
function step_solver!(mp::AbstractManoptProblem, qns::QuasiNewtonState, k)
    M = get_manifold(mp)
    get_gradient!(mp, qns.X, qns.p)
    qns.direction_update(qns.η, mp, qns)
    if !(qns.nondescent_direction_behavior === :ignore)
        qns.nondescent_direction_value = real(inner(M, qns.p, qns.η, qns.X))
        if qns.nondescent_direction_value > 0
            if qns.nondescent_direction_behavior === :step_towards_negative_gradient ||
                qns.nondescent_direction_behavior === :reinitialize_direction_update
                copyto!(M, qns.η, qns.X)
                qns.η .*= -1
            end
            if qns.nondescent_direction_behavior === :reinitialize_direction_update
                initialize_update!(qns.direction_update)
            end
        end
    end
    α = qns.stepsize(mp, qns, k, qns.η)
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
    update_hessian!(qns.direction_update, mp, qns, qns.p_old, k)
    return qns
end

function locking_condition_scale(
    M::AbstractManifold, ::AbstractQuasiNewtonDirectionUpdate, p_old, X, p, vtm
)
    return norm(M, p_old, X) / norm(M, p, vector_transport_to(M, p_old, X, p, vtm))
end

@doc raw"""
    update_hessian!(d::AbstractQuasiNewtonDirectionUpdate, amp, st, p_old, k)

update the Hessian within the [`QuasiNewtonState`](@ref) `st` given a [`AbstractManoptProblem`](@ref) `amp`
as well as the an [`AbstractQuasiNewtonDirectionUpdate`](@ref) `d` and the last iterate `p_old`.
Note that the current (`k`th) iterate is already stored in [`get_iterate`](@ref)`(st)`.

See also [`AbstractQuasiNewtonUpdateRule`](@ref) and its subtypes for the different rules
that are available within `d`.
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
function update_hessian!(
    d::QuasiNewtonMatrixDirectionUpdate{BFGS}, mp::AbstractManoptProblem, st, p_old, iter
)
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
    d::QuasiNewtonMatrixDirectionUpdate{InverseSR1},
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
