@doc raw"""
    QuasiNewtonState <: AbstractManoptSolverState

These Quasi Newton [`AbstractManoptSolverState`](@ref) represent any quasi-Newton based method and can be
used with any update rule for the direction.

# Fields
* `x` – the current iterate, a point on a manifold
* `gradient` – the current gradient
* `sk` – the current step
* `yk` the current gradient difference
* `direction_update` - an [`AbstractQuasiNewtonDirectionUpdate`](@ref) rule.
* `retraction_method` – an `AbstractRetractionMethod`
* `stop` – a [`StoppingCriterion`](@ref)

# Constructor

    QuasiNewtonState(
        M::AbstractManifold,
        x;
        initial_vector=zero_vector(M,x),
        direction_update::D=QuasiNewtonLimitedMemoryDirectionUpdate(M, x, InverseBFGS(), 20;
            vector_transport_method=vector_transport_method,
        )
        stopping_criterion=StopAfterIteration(1000) | StopWhenGradientNormLess(1e-6),
        retraction_method::RM=default_retraction_method(M),
        vector_transport_method::VTM=default_vector_transport_method(M),
        stepsize=default_stepsize(M; QuasiNewtonState)
    )

# See also
[`GradientProblem`](@ref)
"""
mutable struct QuasiNewtonState{
    P,
    T,
    D<:AbstractQuasiNewtonDirectionUpdate,
    SC<:StoppingCriterion,
    S<:Stepsize,
    RTR<:AbstractRetractionMethod,
    VT<:AbstractVectorTransportMethod,
} <: AbstractGradientSolverState
    x::P
    gradient::T
    sk::T
    yk::T
    direction_update::D
    retraction_method::RTR
    stepsize::S
    stop::SC
    vector_transport_method::VT
end
function QuasiNewtonState(
    M::AbstractManifold,
    x::P;
    initial_vector::T=zero_vector(M, x),
    vector_transport_method::VTM=default_vector_transport_method(M),
    direction_update::D=QuasiNewtonLimitedMemoryDirectionUpdate(
        M, x, InverseBFGS(), 20; vector_transport_method=vector_transport_method
    ),
    stopping_criterion::SC=StopAfterIteration(1000) | StopWhenGradientNormLess(1e-6),
    retraction_method::RM=default_retraction_method(M),
    stepsize::S=default_stepsize(
        M,
        QuasiNewtonState;
        retraction_method=retraction_method,
        vector_transport_method=vector_transport_method,
    ),
) where {
    P,
    T,
    D<:AbstractQuasiNewtonDirectionUpdate,
    SC<:StoppingCriterion,
    S<:Stepsize,
    RM<:AbstractRetractionMethod,
    VTM<:AbstractVectorTransportMethod,
}
    sk_init = zero_vector(M, x)
    return QuasiNewtonState{P,typeof(sk_init),D,SC,S,RM,VTM}(
        x,
        initial_vector,
        sk_init,
        copy(M, sk_init),
        direction_update,
        retraction_method,
        stepsize,
        stopping_criterion,
        vector_transport_method,
    )
end
# Temporary
get_iterate(qns::QuasiNewtonState) = qns.x
function set_iterate!(qns::QuasiNewtonState, M, p)
    copyto!(M, qns.x, p)
    return qns
end
get_gradient(qns::QuasiNewtonState) = qns.gradient
function set_gradient!(qns::QuasiNewtonState, M, p, X)
    copyto!(M, p, qns.gradient, X)
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
    quasi_Newton(M, F, gradF, x)

Perform a quasi Newton iteration for `F` on the manifold `M` starting
in the point `x` using a retraction ``R`` and a vector transport ``T``

The ``k``th iteration consists of
1. Compute the search direction ``η_k = -\mathcal{B}_k [\operatorname{grad}f (x_k)]`` or solve ``\mathcal{H}_k [η_k] = -\operatorname{grad}f (x_k)]``.
2. Determine a suitable stepsize ``α_k`` along the curve ``\gamma(α) = R_{x_k}(α η_k)`` e.g. by using [`WolfePowellLinesearch`](@ref).
3. Compute ``x_{k+1} = R_{x_k}(α_k η_k)``.
4. Define ``s_k = T_{x_k, α_k η_k}(α_k η_k)`` and ``y_k = \operatorname{grad}f(x_{k+1}) - T_{x_k, α_k η_k}(\operatorname{grad}f(x_k))``.
5. Compute the new approximate Hessian ``H_{k+1}`` or its inverse ``B_k``.

# Input
* `M` – a manifold ``\mathcal{M}``.
* `F` – a cost function ``F : \mathcal{M} →ℝ`` to minimize.
* `gradF`– the gradient ``\operatorname{grad}F : \mathcal{M} →T_x\mathcal M`` of ``F``.
* `x` – an initial value ``x ∈ \mathcal{M}``.

# Optional
* `basis` – (`DefaultOrthonormalBasis()`) basis within the tangent space(s) to represent the
  Hessian (inverse).
* `cautious_update` – (`false`) – whether or not to use
  a [`QuasiNewtonCautiousDirectionUpdate`](@ref)
* `cautious_function` – (`(x) -> x*10^(-4)`) – a monotone increasing function that is zero
  at 0 and strictly increasing at 0 for the cautious update.
* `direction_update` – ([`InverseBFGS`](@ref)`()`) the update rule to use.
* `evaluation` – ([`AllocatingEvaluation`](@ref)) specify whether the gradient works by
   allocation (default) form `gradF(M, x)` or [`InplaceEvaluation`](@ref) in place, i.e.
   is of the form `gradF!(M, X, x)`.
* `initial_operator` – (`Matrix{Float64}(I,n,n)`) initial matrix to use die the
  approximation, where `n=manifold_dimension(M)`, see also `scale_initial_operator`.
* `memory_size` – (`20`) limited memory, number of ``s_k, y_k`` to store. Set to a negative
  value to use a full memory representation
* `retraction_method` – (`default_retraction_method(M)`) a retraction method to use, by default
  the exponential map.
* `scale_initial_operator` - (`true`) scale initial operator with
  ``\frac{⟨s_k,y_k⟩_{x_k}}{\lVert y_k\rVert_{x_k}}`` in the computation
* `stabilize` – (`true`) stabilize the method numerically by projecting computed (Newton-)
  directions to the tangent space to reduce numerical errors
* `stepsize` – ([`WolfePowellLinesearch`](@ref)`(retraction_method, vector_transport_method)`)
  specify a [`Stepsize`](@ref).
* `stopping_criterion` - (`StopWhenAny(StopAfterIteration(max(1000, memory_size)), StopWhenGradientNormLess(10^(-6))`)
  specify a [`StoppingCriterion`](@ref)
* `vector_transport_method` – (`default_vector_transport_method(M)`) a vector transport to use.

# Output

the obtained (approximate) minimizer ``x^*``, see [`get_solver_return`](@ref) for details
"""
function quasi_Newton(M::AbstractManifold, F::TF, gradF::TDF, x; kwargs...) where {TF,TDF}
    x_res = copy(M, x)
    return quasi_Newton!(M, F, gradF, x_res; kwargs...)
end
@doc raw"""
    quasi_Newton!(M, F, gradF, x; options...)

Perform a quasi Newton iteration for `F` on the manifold `M` starting
in the point `x` using a retraction ``R`` and a vector transport ``T``.

# Input
* `M` – a manifold ``\mathcal{M}``.
* `F` – a cost function ``F: \mathcal{M} →ℝ`` to minimize.
* `gradF`– the gradient ``\operatorname{grad}F : \mathcal{M} → T_x\mathcal M`` of ``F`` implemented as `gradF(M,p)`.
* `x` – an initial value ``x ∈ \mathcal{M}``.

For all optional parameters, see [`quasi_Newton`](@ref).
"""
function quasi_Newton!(
    M::AbstractManifold,
    f::TF,
    grad_f::TDF,
    p;
    cautious_update::Bool=false,
    cautious_function::Function=x -> x * 10^(-4),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M),
    vector_transport_method::AbstractVectorTransportMethod=default_vector_transport_method(
        M
    ),
    basis::AbstractBasis=DefaultOrthonormalBasis(),
    direction_update::AbstractQuasiNewtonUpdateRule=InverseBFGS(),
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    memory_size::Int=20,
    stabilize=true,
    initial_operator::AbstractMatrix=(
        if memory_size >= 0
            fill(1.0, 0, 0) # don't allocate initial_operator for limited memory operation
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
) where {TF,TDF}
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
    mgo = ManifoldGradientObjective(f, grad_f; evaluation=evaluation)
    mp = DefaultManoptProblem(M, mgo)
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
    qns = decorate_state(qns; kwargs...)
    return get_solver_return(solve!(mp, qns))
end

function initialize_solver!(p::AbstractManoptProblem, s::QuasiNewtonState)
    s.gradient = get_gradient(p, s.x)
    s.sk = deepcopy(s.gradient)
    return s.yk = deepcopy(s.gradient)
end

function step_solver!(mp::AbstractManoptProblem, qns::QuasiNewtonState, iter)
    M = get_manifold(mp)
    qns.gradient = get_gradient(mp, qns.x)
    η = qns.direction_update(mp, qns)
    α = qns.stepsize(mp, qns, iter, η)
    p_old = copy(M, get_iterate(qns))
    retract!(M, qns.x, qns.x, α * η, qns.retraction_method)
    β = locking_condition_scale(
        M, qns.direction_update, p_old, α * η, qns.x, qns.vector_transport_method
    )
    vector_transport_to!(
        M, qns.sk, p_old, α * η, qns.x, get_update_vector_transport(qns.direction_update)
    )
    vector_transport_to!(
        M,
        qns.gradient,
        p_old,
        qns.gradient,
        qns.x,
        get_update_vector_transport(qns.direction_update),
    )
    new_grad = get_gradient(mp, qns.x)
    qns.yk = new_grad / β - qns.gradient
    update_hessian!(qns.direction_update, mp, qns, p_old, iter)
    return qns
end

function locking_condition_scale(
    M::AbstractManifold, ::AbstractQuasiNewtonDirectionUpdate, p_old, v, x, vt
)
    return norm(M, p_old, v) / norm(M, x, vector_transport_to(M, p_old, v, x, vt))
end

@doc raw"""
    update_hessian!(d, mp, st, p_old, iter)

update the hessian wihtin the [`QuasiNewtonState`](@ref) `o` given a [`AbstractManoptProblem`](@ref) `p`
as well as the an [`AbstractQuasiNewtonDirectionUpdate`](@ref) `d` and the last iterate `p_old`.
Note that the current (`iter`th) iterate is already stored in `o.x`.

See also [`AbstractQuasiNewtonUpdateRule`](@ref) for the different rules that are available
within `d`.
"""
update_hessian!(d::AbstractQuasiNewtonDirectionUpdate, ::Any, ::Any, ::Any, ::Any)

function update_hessian!(
    d::QuasiNewtonMatrixDirectionUpdate{InverseBFGS}, mp, st, p_old, iter
)
    M = get_manifold(mp)
    update_basis!(d.basis, M, p_old, st.x, d.vector_transport_method)
    yk_c = get_coordinates(M, st.x, st.yk, d.basis)
    sk_c = get_coordinates(M, st.x, st.sk, d.basis)
    skyk_c = inner(M, st.x, st.sk, st.yk)
    if iter == 1 && d.scale == true
        d.matrix = skyk_c / inner(M, st.x, st.yk, st.yk) * d.matrix
    end
    d.matrix =
        (I - sk_c * yk_c' / skyk_c) * d.matrix * (I - yk_c * sk_c' / skyk_c) +
        sk_c * sk_c' / skyk_c
    return d
end

# BFGS update
function update_hessian!(d::QuasiNewtonMatrixDirectionUpdate{BFGS}, mp, st, p_old, iter)
    M = get_manifold(mp)
    update_basis!(d.basis, M, p_old, st.x, d.vector_transport_method)
    yk_c = get_coordinates(M, st.x, st.yk, d.basis)
    sk_c = get_coordinates(M, st.x, st.sk, d.basis)
    skyk_c = inner(M, st.x, st.sk, st.yk)
    if iter == 1 && d.scale == true
        d.matrix = inner(M, st.x, st.yk, st.yk) / skyk_c * d.matrix
    end
    d.matrix =
        d.matrix + yk_c * yk_c' / skyk_c -
        d.matrix * sk_c * sk_c' * d.matrix / dot(sk_c, d.matrix * sk_c)
    return d
end

# Inverse DFP update
function update_hessian!(
    d::QuasiNewtonMatrixDirectionUpdate{InverseDFP}, mp, st, p_old, iter
)
    M = get_manifold(mp)
    update_basis!(d.basis, M, p_old, st.x, d.vector_transport_method)
    yk_c = get_coordinates(M, st.x, st.yk, d.basis)
    sk_c = get_coordinates(M, st.x, st.sk, d.basis)
    skyk_c = inner(M, st.x, st.sk, st.yk)
    if iter == 1 && d.scale == true
        d.matrix = inner(M, st.x, st.sk, st.sk) / skyk_c * d.matrix
    end
    d.matrix =
        d.matrix + sk_c * sk_c' / skyk_c -
        d.matrix * yk_c * yk_c' * d.matrix / dot(yk_c, d.matrix * yk_c)
    return d
end

# DFP update
function update_hessian!(d::QuasiNewtonMatrixDirectionUpdate{DFP}, mp, st, p_old, iter)
    M = get_manifold(mp)
    update_basis!(d.basis, M, p_old, st.x, d.vector_transport_method)
    yk_c = get_coordinates(M, st.x, st.yk, d.basis)
    sk_c = get_coordinates(M, st.x, st.sk, d.basis)
    skyk_c = inner(M, st.x, st.sk, st.yk)
    if iter == 1 && d.scale == true
        d.matrix = skyk_c / inner(M, st.x, st.sk, st.sk) * d.matrix
    end
    d.matrix =
        (I - yk_c * sk_c' / skyk_c) * d.matrix * (I - sk_c * yk_c' / skyk_c) +
        yk_c * yk_c' / skyk_c
    return d
end

# Inverse SR-1 update
function update_hessian!(
    d::QuasiNewtonMatrixDirectionUpdate{InverseSR1}, mp, st, p_old, ::Int
)
    M = get_manifold(mp)
    update_basis!(d.basis, M, p_old, st.x, d.vector_transport_method)
    yk_c = get_coordinates(M, st.x, st.yk, d.basis)
    sk_c = get_coordinates(M, st.x, st.sk, d.basis)

    # computing the new matrix which represents the approximating operator in the next iteration
    srvec = sk_c - d.matrix * yk_c
    if d.update.r < 0 || abs(dot(srvec, yk_c)) >= d.update.r * norm(srvec) * norm(yk_c)
        d.matrix = d.matrix + srvec * srvec' / (srvec' * yk_c)
    end
    return d
end

# SR-1 update
function update_hessian!(d::QuasiNewtonMatrixDirectionUpdate{SR1}, mp, st, p_old, ::Int)
    M = get_manifold(mp)
    update_basis!(d.basis, M, p_old, st.x, d.vector_transport_method)
    yk_c = get_coordinates(M, st.x, st.yk, d.basis)
    sk_c = get_coordinates(M, st.x, st.sk, d.basis)

    # computing the new matrix which represents the approximating operator in the next iteration
    srvec = yk_c - d.matrix * sk_c
    if d.update.r < 0 || abs(dot(srvec, sk_c)) >= d.update.r * norm(srvec) * norm(sk_c)
        d.matrix = d.matrix + srvec * srvec' / (srvec' * sk_c)
    end
    return d
end

# Inverse Broyden update
function update_hessian!(
    d::QuasiNewtonMatrixDirectionUpdate{InverseBroyden}, mp, st, p_old, ::Int
)
    M = get_manifold(mp)
    update_basis!(d.basis, M, p_old, st.x, d.vector_transport_method)
    yk_c = get_coordinates(M, st.x, st.yk, d.basis)
    sk_c = get_coordinates(M, st.x, st.sk, d.basis)
    skyk_c = inner(M, st.x, st.sk, st.yk)
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
function update_hessian!(d::QuasiNewtonMatrixDirectionUpdate{Broyden}, mp, st, p_old, ::Int)
    M = get_manifold(mp)
    update_basis!(d.basis, M, p_old, st.x, d.vector_transport_method)
    yk_c = get_coordinates(M, st.x, st.yk, d.basis)
    sk_c = get_coordinates(M, st.x, st.sk, d.basis)
    skyk_c = inner(M, st.x, st.sk, st.yk)
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
    d::QuasiNewtonCautiousDirectionUpdate{U}, mp, st, p_old, iter
) where {U<:AbstractQuasiNewtonDirectionUpdate}
    M = get_manifold(mp)
    # computing the bound used in the decission rule
    bound = d.θ(norm(M, st.x, st.gradient))
    sk_normsq = norm(M, st.x, st.sk)^2
    if sk_normsq != 0 && (inner(M, st.x, st.sk, st.yk) / sk_normsq) >= bound
        update_hessian!(d.update, mp, st, p_old, iter)
    end
    return d
end

# Limited-memory update
function update_hessian!(
    d::QuasiNewtonLimitedMemoryDirectionUpdate{U}, mp, st, p_old, ::Int
) where {U<:InverseBFGS}
    (capacity(d.memory_s) == 0) && return d
    # only transport the first if it does not get overwritten at the end
    start = length(d.memory_s) == capacity(d.memory_s) ? 2 : 1
    M = get_manifold(mp)
    for i in start:length(d.memory_s)
        # transport all stored tangent vectors in the tangent space of the next iterate
        vector_transport_to!(
            M, d.memory_s[i], p_old, d.memory_s[i], st.x, d.vector_transport_method
        )
        vector_transport_to!(
            M, d.memory_y[i], p_old, d.memory_y[i], st.x, d.vector_transport_method
        )
    end

    # add newest
    push!(d.memory_s, st.sk)
    push!(d.memory_y, st.yk)
    return d
end

# all Cautious Limited Memory
function update_hessian!(
    d::QuasiNewtonCautiousDirectionUpdate{QuasiNewtonLimitedMemoryDirectionUpdate{NT,T,VT}},
    mp,
    st,
    p_old,
    iter,
) where {NT<:AbstractQuasiNewtonUpdateRule,T,VT<:AbstractVectorTransportMethod}
    # computing the bound used in the decission rule
    M = get_manifold(mp)
    bound = d.θ(norm(M, p_old, get_gradient(mp, p_old)))
    sk_normsq = norm(M, st.x, st.sk)^2

    # if the decission rule is fulfilled, the new sk and yk are added
    if sk_normsq != 0 && (inner(M, st.x, st.sk, st.yk) / sk_normsq) >= bound
        update_hessian!(d.update, mp, st, p_old, iter)
    else
        # the stored vectores are just transported to the new tangent space, sk and yk are not added
        for i in 1:length(d.update.memory_s)
            vector_transport_to!(
                M,
                d.update.memory_s[i],
                p_old,
                d.update.memory_s[i],
                st.x,
                d.update.vector_transport_method,
            )
            vector_transport_to!(
                M,
                d.update.memory_y[i],
                p_old,
                d.update.memory_y[i],
                st.x,
                d.update.vector_transport_method,
            )
        end
    end
    return d
end
