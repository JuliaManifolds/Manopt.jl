@doc raw"""
    quasi_Newton(M, F, gradF, x)

Perform a quasi Newton iteration for `F` on the manifold `M` starting
in the point `x` using a retraction ``R`` and a vector transport ``T``

The ``k``th iteration consists of
1. Compute the search direction ``η_k = -\mathcal{B}_k [\operatorname{grad}f (x_k)]`` or solve ``\mathcal{H}_k [η_k] = -\operatorname{grad}f (x_k)]``.
2. Determine a suitable stepsize ``α_k`` along the curve ``\gamma(α) = R_{x_k}(α η_k)`` e.g. by using [`WolfePowellLineseach`](@ref).
3. Compute ``x_{k+1} = R_{x_k}(α_k η_k)``.
4. Define ``s_k = T_{x_k, α_k η_k}(α_k η_k)`` and ``y_k = \operatorname{grad}f(x_{k+1}) - T_{x_k, α_k η_k}(\operatorname{grad}f(x_k))``.
5. Compute the new approximate Hessian ``H_{k+1}`` or its inverse ``B_k``.

# Input
* `M` – a manifold ``\mathcal{M}``.
* `F` – a cost function ``F : \mathcal{M} →ℝ`` to minimize.
* `gradF`– the gradient ``\operatorname{grad}F : \mathcal{M} →T_x\mathcal M`` of ``F``.
* `x` – an initial value ``x ∈ \mathcal{M}``.

    stopping_criterion::StoppingCriterion=StopWhenAny(
        StopAfterIteration(max(1000, memory_size)), StopWhenGradientNormLess(10^(-6))
    ),
    return_options=false,


# Optional
* `basis` – (`DefaultOrthonormalBasis()`) basis within the tangent space(s) to represent the
  Hessian (inverse).
* `cautious_update` – (`false`) – whether or not to use
  a [`QuasiNewtonCautiousDirectionUpdate`](@ref)
* `cautious_function` – (`(x) -> x*10^(-4)`) – a monotone increasing function that is zero
  at 0 and strictly increasing at 0 for the cautious update.
* `direction_update` – ([`InverseBFGS`](@ref)`()`) the update rule to use.
* `evaluation` – ([`AllocatingEvaluation`](@ref)) specify whether the gradient works by
   allocation (default) form `gradF(M, x)` or [`MutatingEvaluation`](@ref) in place, i.e.
   is of the form `gradF!(M, X, x)`.
* `initial_operator` – (`Matrix{Float64}(I,n,n)`) initial matrix to use die the
  approximation, where `n=manifold_dimension(M)`, see also `scale_initial_operator`.
* `memory_size` – (`20`) limited memory, number of ``s_k, y_k`` to store. Set to a negative
  value to use a full memory representation
* `retraction_method` – (`default_retraction_method(M)`) a retraction method to use, by default
  the exponential map.
* `scale_initial_operator` - (`true`) scale initial operator with
  ``\frac{⟨s_k,y_k⟩_{x_k}}{\lVert y_k\rVert_{x_k}}`` in the computation
* `step_size` – ([`WolfePowellLineseach`](@ref)`(retraction_method, vector_transport_method)`)
  specify a [`Stepsize`](@ref).
* `stopping_criterion` - (`StopWhenAny(StopAfterIteration(max(1000, memory_size)), StopWhenGradientNormLess(10^(-6))`)
  specify a [`StoppingCriterion`](@ref)
* `vector_transport_method` – (`default_vector_transport_method(M)`) a vector transport to use.
* `return_options` – (`false`) – specify whether to return just the result `x` (default) or the complete [`Options`](@ref), e.g. to access recorded values. if activated, the extended result, i.e. the

# Output
* `x_opt` – the resulting (approximately critical) point of the quasi–Newton method
OR
* `options` – the options returned by the solver (see `return_options`)
"""
function quasi_Newton(
    M::AbstractManifold, F::Function, gradF::G, x::P; kwargs...
) where {P,G}
    x_res = allocate(x)
    copyto!(M, x_res, x)
    return quasi_Newton!(M, F, gradF, x_res; kwargs...)
end
@doc raw"""
    quasi_Newton!(M, F, gradF, x; options...)

Perform a quasi Newton iteration for `F` on the manifold `M` starting
in the point `x` using a retraction ``R`` and a vector transport ``T``.

# Input
* `M` – a manifold ``\mathcal{M}``.
* `F` – a cost function ``F: \mathcal{M} →ℝ`` to minimize.
* `gradF`– the gradient ``\operatorname{grad}F : \mathcal{M} → T_x\mathcal M`` of ``F``.
* `x` – an initial value ``x ∈ \mathcal{M}``.

For all optional parameters, see [`quasi_Newton`](@ref).
"""
function quasi_Newton!(
    M::AbstractManifold,
    F::Function,
    gradF::G,
    x::P;
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
    initial_operator::AbstractMatrix=Matrix{Float64}(
        I, manifold_dimension(M), manifold_dimension(M)
    ),
    scale_initial_operator::Bool=true,
    step_size::Stepsize=WolfePowellLineseach(retraction_method, vector_transport_method),
    stopping_criterion::StoppingCriterion=StopWhenAny(
        StopAfterIteration(max(1000, memory_size)), StopWhenGradientNormLess(10^(-6))
    ),
    return_options=false,
    kwargs...,
) where {P,G}
    if memory_size >= 0
        local_dir_upd = QuasiNewtonLimitedMemoryDirectionUpdate(
            direction_update,
            zero_vector(M, x),
            memory_size;
            scale=scale_initial_operator,
            vector_transport_method=vector_transport_method,
        )
    else
        local_dir_upd = QuasiNewtonMatrixDirectionUpdate(
            direction_update,
            basis,
            initial_operator;
            scale=scale_initial_operator,
            vector_transport_method=vector_transport_method,
        )
    end
    if cautious_update == true
        local_dir_upd = QuasiNewtonCautiousDirectionUpdate(
            local_dir_upd; θ=cautious_function
        )
    end

    p = GradientProblem(M, F, gradF; evaluation=evaluation)
    o = QuasiNewtonOptions(
        x,
        get_gradient(p, x),
        local_dir_upd,
        stopping_criterion,
        step_size;
        retraction_method=retraction_method,
        vector_transport_method=vector_transport_method,
    )

    o = decorate_options(o; kwargs...)
    resultO = solve(p, o)

    if return_options
        return resultO
    else
        return get_solver_result(resultO)
    end
end

function initialize_solver!(::GradientProblem, ::QuasiNewtonOptions) end

function step_solver!(p::GradientProblem, o::QuasiNewtonOptions, iter)
    o.gradient = get_gradient(p, o.x)
    η = o.direction_update(p, o)
    α = o.stepsize(p, o, iter, η)
    x_old = deepcopy(o.x)
    retract!(p.M, o.x, o.x, α * η, o.retraction_method)
    β = locking_condition_scale(
        p.M, o.direction_update, x_old, α * η, o.x, o.vector_transport_method
    )
    vector_transport_to!(
        p.M, o.sk, x_old, α * η, o.x, get_update_vector_transport(o.direction_update)
    )
    vector_transport_to!(
        p.M,
        o.gradient,
        x_old,
        o.gradient,
        o.x,
        get_update_vector_transport(o.direction_update),
    )
    o.yk = get_gradient(p, o.x) / β - o.gradient
    update_hessian!(o.direction_update, p, o, x_old, iter)
    return o
end

function locking_condition_scale(
    M::AbstractManifold, ::AbstractQuasiNewtonDirectionUpdate, x_old, v, x, vt
)
    return norm(M, x_old, v) / norm(M, x, vector_transport_to(M, x_old, v, x, vt))
end

@doc raw"""
    update_hessian!(d, p, o, x_old, iter)

update the hessian wihtin the [`QuasiNewtonOptions`](@ref) `o` given a [`Problem`](@ref) `p`
as well as the an [`AbstractQuasiNewtonDirectionUpdate`](@ref) `d` and the last iterate `x_old`.
Note that the current (`iter`th) iterate is already stored in `o.x`.

See also [`AbstractQuasiNewtonUpdateRule`](@ref) for the different rules that are available
within `d`.
"""
update_hessian!(d::AbstractQuasiNewtonDirectionUpdate, ::Any, ::Any, ::Any, ::Any)

function update_hessian!(
    d::QuasiNewtonMatrixDirectionUpdate{InverseBFGS}, p, o, x_old, iter
)
    update_basis!(d.basis, p.M, x_old, o.x, d.vector_transport_method)
    yk_c = get_coordinates(p.M, o.x, o.yk, d.basis)
    sk_c = get_coordinates(p.M, o.x, o.sk, d.basis)
    skyk_c = inner(p.M, o.x, o.sk, o.yk)
    if iter == 1 && d.scale == true
        d.matrix = skyk_c / inner(p.M, o.x, o.yk, o.yk) * d.matrix
    end
    d.matrix =
        (I - sk_c * yk_c' / skyk_c) * d.matrix * (I - yk_c * sk_c' / skyk_c) +
        sk_c * sk_c' / skyk_c
    return d
end

# BFGS update
function update_hessian!(d::QuasiNewtonMatrixDirectionUpdate{BFGS}, p, o, x_old, iter)
    update_basis!(d.basis, p.M, x_old, o.x, d.vector_transport_method)
    yk_c = get_coordinates(p.M, o.x, o.yk, d.basis)
    sk_c = get_coordinates(p.M, o.x, o.sk, d.basis)
    skyk_c = inner(p.M, o.x, o.sk, o.yk)
    if iter == 1 && d.scale == true
        d.matrix = inner(p.M, o.x, o.yk, o.yk) / skyk_c * d.matrix
    end
    d.matrix =
        d.matrix + yk_c * yk_c' / skyk_c -
        d.matrix * sk_c * sk_c' * d.matrix / dot(sk_c, d.matrix * sk_c)
    return d
end

# Inverse DFP update
function update_hessian!(d::QuasiNewtonMatrixDirectionUpdate{InverseDFP}, p, o, x_old, iter)
    update_basis!(d.basis, p.M, x_old, o.x, d.vector_transport_method)
    yk_c = get_coordinates(p.M, o.x, o.yk, d.basis)
    sk_c = get_coordinates(p.M, o.x, o.sk, d.basis)
    skyk_c = inner(p.M, o.x, o.sk, o.yk)
    if iter == 1 && d.scale == true
        d.matrix = inner(p.M, o.x, o.sk, o.sk) / skyk_c * d.matrix
    end
    d.matrix =
        d.matrix + sk_c * sk_c' / skyk_c -
        d.matrix * yk_c * yk_c' * d.matrix / dot(yk_c, d.matrix * yk_c)
    return d
end

# DFP update
function update_hessian!(d::QuasiNewtonMatrixDirectionUpdate{DFP}, p, o, x_old, iter)
    update_basis!(d.basis, p.M, x_old, o.x, d.vector_transport_method)
    yk_c = get_coordinates(p.M, o.x, o.yk, d.basis)
    sk_c = get_coordinates(p.M, o.x, o.sk, d.basis)
    skyk_c = inner(p.M, o.x, o.sk, o.yk)
    if iter == 1 && d.scale == true
        d.matrix = skyk_c / inner(p.M, o.x, o.sk, o.sk) * d.matrix
    end
    d.matrix =
        (I - yk_c * sk_c' / skyk_c) * d.matrix * (I - sk_c * yk_c' / skyk_c) +
        yk_c * yk_c' / skyk_c
    return d
end

# Inverse SR-1 update
function update_hessian!(
    d::QuasiNewtonMatrixDirectionUpdate{InverseSR1}, p, o, x_old, ::Int
)
    update_basis!(d.basis, p.M, x_old, o.x, d.vector_transport_method)
    yk_c = get_coordinates(p.M, o.x, o.yk, d.basis)
    sk_c = get_coordinates(p.M, o.x, o.sk, d.basis)

    # computing the new matrix which represents the approximating operator in the next iteration
    srvec = sk_c - d.matrix * yk_c
    if d.update.r < 0 || abs(dot(srvec, yk_c)) >= d.update.r * norm(srvec) * norm(yk_c)
        d.matrix = d.matrix + srvec * srvec' / (srvec' * yk_c)
    end
    return d
end

# SR-1 update
function update_hessian!(d::QuasiNewtonMatrixDirectionUpdate{SR1}, p, o, x_old, ::Int)
    update_basis!(d.basis, p.M, x_old, o.x, d.vector_transport_method)
    yk_c = get_coordinates(p.M, o.x, o.yk, d.basis)
    sk_c = get_coordinates(p.M, o.x, o.sk, d.basis)

    # computing the new matrix which represents the approximating operator in the next iteration
    srvec = yk_c - d.matrix * sk_c
    if d.update.r < 0 || abs(dot(srvec, sk_c)) >= d.update.r * norm(srvec) * norm(sk_c)
        d.matrix = d.matrix + srvec * srvec' / (srvec' * sk_c)
    end
    return d
end

# Inverse Broyden update
function update_hessian!(
    d::QuasiNewtonMatrixDirectionUpdate{InverseBroyden}, p, o, x_old, ::Int
)
    update_basis!(d.basis, p.M, x_old, o.x, d.vector_transport_method)
    yk_c = get_coordinates(p.M, o.x, o.yk, d.basis)
    sk_c = get_coordinates(p.M, o.x, o.sk, d.basis)
    skyk_c = inner(p.M, o.x, o.sk, o.yk)
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
function update_hessian!(d::QuasiNewtonMatrixDirectionUpdate{Broyden}, p, o, x_old, ::Int)
    update_basis!(d.basis, p.M, x_old, o.x, d.vector_transport_method)
    yk_c = get_coordinates(p.M, o.x, o.yk, d.basis)
    sk_c = get_coordinates(p.M, o.x, o.sk, d.basis)
    skyk_c = inner(p.M, o.x, o.sk, o.yk)
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
    d::QuasiNewtonCautiousDirectionUpdate{U}, p, o, x_old, iter
) where {U<:AbstractQuasiNewtonDirectionUpdate}
    # computing the bound used in the decission rule
    bound = d.θ(norm(p.M, o.x, o.gradient))
    sk_normsq = norm(p.M, o.x, o.sk)^2
    if sk_normsq != 0 && (inner(p.M, o.x, o.sk, o.yk) / sk_normsq) >= bound
        update_hessian!(d.update, p, o, x_old, iter)
    end
    return d
end

# Limited-memory update
function update_hessian!(
    d::QuasiNewtonLimitedMemoryDirectionUpdate{U}, p, o, x_old, ::Int
) where {U<:InverseBFGS}
    (capacity(d.memory_s) == 0) && return d
    # only transport the first if it does not get overwritten at the end
    start = length(d.memory_s) == capacity(d.memory_s) ? 2 : 1

    for i in start:length(d.memory_s)
        # transport all stored tangent vectors in the tangent space of the next iterate
        vector_transport_to!(
            p.M, d.memory_s[i], x_old, d.memory_s[i], o.x, d.vector_transport_method
        )
        vector_transport_to!(
            p.M, d.memory_y[i], x_old, d.memory_y[i], o.x, d.vector_transport_method
        )
    end

    # add newest
    push!(d.memory_s, o.sk)
    push!(d.memory_y, o.yk)
    return d
end

# all Cautious Limited Memory
function update_hessian!(
    d::QuasiNewtonCautiousDirectionUpdate{QuasiNewtonLimitedMemoryDirectionUpdate{NT,T,VT}},
    p,
    o,
    x_old,
    iter,
) where {NT<:AbstractQuasiNewtonUpdateRule,T,VT<:AbstractVectorTransportMethod}
    # computing the bound used in the decission rule
    bound = d.θ(norm(p.M, x_old, get_gradient(p, x_old)))
    sk_normsq = norm(p.M, o.x, o.sk)^2

    # if the decission rule is fulfilled, the new sk and yk are added
    if sk_normsq != 0 && (inner(p.M, o.x, o.sk, o.yk) / sk_normsq) >= bound
        update_hessian!(d.update, p, o, x_old, iter)
    else
        # the stored vectores are just transported to the new tangent space, sk and yk are not added
        for i in 1:length(d.update.memory_s)
            vector_transport_to!(
                p.M,
                d.update.memory_s[i],
                x_old,
                d.update.memory_s[i],
                o.x,
                d.update.vector_transport_method,
            )
            vector_transport_to!(
                p.M,
                d.update.memory_y[i],
                x_old,
                d.update.memory_y[i],
                o.x,
                d.update.vector_transport_method,
            )
        end
    end
    return d
end

get_solver_result(o::QuasiNewtonOptions) = o.x
