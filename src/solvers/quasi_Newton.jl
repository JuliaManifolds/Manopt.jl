@doc raw"""
    quasi_Newton(M, F, ∇F, x)

evaluate a Riemannian quasi-Newton solver for optimization on manifolds.
It will attempt to minimize the cost function F on the Manifold M.

# Input
* `M` – a manifold ``\mathcal{M}``.
* `F` – a cost function ``F \colon \mathcal{M} \to ℝ`` to minimize.
* `∇F`– the gradient ``∇F \colon \mathcal{M} \to T_x\mathcal M`` of ``F``.
* `x` – an initial value ``x \in \mathcal{M}``.

# Optional
* `retraction_method` – (`ExponentialRetraction()`) a retraction method to use, by default the exponntial map.
* `vector_transport_method` – (`ParallelTransport()`) a vector transport to use, by default the parallel transport.
* `broyden_factor` – (`0.`) – specifies the proportion of DFP update in the convex combination if the Broyden Class method is to be used. By default, BFGS is used.
* `cautious_update` – (`false`) – specifies whether a cautious update should be used, which means that a decision rule based on the calculated values decides whether the operator remains the same and no new information is received, or whether it is updated as usual.
* `cautious_function` – (`(x) -> x*10^(-4)`) – a monotone increasing function that is zero at 0 and strictly increasing at 0.
* `memory_size` – (`20`) – number of vectors to be stored.
* `memory_steps`– (`[`zero_tangent_vector(M,x) for _ ∈ 1:memory_size]`) – the currently stored tangent vectors $s_k$ for a LRBFGS method.
* `memory_gradients` – (`[`zero_tangent_vector(M,x) for _ ∈ 1:memory_size]`) – the currently stored tangent vectors $y_k$ for a LRBFGS method.
* `initial_operator` – (`Matrix(I, manifold_dimension(M), manifold_dimension(M))`) – the initial operator, represented as a matrix.
* `scalling_initial_operator` – (`true`) specifies if the initial operator is scalled after the first step but before the first update.
* `step_size` – ([`WolfePowellLineseach`](@ref)`(retraction_method, vector_transport_method)`) specify a [`Stepsize`](@ref) functor.
* `stopping_criterion`– ([`StopWhenAny`](@ref)`(`[`StopAfterIteration`](@ref)`(max(1000, memory_size)), `[`StopWhenGradientNormLess`](@ref)`(10.0^68))`)
* `return_options` – (`false`) – if activated, the extended result, i.e. the
    complete [`Options`](@ref) are returned. This can be used to access recorded values.
    If set to false (default) just the optimal value `x_opt` if returned

# Output
* `x_opt` – the resulting (approximately critical) point of the quasi–Newton method
OR
* `options` – the options returned by the solver (see `return_options`)

"""
function quasi_Newton(M::Manifold, F::Function, ∇F::G, x::P; kwargs...) where {P,G}
    x_res = allocate(x)
    copyto!(x_res, x)
    return quasi_Newton!(M, F, ∇F, x_res; kwargs...)
end
@doc raw"""
    quasi_Newton!(M, F, ∇F, x; options...)

evaluate a Riemannian quasi-Newton solver for optimization on manifolds.
It will attempt to minimize the cost function F on the Manifold M.
This method works in-place in `x`.

# Input
* `M` – a manifold ``\mathcal{M}``.
* `F` – a cost function ``F \colon \mathcal{M} \to ℝ`` to minimize.
* `∇F`– the gradient ``∇F \colon \mathcal{M} \to T_x\mathcal M`` of ``F``.
* `x` – an initial value ``x \in \mathcal{M}``.

For all optional parameters, see [`quasi_Newton`](@ref).
"""
function quasi_Newton!(
    M::Manifold,
    F::Function,
    ∇F::G,
    x::P;
    retraction_method::AbstractRetractionMethod=ExponentialRetraction(),
    vector_transport_method::AbstractVectorTransportMethod=ParallelTransport(),
    basis::AbstractBasis=DefaultOrthonormalBasis(),
    direction_update::AbstractQuasiNewtonUpdateRule=InverseBFGS(),
    cautious_update::Bool=false,
    cautious_function::Function=x -> x * 10^(-4),
    memory_size::Int=20,
    initial_operator::AbstractMatrix=Matrix{Float64}(
        I, manifold_dimension(M), manifold_dimension(M)
    ),
    scalling_initial_operator::Bool=true,
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
            zero_tangent_vector(M, x),
            memory_size;
            scale=scalling_initial_operator,
            vector_transport_method=vector_transport_method,
        )
    else
        local_dir_upd = QuasiNewtonMatrixDirectionUpdate(
            direction_update,
            basis,
            initial_operator;
            scale=scalling_initial_operator,
            vector_transport_method=vector_transport_method,
        )
    end
    if cautious_update == true
        local_dir_upd = QuasiNewtonCautiousDirectionUpdate(
            local_dir_upd; θ=cautious_function
        )
    end

    o = QuasiNewtonOptions(
        x,
        ∇F(x),
        local_dir_upd,
        stopping_criterion,
        step_size;
        retraction_method=retraction_method,
        vector_transport_method=vector_transport_method,
    )
    p = GradientProblem(M, F, ∇F)

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
    o.∇ = get_gradient(p, o.x)
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
        p.M, o.∇, x_old, o.∇, o.x, get_update_vector_transport(o.direction_update)
    )
    o.yk = get_gradient(p, o.x) / β - o.∇
    update_hessian!(o.direction_update, p, o, x_old, iter)
    return o
end

function locking_condition_scale(
    M::Manifold, ::AbstractQuasiNewtonDirectionUpdate, x_old, v, x, vt
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

# Inverese DFP update
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
    if d.update.r < 0 || dot(srvec, yk_c) >= d.update.r * norm(srvec) * norm(yk_c)
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
    if d.update.r < 0 || dot(srvec, sk_c) >= d.update.r * norm(srvec) * norm(sk_c)
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
    if skyk_c <= 2 * (skHksk_c * ykyk_c_c) / (skHksk_c + ykyk_c_c)
        return d.update.φ =
            (skyk_c * (ykyk_c_c - skyk_c)) / (skHksk_c * ykyk_c_c - skyk_c^2)
    else
        return d.update.φ = skyk_c / (skyk_c - skHksk_c)
    end
end

function update_broyden_factor!(d, sk_c, ::Any, skyk_c, ykBkyk_c, ::Val{:InverseDavidon})
    sk_c_c = d.matrix \ sk_c
    sksk_c_c = sk_c' * sk_c_c
    if skyk_c <= 2 * (ykBkyk_c * sksk_c_c) / (ykBkyk_c + sksk_c_c)
        return d.update.φ =
            (skyk_c * (sksk_c_c - skyk_c)) / (ykBkyk_c * sksk_c_c - skyk_c^2)
    else
        return d.update.φ = skyk_c / (skyk_c - ykBkyk_c)
    end
end

function update_basis!(
    b::AbstractBasis, ::Manifold, ::P, ::P, ::AbstractVectorTransportMethod
) where {P}
    return b
end

function update_basis!(
    b::CachedBasis, M::Manifold, x::P, y::P, m::AbstractVectorTransportMethod
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
    bound = d.θ(norm(p.M, o.x, o.∇))
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
