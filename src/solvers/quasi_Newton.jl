@doc raw"""
    quasi_Newton(M, F, ∇F, x)

evaluate a Riemannian quasi-Newton solver for optimization on manifolds.
It will attempt to minimize the cost function F on the Manifold M.

# Input
* `M` – a manifold $\mathcal{M}$.
* `F` – a cost function $F \colon \mathcal{M} \to \mathbb{R}$ to minimize.
* `∇F`– the gradient $\nabla F \colon \mathcal M \to \tangent{x}$ of $F$.
* `x` – an initial value $x \in \mathcal{M}$.

# Optional
* `retraction_method` – `retraction_method` – (`ExponentialRetraction()`) a retraction method to use, by default the exponntial map.
* `vector_transport_method` – (`ParallelTransport()`) a vector transport to use, by default the parallel transport.
* `broyden_factor` – (`0.`) – specifies the proportion of DFP update in the convex combination if the Broyden Class method is to be used. By default, BFGS is used.
* `cautious_update` – (`false`) – specifies whether a cautious update should be used, which means that a decision rule based on the calculated values decides whether the operator remains the same and no new information is received, or whether it is updated as usual.
* `cautious_function` – (`(x) -> x*10^(-4)`) – a monotone increasing function that is zero at 0 and strictly increasing at 0.
* `memory_size` – (`20`) – number of vectors to be stored.
* `memory_steps`– (`[`[`zero_tangent_vector`](@ref)`(M,x) for _ ∈ 1:memory_size]`) – the currently stored tangent vectors $s_k$ for a LRBFGS method.
* `memory_gradients` – (`[`[`zero_tangent_vector`](@ref)`(M,x) for _ ∈ 1:memory_size]`) – the currently stored tangent vectors $y_k$ for a LRBFGS method.
* `initial_operator` – (`Matrix(I, [`manifold_dimension`](@ref)`(M), [`manifold_dimension`](@ref)`(M))`) – the initial operator, represented as a matrix.
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
function quasi_Newton(
    M::MT,
    F::Function,
    ∇F::G,
    x::P;
    retraction_method::AbstractRetractionMethod=ExponentialRetraction(),
    vector_transport_method::AbstractVectorTransportMethod=ParallelTransport(),
    basis::AbstractBasis=DefaultOrthonormalBasis(),
    direction_update::AbstractQuasiNewtonType=InverseBFGS(),
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
) where {MT<:Manifold,P,G}
    if memory_size >= 0
        local_dir_upd = LimitedMemoryQuasiNewctionDirectionUpdate(
            direction_update,
            zero_tangent_vector(M, x),
            memory_size;
            scale=scalling_initial_operator,
            vector_transport_method=vector_transport_method,
        )
    else
        local_dir_upd = QuasiNewtonDirectionUpdate(
            direction_update,
            basis,
            initial_operator;
            scale=scalling_initial_operator,
            vector_transport_method=vector_transport_method,
        )
    end
    if cautious_update
        local_dir_upd = CautiousUpdate(local_dir_upd; θ=cautious_function)
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
    # compute serach direction
    η = o.direction_update(p, o)
    # compute stepsize
    α = o.stepsize(p, o, iter, η)
    x_old = deepcopy(o.x)
    # compute next iterate
    retract!(p.M, o.x, o.x, α * η, o.retraction_method)
    # compute locking condition parameter
    β = locking_condition_scale(
        p.M, o.direction_update, x_old, α * η, o.x, o.vector_transport_method
    )
    # compute new sk
    vector_transport_to!(p.M, o.sk, x_old, α * η, o.x, o.vector_transport_method)
    # reuse ∇
    vector_transport_to!(p.M, o.∇, x_old, o.∇, o.x, o.vector_transport_method)
    # compute new yk
    o.yk = get_gradient(p, o.x) / β - o.∇
    # update of the approximating operator
    update_hessian!(o.direction_update, p, o, x_old, iter)
    return o
end

function locking_condition_scale(M::Manifold, ::AbstractQuasiNewtonDirectionUpdate, x_old, v, x, vt)
    return norm(M, x_old, v) / norm(M, x, vector_transport_to(M, x_old, v, x, vt))
end


# Inverese BFGS update 
function update_hessian!(d::QuasiNewtonDirectionUpdate{InverseBFGS}, p, o, x_old, iter)
    # transport orthonormal basis in new tangent space
    update_basis!(d.basis, p.M, x_old, o.x, d.vector_transport_method)
    # compute coordinates of yk
    yk_c = get_coordinates(p.M, o.x, o.yk, d.basis)
    # compute coordinates of sk
    sk_c = get_coordinates(p.M, o.x, o.sk, d.basis)
    # compute real-valued inner product of sk and yk
    skyk_c = inner(p.M, o.x, o.sk, o.yk)
    
    # scaling the matrix before the first update is done
    if iter == 1 && d.scale == true
        d.matrix = skyk_c / inner(p.M, o.x, o.yk, o.yk) * d.matrix
    end

    # computing the new matrix which represents the approximating operator in the next iteration
    return d.matrix =
        (I - sk_c * yk_c' / skyk_c) * d.matrix * (I - yk_c * sk_c' / skyk_c) +
        sk_c * sk_c' / skyk_c
end

# BFGS update 
function update_hessian!(d::QuasiNewtonDirectionUpdate{BFGS}, p, o, x_old, iter)
    # transport orthonormal basis in new tangent space
    update_basis!(d.basis, p.M, x_old, o.x, d.vector_transport_method)
    # compute coordinates of yk
    yk_c = get_coordinates(p.M, o.x, o.yk, d.basis)
    # compute coordinates of sk
    sk_c = get_coordinates(p.M, o.x, o.sk, d.basis)
    # compute real-valued inner product of sk and yk
    skyk_c = inner(p.M, o.x, o.sk, o.yk)
    
    # scaling the matrix before the first update is done
    if iter == 1 && d.scale == true
        d.matrix = inner(p.M, o.x, o.yk, o.yk) / skyk_c * d.matrix
    end

    # computing the new matrix which represents the approximating operator in the next iteration
    return d.matrix =
        d.matrix + yk_c * yk_c' / skyk_c -
        d.matrix * sk_c * sk_c' * d.matrix / dot(sk_c, d.matrix * sk_c)
end

# Inverese DFP update 
function update_hessian!(d::QuasiNewtonDirectionUpdate{InverseDFP}, p, o, x_old, iter)
    # transport orthonormal basis in new tangent space
    update_basis!(d.basis, p.M, x_old, o.x, d.vector_transport_method)
    # compute coordinates of yk
    yk_c = get_coordinates(p.M, o.x, o.yk, d.basis)
    # compute coordinates of sk
    sk_c = get_coordinates(p.M, o.x, o.sk, d.basis)
    # compute real-valued inner product of sk and yk
    skyk_c = inner(p.M, o.x, o.sk, o.yk)
    
    # scaling the matrix before the first update is done
    if iter == 1 && d.scale == true
        d.matrix = inner(p.M, o.x, o.sk, o.sk) / skyk_c * d.matrix
    end

    # computing the new matrix which represents the approximating operator in the next iteration
    return d.matrix =
        d.matrix + sk_c * sk_c' / skyk_c -
        d.matrix * yk_c * yk_c' * d.matrix / dot(yk_c, d.matrix * yk_c)
end

# DFP update 
function update_hessian!(d::QuasiNewtonDirectionUpdate{DFP}, p, o, x_old, iter)
    # transport orthonormal basis in new tangent space
    update_basis!(d.basis, p.M, x_old, o.x, d.vector_transport_method)
    # compute coordinates of yk
    yk_c = get_coordinates(p.M, o.x, o.yk, d.basis)
    # compute coordinates of sk
    sk_c = get_coordinates(p.M, o.x, o.sk, d.basis)
    # compute real-valued inner product of sk and yk
    skyk_c = inner(p.M, o.x, o.sk, o.yk)

    # scaling the matrix before the first update is done
    if iter == 1 && d.scale == true
        d.matrix = skyk_c / inner(p.M, o.x, o.sk, o.sk) * d.matrix
    end

    # computing the new matrix which represents the approximating operator in the next iteration
    return d.matrix =
        (I - yk_c * sk_c' / skyk_c) * d.matrix * (I - sk_c * yk_c' / skyk_c) +
        yk_c * yk_c' / skyk_c
end

# Inverse SR-1 update
function update_hessian!(d::QuasiNewtonDirectionUpdate{InverseSR1}, p, o, x_old, iter)
    # transport orthonormal basis in new tangent space
    update_basis!(d.basis, p.M, x_old, o.x, d.vector_transport_method)
    # compute coordinates of yk
    yk_c = get_coordinates(p.M, o.x, o.yk, d.basis)
    # compute coordinates of sk
    sk_c = get_coordinates(p.M, o.x, o.sk, d.basis)
    # compute real-valued inner product of sk and yk
    skyk_c = inner(p.M, o.x, o.sk, o.yk)

    # scaling the matrix before the first update is done
    if iter == 1 && d.scale == true
        d.matrix = skyk_c / norm(p.M, o.x, o.yk)^2 * d.matrix
    end

    # computing the new matrix which represents the approximating operator in the next iteration
    return d.matrix =
        d.matrix +
        (sk_c - d.matrix * yk_c) * (sk_c - d.matrix * yk_c)' / (sk_c - d.matrix * yk_c)' *
        yk_c
end

# SR-1 update
function update_hessian!(d::QuasiNewtonDirectionUpdate{SR1}, p, o, x_old, iter)
    # transport orthonormal basis in new tangent space
    update_basis!(d.basis, p.M, x_old, o.x, d.vector_transport_method)
    # compute coordinates of yk
    yk_c = get_coordinates(p.M, o.x, o.yk, d.basis)
    # compute coordinates of sk
    sk_c = get_coordinates(p.M, o.x, o.sk, d.basis)
    # compute real-valued inner product of sk and yk
    skyk_c = inner(p.M, o.x, o.sk, o.yk)

    # scaling the matrix before the first update is done
    if iter == 1 && d.scale == true
        d.matrix = skyk_c / norm(p.M, o.x, o.yk)^2 * d.matrix
    end

    # computing the new matrix which represents the approximating operator in the next iteration
    return d.matrix =
        d.matrix +
        (yk_c - d.matrix * sk_c) * (yk_c - d.matrix * sk_c)' / (yk_c - d.matrix * sk_c)' *
        sk_c
end

function update_basis!(
    b::AbstractBasis, ::Manifold, ::P, ::P, ::AbstractVectorTransportMethod
) where {P}
    return b
end

function update_basis!(
    b::CachedBasis, M::Manifold, x::P, y::P, v::AbstractVectorTransportMethod
) where {P}
    # transport all basis tangent vectors in the tangent space of the next iterate
    for i in 1:length(b.data)
        vector_transport_to!(M, b.data[i], y, b.data[i], x, v)
    end
    return b
end

# Cautious update
function update_hessian!(
    d::CautiousUpdate{U}, p, o, x_old, iter
) where {U<:AbstractQuasiNewtonDirectionUpdate}
    # computing the bound used in the decission rule
    bound = d.θ(norm(p.M, o.x, o.∇))
    sk_normsq = norm(p.M, o.x, o.sk)^2

    # if the decission rule is fulfilled, the operator is updated as usual
    if sk_normsq != 0 && (inner(p.M, o.x, o.sk, o.yk) / sk_normsq) >= bound
        update_hessian!(d.update, p, o, x_old, iter)
    end

    return d
end

# Limited-memory update
function update_hessian!(
    d::LimitedMemoryQuasiNewctionDirectionUpdate{U}, p, o, x_old, iter
) where {U<:AbstractQuasiNewtonType}
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
    return push!(d.memory_y, o.yk)
end

# all Cautious Limited Memory
function update_hessian!(
    d::CautiousUpdate{LimitedMemoryQuasiNewctionDirectionUpdate{U}}, p, o, x_old, iter
) where {U<:AbstractQuasiNewtonType}
    # computing the bound used in the decission rule
    bound = d.θ(norm(p.M, x_old, get_gradient(p, x_old)))
    sk_normsq = norm(p.M, o.x, o.sk)^2

    # if the decission rule is fulfilled, the new sk and yk are added
    if sk_normsq != 0 && (inner(p.M, o.x, o.sk, o.yk) / sk_normsq) >= bound
        update_hessian!(d.update, p, o, x_old, iter)
    else
        # the stored vectores are just transported to the new tangent space, sk and yk are not added
        for i in 1:length(d.memory_s)
            vector_transport_to!(
                p.M, d.memory_s[i], x_old, d.memory_s[i], o.x, d.vector_transport_method
            )
            vector_transport_to!(
                p.M, d.memory_y[i], x_old, d.memory_y[i], o.x, d.vector_transport_method
            )
        end
    end
    return d
end
function update_hessian!(d::Broyden, p, o, x_old, iter)
    update_hessian!(d.update1, p, o, x_old, iter)
    update_hessian!(d.update2, p, o, x_old, iter)
    return d
end



get_solver_result(o::QuasiNewtonOptions) = o.x
