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
    ∇F::Function,
	x::P;
    retraction_method::AbstractRetractionMethod = ExponentialRetraction(),
	vector_transport_method::AbstractVectorTransportMethod = ParallelTransport(),
	basis::AbstractBasis = DefaultOrthonormalBasis()
	direction_update::AbstractQuasiNewtonDirectionUpdate=InverseBFGS()
    cautious_update::Bool=false,
    cautious_function::Function = (x) -> x*10^(-4),
    memory_size::Int = 20,
	initial_operator::AbstractMatrix = Matrix(I,manifold_dimension(M), manifold_dimension(M)),
    scalling_initial_operator::Bool = true,
    step_size::Stepsize = WolfePowellLineseach(retraction_method, vector_transport_method),
    stopping_criterion::StoppingCriterion = StopWhenAny(
        StopAfterIteration(max(1000, memory_size)),
        StopWhenGradientNormLess(10^(-6))),
	return_options=false,
    kwargs...
) where {MT <: Manifold, P, T}
	if memory_size >= 0
		local_dir_upd = LimitedMemoryQuasiNewctionDirectionUpdate(
			direction_update,
			zero_tangent_vector(M,x),
			memory_size;
			scale = scalling_initial_operator,
			vector_transport_method = vector_transport_method
		)
	else
		local_dir_upd = QuasiNewtonDirectionUpdate(
			direction_update,
			basis,
			initial_operator;
			scale = scalling_initial_operator,
			vector_transport_method = vector_transport_method
		)
	end
	if cautious_update
		local_dir_upd = CautiousUpdate(
			local_dir_upd;
			φ = cautious_function
		)
	end
	o = QuasiNewtonOptions(x,∇F(x),
		direction_update,
		stop,
		step_size,
		retraction_method,
		vector_transport_method
	)
	grad_x = ∇F(x)
	if memory_size < 0 && memory_steps_size == 0
		basis = get_basis(M, x, DefaultOrthonormalBasis())
		if cautious_update == true
			o = CautiuosQuasiNewtonOptions(x, grad_x, initial_operator, basis; scalling_initial_operator = scalling_initial_operator, cautious_function = cautious_function, retraction_method = retraction_method, vector_transport_method = vector_transport_method, stop = stopping_criterion, stepsize = step_size, broyden_factor = broyden_factor)
		else
			o = QuasiNewtonOptions(x, grad_x, initial_operator, basis; scalling_initial_operator = scalling_initial_operator, retraction_method = retraction_method, vector_transport_method = vector_transport_method, stop = stopping_criterion, stepsize = step_size, broyden_factor = broyden_factor)
		end
	else
		if cautious_update == true
			o = CautiuosRLBFGSOptions(x, grad_x, memory_gradients, memory_steps; cautious_function = cautious_function, current_memory_size = memory_position, retraction_method = retraction_method, vector_transport_method = vector_transport_method, stop = stopping_criterion, stepsize = step_size)
		else
			o = RLBFGSOptions(x, grad_x, memory_gradients, memory_steps; current_memory_size = memory_position, retraction_method = retraction_method, vector_transport_method = vector_transport_method, stop = stopping_criterion, stepsize = step_size)
		end
	end

	p = GradientProblem(M,F,∇F)

	o = decorate_options(o; kwargs...)
	resultO = solve(p,o)

	if return_options
		return resultO
	else
		return get_solver_result(resultO)
	end
end


function initialize_solver!(::GradientProblem,::QuasiNewtonOptions)
end

function step_solver!(p::GradientProblem,o::QuasiNewtonOptions,iter)
	o.∇ = get_gradient(p,o.x)
	η = o.direction_update(p, o)
	α = o.stepsize(p,o,iter,η)
	x_old = o.x
	retract!(p.M, o.x, o.x, α*η, o.retraction_method)
	# update sk
	vector_transport_to!(p.M, o.sk, x_old, α*η, o.x, o.vector_transport_method)
	# reuse ∇
	vector_transport_to!(p.M, o.∇, x_old, o.∇, o.x, o.vector_transport_method)
	o.yk = get_gradient(p,o.x) - o.∇
	update_hessian!(o.d, p, o, x_old, iter)
end
# update the HEssian representation
function update_hessian(d::QuasiNewtonDirectionUpdate{InverseBFGS}, p, o, x_old, iter)
	for i=1:length(o.basis.data)
		vector_transport_to!(p.M, o.basis.data[i], x_old, o.basis.data[i], o.x, o.vector_transport_method)
	end
	yk_c = get_coordinates(p.M, o.x, o.yk, o.basis)
	sk_c = get_coordinates(p.M, o.x, o.sk, o.basis)
	skyk_c = dot(sk_c, yk_c)

	if iter == 1 && o.scalling_initial_operator == true
		d.matrix = skyk_c/dot(yk_c, yk_c) * o.matrix
	end
	d.matrix = (I - sk_c * yk_c' / skyk_c) * o.matrix * (I - yk_c * sk_c' / skyk_c) + sk_c * sk_c' / skyk_c
end
function update_hessian(d::QuasiNewtonDirectionUpdate{InverseDFP}, p, o, x_old, iter)
	for i=1:length(o.basis.data)
		vector_transport_to!(p.M, o.basis.data[i], x_old, o.basis.data[i], o.x, o.vector_transport_method)
	end
	yk_c = get_coordinates(p.M, o.x, o.yk, o.basis)
	sk_c = get_coordinates(p.M, o.x, o.sk, o.basis)
	skyk_c = inner(p.M, o.x, o.sk, o.yk)

	if iter == 1 && o.scalling_initial_operator == true
		d.matrix = skyk_c/norm(p.M, o.x, o.yk)^2 * o.matrix
	end
	d.matrix = o.matrix + sk_c * sk_c' / skyk_c - o.matrix * yk_c * (o.matrix * yk_c)' / dot(yk_c, o.matrix * yk_c)
end
# all matrix cautious ones
function update_hessian!(d::CautiousUpdate{U}, p, o, x_old, iter) where {U <: AbstractQuasiNewtonDirectionUpdate}
	bound = d.φ(norm(p.M, o.x, o.∇))
	sk_normsq = norm(p.M, o.x, o.sk)^2
	if sk_normsq != 0 && (inner(p.M, o.x, o.sk, o.yk) / sk_normsq) >= bound
		update_hessian!(d.update, p, o, x_old, iter)
	end
	return d
end
# all limited memory updates
function update_hessian!(d::LimitedMemoryQuasiNewctionDirectionUpdate{U}, p, o, x_old, iter) where{U <: AbstractQuasiNewtonType}
	(d.memory_size==0) && return d
	if d.memory_size == length(d.sk_memory)
		for  i in 2 : d.memory_size
			vector_transport_to!(p.M, d.sk_memory[i-1], x_old, d.sk_memory[i], o.x, d.vector_transport_method)
            vector_transport_to!(p.M, o.yk_memory[i-1], x_old, d.yk_memory[i], o.x, d.vector_transport_method)
        end
        d.sk_memory[memory_steps_size] .= o.sk
        d.yk_memory[memory_steps_size] .= o.yk
    else
        for i in 1:d.memory_size
            vector_transport_to!(p.M, d.sk_memory[i], x, d.sk_memory[i], o.x, d.vector_transport_method)
            vector_transport_to!(p.M, d.yk_memory[i], x, d.yk_memory[i], o.x, o.vector_transport_method)
        end
		d.memory_size +=1
		d.steps[d.memory_size] = o.sk
        d.gradient_diffrences[d.memory_size] = o.yk
	end
end
# all Cautious Limited Memory
function update_hessian!(d::CautiousUpdate{LimitedMemoryQuasiNewctionDirectionUpdate{U}}, p, o, x_old, iter) where {U<:AbstractQuasiNewtonType}
	bound = d.φ(norm(p.M, x_old, get_gradient(p, x_old)))
	sk_normsq = norm(p.M, o.x, o.sk)^2

	if sk_normsq != 0 && (inner(p.M, o.x, o.sk, o.yk) / sk_normsq) >= bound
		update_hessian(d.update, p, o, x_old, iter)
	else # just PT but do not save
		for  i = 1 : d.update.memory_size
			vector_transport_to!(p.M, d.update.sk_memory[i], x_old, d.update.sk_memory[i], o.x, o.vector_transport_method)
			vector_transport_to!(p.M, d.update.yk_memory[i], x_old, o.update.yk_memory[i], o.x, o.vector_transport_method)
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
