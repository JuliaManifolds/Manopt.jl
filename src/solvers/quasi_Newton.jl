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
    broyden_factor::Float64 = 0.0,
    cautious_update::Bool=false,
    cautious_function::Function = (x) -> x*10^(-4),
    memory_size::Int = 20,
    memory_steps::AbstractVector{T} = [zero_tangent_vector(M,x) for _ ∈ 1:memory_size],
    memory_gradients::AbstractVector{T} = [zero_tangent_vector(M,x) for _ ∈ 1:memory_size],
    memory_position::Int = 0,
	initial_operator::AbstractMatrix = Matrix(I,manifold_dimension(M), manifold_dimension(M)),
    scalling_initial_operator::Bool = true,
    step_size::Stepsize = WolfePowellLineseach(retraction_method, vector_transport_method),
    stopping_criterion::StoppingCriterion = StopWhenAny(
        StopAfterIteration(max(1000, memory_size)),
        StopWhenGradientNormLess(10^(-6))),
	return_options=false,
    kwargs...
) where {MT <: Manifold, P, T}

	(broyden_factor < 0. || broyden_factor > 1.) && throw( ErrorException( "broyden_factor must be in the interval [0,1], but it is $broyden_factor."))

	memory_steps_size = length(memory_steps)
	g = length(memory_gradients)

	(memory_steps_size != g) && throw( ErrorException( "The number of given vectors in memory_steps ($memory_steps_size) is different from the number of memory_gradients ($g)."))

	(memory_steps_size < memory_position) && throw( ErrorException( "The number of given vectors in memory_steps ($memory_steps_size) is too small compared to memory_position ($memory_position)."))

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


function initialize_solver!(p::GradientProblem,o::AbstractQuasiNewtonOptions)
end

function step_solver!(p::GradientProblem,o::AbstractQuasiNewtonOptions,iter)
	o.∇ = get_gradient(p,o.x)
	print("\n")
	# print("$(o.∇) \n")
	print("\n")
	η = get_quasi_newton_direction(p, o)
	print("$(o.∇ - get_gradient(p,o.x)) \n")
	α = o.stepsize(p,o,iter,η)

	x_old = o.x

	retract!(p.M, o.x, o.x, α*η, o.retraction_method)

	update_parameters!(p, o, α, η, x_old, iter)

end

"""
	get_quasi_newton_direction(p::GradientProblem, o::QuasiNewtonOptions)
	get_quasi_newton_direction(p::GradientProblem, o::CautiuosQuasiNewtonOptions)

Compute the quasi-Newton search direction.
"""

function get_quasi_newton_direction(p::GradientProblem, o::Union{QuasiNewtonOptions{P,T}, CautiuosQuasiNewtonOptions{P,T}}) where {P, T}
	gradient = get_coordinates(p.M,o.x,o.∇,o.basis)
	return get_vector(p.M,o.x,-o.inverse_hessian_approximation*gradient,o.basis)
end


"""
	get_quasi_newton_direction(p::GradientProblem, o::RLBFGSOptions)
	get_quasi_newton_direction(p::GradientProblem, o::CautiousRLBFGSOptions)

Compute the limited-memory RBFGS variant of the search direction using the two-loop
recursion [^HuangAbsilGallivan2006] (cf. Algorithm 7.4 of [^NocedalWright2006] for the Euclidean description)

[^NocedalWright2015]:
	> Nocedal, J., Wright, S. J.: Numerical Optimization, Springer New York, NY, 2006.
	> doi: [10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)

[^HuangAbsilGallivan2015]:
	>   Huang, W., Absil, P.-A., Gallivan, K. A.:
    > _A Broyden class of quasi-Newton methods for Riemannian optimization_,
	> SIAM Journal on Optimization (25.3), pp. 1660–1685, 2015.
	> doi: [10.1137/140955483](https://doi.org/10.1137/140955483)
	> PDF: [https://www.math.fsu.edu/~aluffi/archive/paper488.pdf](https://www.math.fsu.edu/~aluffi/archive/paper488.pdf)
"""
function get_quasi_newton_direction(p::GradientProblem, o::Union{RLBFGSOptions{P,T}, CautiuosRLBFGSOptions{P,T}}) where {P, T}
	r = o.∇
	current_memory = o.current_memory_size
	ξ = zeros(current_memory)
	ρ = zeros(current_memory)

	for i in current_memory : -1 : 1
		ρ[i] = 1 / inner(p.M, o.x, o.steps[i], o.gradient_diffrences[i])
		ξ[i] = inner(p.M, o.x, o.steps[i], r) * ρ[i]
		r .=  r .- ξ[i] .* o.gradient_diffrences[i]
	end

	if current_memory != 0
		r .= 1 / ( ρ[current_memory] * norm(p.M, o.x, o.gradient_diffrences[current_memory])^2) .* r
	end

	for i in 1 : current_memory
		ω = ρ[i]*inner(p.M, o.x, o.gradient_diffrences[i],r)
		r .= r .+ (ξ[i] - ω) .* o.steps[i]
	end
	project!(p.M, r, o.x, r)
	return -r
end


# Updating the parameters

function update_parameters!(p::GradientProblem, o::QuasiNewtonOptions{P,T}, α::Float64, η::T, x::P, iter) where {P,T}
	vector_transport_to!(p.M, o.∇, x, o.∇, o.x, o.vector_transport_method)
	sk = vector_transport_to(p.M, x, α*η, o.x, o.vector_transport_method)

	yk = get_gradient(p,o.x) - o.∇

	for i=1:length(o.basis.data)
		vector_transport_to!(p.M, o.basis.data[i], x, o.basis.data[i], o.x, o.vector_transport_method)
	end

	yk_c = get_coordinates(p.M, o.x, yk, o.basis)
	sk_c = get_coordinates(p.M, o.x, sk, o.basis)
	skyk_c = dot(sk_c, yk_c)

	if iter == 1 && o.scalling_initial_operator == true
		o.inverse_hessian_approximation = skyk_c/dot(yk_c, yk_c) * o.inverse_hessian_approximation
	end


	if o.broyden_factor==1.0
		o.inverse_hessian_approximation = o.inverse_hessian_approximation + sk_c * sk_c' / skyk_c - o.inverse_hessian_approximation * yk_c * (o.inverse_hessian_approximation * yk_c)' / dot(yk_c, o.inverse_hessian_approximation * yk_c)
	end

	if o.broyden_factor==0.0
		o.inverse_hessian_approximation = (I - sk_c * yk_c' / skyk_c) * o.inverse_hessian_approximation * (I - yk_c * sk_c' / skyk_c) + sk_c * sk_c' / skyk_c
	end

	if o.broyden_factor > 0 && o.broyden_factor < 1
		RBFGS = (I - sk_c * yk_c' / skyk_c) * o.inverse_hessian_approximation * (I - yk_c * sk_c' / skyk_c) + sk_c * sk_c' / skyk_c
		DFP = o.inverse_hessian_approximation + sk_c * sk_c' / skyk_c - o.inverse_hessian_approximation * yk_c * (o.inverse_hessian_approximation * yk_c)' / dot(yk_c, o.inverse_hessian_approximation * yk_c)
		o.inverse_hessian_approximation = (1 - o.broyden_factor) * RBFGS + o.broyden_factor * DFP
	end
	return o
end

function update_parameters!(p::GradientProblem, o::CautiuosQuasiNewtonOptions{P,T}, α::Float64, η::T, x::P, iter) where {P,T}
	vector_transport_to!(p.M, o.∇, x, o.∇, o.x, o.vector_transport_method)
	sk = vector_transport_to(p.M, x, α*η, o.x, o.vector_transport_method)

	yk = get_gradient(p,o.x) - o.∇
	for i= 1:length(o.basis.data)
		vector_transport_to!(p.M, o.basis.data[i], x, o.basis.data[i], o.x, o.vector_transport_method)
	end

	yk_c = get_coordinates(p.M, o.x, yk, o.basis)
	sk_c = get_coordinates(p.M, o.x, sk, o.basis)
	skyk_c = dot(sk_c, yk_c)
	sksk_c = dot(sk_c, sk_c)

	bound = o.cautious_function(norm(p.M, x, o.∇))

	if sksk_c != 0 && (skyk_c / sksk_c) >= bound

		if iter == 1 && o.scalling_initial_operator == true
			o.inverse_hessian_approximation = skyk_c/dot(yk_c, yk_c) * o.inverse_hessian_approximation
		end

		if o.broyden_factor==1.0
			o.inverse_hessian_approximation = o.inverse_hessian_approximation + sk_c * sk_c' / skyk_c - o.inverse_hessian_approximation * yk_c * (o.inverse_hessian_approximation * yk_c)' / dot(yk_c, o.inverse_hessian_approximation * yk_c)
		end

		if o.broyden_factor==0.0
			o.inverse_hessian_approximation = (I - sk_c * yk_c' / skyk_c) * o.inverse_hessian_approximation * (I - yk_c * sk_c' / skyk_c) + sk_c * sk_c' / skyk_c
		end

		if o.broyden_factor > 0 && o.broyden_factor < 1
			RBFGS = (I - sk_c * yk_c' / skyk_c) * o.inverse_hessian_approximation * (I - yk_c * sk_c' / skyk_c) + sk_c * sk_c' / skyk_c
			DFP = o.inverse_hessian_approximation + sk_c * sk_c' / skyk_c - o.inverse_hessian_approximation * yk_c * (o.inverse_hessian_approximation * yk_c)' / dot(yk_c, o.inverse_hessian_approximation * yk_c)
			o.inverse_hessian_approximation = (1 - o.broyden_factor) * RBFGS + o.broyden_factor * DFP
		end
	end
	return o
end


# Limited memory variants

function update_parameters!(p::GradientProblem, o::RLBFGSOptions{P,T}, α::Float64, η::T, x::P, iter) where {P,T}
	vector_transport_to!(p.M, o.∇, x, o.∇, o.x, o.vector_transport_method)
	sk = vector_transport_to(p.M, x, α*η, o.x, o.vector_transport_method)

	yk = get_gradient(p,o.x) - o.∇
    return limited_memory_update!(p,o,sk,yk,x)
end


function update_parameters!(p::GradientProblem, o::CautiuosRLBFGSOptions{P,T}, α::Float64, η::T, x::P, iter) where {P,T}
	vector_transport_to!(p.M, o.∇, x, o.∇, o.x, o.vector_transport_method)
	sk = vector_transport_to(p.M, x, α*η, o.x, o.vector_transport_method)
	beta = norm(p.M, x, α*η) / norm(p.M, o.x, sk)
	yk = get_gradient(p,o.x)/beta - o.∇

	sk_yk = inner(p.M, o.x, sk, yk)
	norm_sk = norm(p.M, o.x, sk)
	bound = o.cautious_function(norm(p.M, x, get_gradient(p,x)))

	if norm_sk != 0 && (sk_yk / norm_sk^2) >= bound
		limited_memory_update!(p,o,sk,yk,x)
	else
		memory_steps_size = length(o.steps)
        for  i = 1 : min(o.current_memory_size, memory_steps_size)
			vector_transport_to!(p.M, o.steps[i], x, o.steps[i], o.x, o.vector_transport_method)
			vector_transport_to!(p.M, o.gradient_diffrences[i], x, o.gradient_diffrences[i], o.x, o.vector_transport_method)
        end
	end
	return o
end

function limited_memory_update!(p::GradientProblem, o::Union{RLBFGSOptions{P,T}, CautiuosRLBFGSOptions{P,T}}, sk::T, yk::T, x::P) where {P,T}
	memory_steps_size = length(o.steps)
	current_memory = o.current_memory_size

    if current_memory >= memory_steps_size
        for  i in 2 : memory_steps_size
            vector_transport_to!(p.M, o.steps[i-1], x, o.steps[i], o.x, o.vector_transport_method)
            vector_transport_to!(p.M, o.gradient_diffrences[i-1], x, o.gradient_diffrences[i], o.x, o.vector_transport_method)
        end
        if memory_steps_size > 0
            o.steps[memory_steps_size] = sk
            o.gradient_diffrences[memory_steps_size] = yk
        end
    else
        for i in 1:current_memory
            vector_transport_to!(p.M, o.steps[i], x, o.steps[i], o.x, o.vector_transport_method)
            vector_transport_to!(p.M, o.gradient_diffrences[i], x, o.gradient_diffrences[i], o.x, o.vector_transport_method)
        end
        o.steps[current_memory + 1] = sk
        o.gradient_diffrences[current_memory + 1] = yk
        o.current_memory_size = current_memory + 1
	end
	return o
end

function operator_to_matrix(M::Manifold, x::P, operator::Function; basis::B = get_basis(M, x, DefaultOrthonormalBasis())) where {P,T,B<:AbstractBasis}
	orthonormal_basis = get_vectors(M, x, basis)
	n = length(orthonormal_basis)
	matrix_rep = zeros(n, n)
	column = [operator(orthonormal_basis[i]) for i ∈ 1:n]
	for i = 1:n
		matrix_rep[:,i] = get_coordinates(M, x, column[i], basis)
	end
	return matrix_rep
end
get_solver_result(o::O) where {O <: AbstractQuasiNewtonOptions} = o.x
