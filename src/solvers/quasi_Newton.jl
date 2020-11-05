@doc raw"""
    quasi_Newton(M, F, ∇F, x)

evaluate a Riemannian quasi-Newton solver for optimization on manifolds.
It will attempt to minimize the cost function F on the Manifold M.

# Input
* `M` – a manifold $\mathcal{M}$
* `F` – a cost function $F \colon \mathcal{M} \to \mathbb{R}$ to minimize
* `∇F`- the gradient $\nabla F \colon \mathcal M \to \tangent{x}$ of $F$
* `x` – an initial value $x \in \mathcal{M}$

# Optional

# Output
* `x` – the last reached point on the manifold

# see also

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
	η = get_quasi_newton_direction(p, o)

	α = o.stepsize(p,o,iter,η)

	x_old = o.x

	o.x = retract(p.M, o.x, α*η, o.retraction_method)

	update_parameters(p, o, α, η, x_old, iter)

end

# Computing the direction

function get_quasi_newton_direction(p::GradientProblem, o::Union{QuasiNewtonOptions{P,T}, CautiuosQuasiNewtonOptions{P,T}}) where {P, T}
	gradient = get_coordinates(p.M,o.x,o.∇,o.basis)
	return get_vector(p.M,o.x,-o.inverse_hessian_approximation*gradient,o.basis)
end


function get_quasi_newton_direction(p::GradientProblem, o::Union{RLBFGSOptions{P,T}, CautiuosRLBFGSOptions{P,T}}) where {P, T}
	q = o.∇
	current_memory = o.current_memory_size
	ξ = zeros(current_memory)

	for i in current_memory : -1 : 1
		ξ[i] = inner(p.M, o.x, o.steps[i], q) / inner(p.M, o.x, o.steps[i], o.gradient_diffrences[i])
		q =  q - ξ[i]*o.gradient_diffrences[i]
	end

	if current_memory == 0
		r = q
	else
		r = (inner(p.M, o.x, o.steps[current_memory], o.gradient_diffrences[current_memory]) / norm(p.M, o.x, o.gradient_diffrences[current_memory])^2) * q
	end

	for i in 1 : current_memory
		ω = inner(p.M, o.x, o.gradient_diffrences[i], r) / inner(p.M, o.x, o.steps[i], o.gradient_diffrences[i])
		r = r + (ξ[i] - ω) * o.steps[i]
	end

	r = project(p.M, o.x, r)

	return -r
end


# Updating the parameters

function update_parameters(p::GradientProblem, o::QuasiNewtonOptions{P,T}, α::Float64, η::T, x::P, iter) where {P,T}
	gradf_xold = o.∇
	β = norm(p.M, x, α*η) / norm(p.M, o.x, vector_transport_to(p.M, x, α*η, o.x, o.vector_transport_method))
	yk = get_gradient(p,o.x)/β - vector_transport_to(p.M, x, gradf_xold, o.x, o.vector_transport_method)
	sk = vector_transport_to(p.M, x, α*η, o.x, o.vector_transport_method)

	o.basis.data .= [ vector_transport_to(p.M, x, v, o.x, o.vector_transport_method) for v ∈ get_vectors(p.M,o.x,o.basis) ]

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
end

function update_parameters(p::GradientProblem, o::CautiuosQuasiNewtonOptions{P,T}, α::Float64, η::T, x::P, iter) where {P,T}
	gradf_xold = o.∇
	β = norm(p.M, x, α*η) / norm(p.M, x, vector_transport_to(p.M, x, α*η, o.x, o.vector_transport_method))
	yk = get_gradient(p,o.x)/β - vector_transport_to(p.M, x, gradf_xold, o.x, o.vector_transport_method)
	sk = vector_transport_to(p.M, x, α*η, o.x, o.vector_transport_method)

	o.basis.data .= [ vector_transport_to(p.M, x, v, o.x, o.vector_transport_method) for v ∈ get_vectors(p.M,o.x,o.basis) ]

	yk_c = get_coordinates(p.M, o.x, yk, o.basis)
	sk_c = get_coordinates(p.M, o.x, sk, o.basis)
	skyk_c = dot(sk_c, yk_c)
	sksk_c = dot(sk_c, sk_c)

	bound = o.cautious_function(norm(p.M, x, gradf_xold))

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

end


# Limited memory variants

function update_parameters(p::GradientProblem, o::RLBFGSOptions{P,T}, α::Float64, η::T, x::P, iter) where {P,T}
    limited_memory_update(p,o,α,η,x)
end


function update_parameters(p::GradientProblem, o::CautiuosRLBFGSOptions{P,T}, α::Float64, η::T, x::P, iter) where {P,T}
	gradf_xold = o.∇
    β = norm(p.M, x, α*η) / norm(p.M, o.x, vector_transport_to(p.M, x, α*η, o.x, o.vector_transport_method))
    yk = get_gradient(p,o.x)/β - vector_transport_to(p.M, x, gradf_xold, o.x, o.vector_transport_method)
    sk = vector_transport_to(p.M, x, α*η, o.x, o.vector_transport_method)

	sk_yk = inner(p.M, o.x, sk, yk)
	norm_sk = norm(p.M, o.x, sk)
	bound = o.cautious_function(norm(p.M, x, get_gradient(p,x)))

	if norm_sk != 0 && (sk_yk / norm_sk) >= bound
		limited_memory_update(p,o,α,η,x)
	else
		memory_steps_size = length(o.steps)
        for  i = 1 : min(o.current_memory_size, memory_steps_size)
			o.steps[i] = vector_transport_to(p.M, x, o.steps[i], o.x, o.vector_transport_method)
			o.gradient_diffrences[i] = vector_transport_to(p.M, x, o.gradient_diffrences[i], o.x, o.vector_transport_method)
        end
	end
end

function limited_memory_update(p::GradientProblem, o::Union{RLBFGSOptions{P,T}, CautiuosRLBFGSOptions{P,T}}, α::Float64, η::T, x::P) where {P,T}
	gradf_xold = o.∇
    β = norm(p.M, x, α*η) / norm(p.M, o.x, vector_transport_to(p.M, x, α*η, o.x, o.vector_transport_method))
    yk = get_gradient(p,o.x)/β - vector_transport_to(p.M, x, gradf_xold, o.x, o.vector_transport_method)
    sk = vector_transport_to(p.M, x, α*η, o.x, o.vector_transport_method)

	memory_steps_size = length(o.steps)
	current_memory = o.current_memory_size

    if current_memory >= memory_steps_size
        for  i in 2 : memory_steps_size
            o.steps[i] = vector_transport_to(p.M, x, o.steps[i], o.x, o.vector_transport_method)
            o.gradient_diffrences[i] = vector_transport_to(p.M, x, o.gradient_diffrences[i], o.x, o.vector_transport_method)
        end

        if memory_steps_size > 1
			o.steps[1:(memory_steps_size-1)] = o.steps[2:memory_steps_size]
			o.gradient_diffrences[1:(memory_steps_size-1)] = o.gradient_diffrences[2:memory_steps_size]
        end

        if memory_steps_size > 0
            o.steps[memory_steps_size] = sk
            o.gradient_diffrences[memory_steps_size] = yk
        end
    else

        for i in 1:current_memory
            o.steps[i] = vector_transport_to(p.M, x, o.steps[i], o.x, o.vector_transport_method)
            o.gradient_diffrences[i] = vector_transport_to(p.M, x, o.gradient_diffrences[i], o.x, o.vector_transport_method)
        end

        o.steps[current_memory + 1] = sk
        o.gradient_diffrences[current_memory + 1] = yk

        o.current_memory_size = current_memory + 1
    end
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
