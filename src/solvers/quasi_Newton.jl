@doc raw"""
    quasi_Newton(M, F, ∇F, x, H, )

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
    x,
    H::Union{Function,Missing}
    ) where {MT <: Manifold}


    if return_options
        return resultO
    else
        return get_solver_result(resultO)
    end
end


function initialize_solver!(p::GradientProblem,o::QuasiNewtonOptions)
    o.steps = [zero_tangent_vector(p.M,o.x) for i ∈ 1:o.memory_size]
    o.gradient_differences = [zero_tangent_vector(p.M,o.x) for i ∈ 1:o.memory_size]
end


function step_solver!(p::GradientProblem,o::QuasiNewtonOptions,iter)
        # Compute BFGS direction
        η = get_quasi_newton_direction(p, o)

        # Execute line-search
        α = o.stepsize(p,o,iter,η)
        print(α," ")
        # Compute Step
        x_old = o.x
        o.x = retract(p.M, o.x, α*η, o.retraction_method)
        # Update the Parameters
        update_parameters(p, o, α, η, x_old)
end

# Computing the direction

function get_quasi_newton_direction(p::GradientProblem, o::QuasiNewtonOptions)
        o.∇ = get_gradient(p,o.x)
        return square_matrix_vector_product(p.M, o.x, o.inverse_hessian_approximation, -o.∇)
end

# Limited memory variants

function get_quasi_newton_direction(p::GradientProblem, o::RLBFGSOptions)
        q = get_gradient(p,o.x)
        o.current_memory_size

        inner_s_q = zeros(o.current_memory_size)

        for i in o.current_memory_size : -1 : 1
                inner_s_q[i] = inner(p.M, o.x, o.steps[i], q) / inner(p.M, o.x, o.steps[i], o.gradient_diffrences[i])
                q =  q - inner_s_q[i]*o.gradient_diffrences[i]
        end

        if o.current_memory_size <= 1
                r = q
        else
                r = (inner(p.M, o.x, o.steps[o.current_memory_size - 1], o.gradient_diffrences[o.current_memory_size - 1]) / norm(p.M, o.x, o.gradient_diffrences[o.current_memory_size - 1])^2) * q
        end

        for i in 1 : o.current_memory_size
                omega = inner(p.M, o.x, o.gradient_diffrences[i], r) / inner(p.M, o.x, o.steps[i], o.gradient_diffrences[i])
                r = r + (inner_s_q[i] + omega) * o.steps[i]
        end

        return -r
end


# Updating the parameters

function update_parameters(p::GradientProblem, o::RBFGSQuasiNewton{P,T}, α::Float64, η::T, x::P) where {P,T}
        gradf_xold = o.∇
        β = norm(p.M, x, α*η) / norm(p.M, o.x, vector_transport_to(p.M, x, α*η, o.x, o.vector_transport_method))
        yk = β*get_gradient(p,o.x) - vector_transport_to(p.M, x, gradf_xold, o.x, o.vector_transport_method)
        sk = vector_transport_to(p.M, x, α*η, o.x, o.vector_transport_method)

        b = vector_transport_to.(p.M, x, o.inverse_hessian_approximation, o.x, o.vector_transport_method)
        basis = get_vectors(p.M, o.x, get_basis(p.M, o.x, DefaultOrthonormalBasis())) # Here we need to compute an array of orthogonal basis vectors

        n = manifold_dimension(p.M)
        Bkyk = square_matrix_vector_product(p.M, o.x, b, yk; orthonormal_basis = basis)
        skyk = inner(p.M, o.x, yk, sk)

        for i in 1:n
                o.inverse_hessian_approximation[i] = b[i] - (inner(p.M, o.x, basis[i], sk) / skyk) * Bkyk - (inner(p.M, o.x, Bkyk, basis[i]) / skyk) * sk + ((inner(p.M, o.x, yk, Bkyk)*inner(p.M, o.x, sk, basis[i])) / skyk^2) * sk + (inner(p.M, o.x, sk, basis[i]) / skyk) * sk
        end
end

function update_parameters(p::GradientProblem, o::CautiuosRBFGSQuasiNewton{P,T}, α::Float64, η::T, x::P) where {P,T}
        gradf_xold = o.∇
        β = norm(p.M, x, α*η) / norm(p.M, x, vector_transport_to(p.M, x, α*η, o.x, o.vector_transport_method))
        yk = β*get_gradient(p,o.x) - vector_transport_to(p.M, x, gradf_xold, o.x, o.vector_transport_method)
        sk = vector_transport_to(p.M, x, α*η, o.x, o.vector_transport_method)

        sk_yk = inner(p.M, o.x, sk, yk)
        norm_sk = norm(p.M, o.x, sk)

        bound = o.cautious_fct(norm(p.M, x, gradf_xold))

        if norm_sk != 0 && (sk_yk / norm_sk) >= bound
                b = vector_transport_to.(p.M, x, o.inverse_hessian_approximation, o.x, o.vector_transport_method)
                basis = get_vectors(p.M, o.x, get_basis(p.M, o.x, DefaultOrthonormalBasis())) # Here we need to compute an array of orthogonal basis vectors

                n = manifold_dimension(p.M)
                Bkyk = square_matrix_vector_product(p.M, o.x, b, yk; orthonormal_basis = basis)
                skyk = inner(p.M, o.x, yk, sk)

                for i in 1:n
                        o.inverse_hessian_approximation[i] = b[i] - (inner(p.M, o.x, basis[i], sk) / skyk) * Bkyk - (inner(p.M, o.x, Bkyk, basis[i]) / skyk) * sk + ((inner(p.M, o.x, yk, Bkyk)*inner(p.M, o.x, sk, basis[i])) / skyk^2) * sk + (inner(p.M, o.x, sk, basis[i]) / skyk) * sk
                end
        else
                o.inverse_hessian_approximation = vector_transport_to.(p.M, x, o.inverse_hessian_approximation, o.x, o.vector_transport_method)
        end

end

# Limited memory variants

function update_parameters(p::GradientProblem, o::LimitedMemoryQuasiNewtonOptions, α::Float64, η::T, xk::P) where {P,T}
        gradf_xold = get_gradient(p,xk)
        β = norm(p.M, xk, α*η) / norm(p.M, o.x, vector_transport_to(p.M, xk, α*η, o.x, o.vector_transport_method))
        yk = β*get_gradient(p,o.x) - vector_transport_to(p.M, xk, gradf_xold, o.x, o.vector_transport_method)
        sk = vector_transport_to(p.M, xk, α*η, o.x, o.vector_transport_method)

        if o.current_memory_size >= o.memory_size
                for  i in 2 : o.memory_size
                        o.steps[i] = vector_transport_to(p.M, xk, o.steps[i], o.x, o.vector_transport_method)
                        o.gradient_diffrences[i] = vector_transport_to(p.M, xk, o.gradient_diffrences[i], o.x, o.vector_transport_method)
                end

                if o.memory_size > 1
                        o.steps = o.steps[2:end]
                        o.gradient_diffrences = o.gradient_diffrences[2:end]
                end

                if o.memory_size > 0
                        o.steps[o.memory_size] = sk
                        o.gradient_diffrences[o.memory_size] = yk
                end
        else

                for i in 1:o.current_memory_size
                        o.steps[i] = vector_transport_to(p.M, xk, o.steps[i], o.x, o.vector_transport_method)
                        o.gradient_diffrences[i] = vector_transport_to(p.M, xk, o.gradient_diffrences[i], o.x, o.vector_transport_method)
                end

                o.steps[o.current_memory_size + 1] = sk
                o.gradient_diffrences[o.current_memory_size + 1] = yk

                o.current_memory_size = o.current_memory_size + 1
        end
end


function update_parameters(p::GradientProblem, o::CautiuosLimitedMemoryQuasiNewtonOptions, α::Float64, η::T, x::P) where {P,T}
        gradf_xold = get_gradient(p,x)
        β = norm(p.M, x, α*η) / norm(p.M, x, vector_transport_to(p.M, x, α*η, o.x, o.vector_transport_method))
        yk = β*get_gradient(p,o.x) - vector_transport_to(p.M, x, gradf_xold, o.x, o.vector_transport_method)
        sk = vector_transport_to(p.M, x, α*η, o.x, o.vector_transport_method)

        sk_yk = inner(p.M, o.x, sk, yk)
        norm_sk = norm(p.M, o.x, sk)
        bound = o.cautious_fct(norm(p.M, x, get_gradient(p,x)))

        if norm_sk != 0 && (sk_yk / norm_sk) >= bound
                if o.current_memory_size >= o.memory_size
                        for  i in 2 : o.memory_size
                                vector_transport_to(p.M, x, o.steps[i], o.x, o.vector_transport_method)
                                o.steps[i] = vector_transport_to(p.M, x, o.steps[i], o.x, o.vector_transport_method)
                                o.gradient_diffrences[i] = vector_transport_to(p.M, x, o.gradient_diffrences[i], o.x, o.vector_transport_method)
                        end

                        if o.memory_size > 1
                                o.steps = o.steps[2:end]
                                o.gradient_diffrences = o.gradient_diffrences[2:end]
                        end

                        if o.memory_size > 0
                                o.steps[o.memory_size] = sk
                                o.gradient_diffrences[o.memory_size] = yk
                        end
                else

                        for i in 1:o.current_memory_size
                                o.steps[i] = vector_transport_to(p.M, x, o.steps[i], o.x, o.vector_transport_method)
                                o.gradient_diffrences[i] = vector_transport_to(p.M, x, o.gradient_diffrences[i], o.x, o.vector_transport_method)
                        end

                        o.steps[o.current_memory_size+1] = sk
                        o.gradient_diffrences[o.current_memory_size+1] = yk

                        o.current_memory_size = o.current_memory_size + 1
                end
        else
                for  i = 1 : min(o.current_memory_size, o.memory_size)
                        o.steps[i] = o.Vector_Transport(p.M, x, o.x, o.steps[i])
                        o.gradient_diffrences[i] = o.Vector_Transport(p.M, x, o.x, o.gradient_diffrences[i])
                end
        end

end

function square_matrix_vector_product(M::Manifold, p::P, A::AbstractVector{T}, X::T; orthonormal_basis::AbstractVector{T} = get_vectors(M, p, get_basis(M, p, DefaultOrthonormalBasis()))) where {P,T}
        Y = zero_tangent_vector(M,p)
        n = manifold_dimension(M)
        for i in 1 : n
                Y = Y + inner(M, p, A[i], X) * orthonormal_basis[i]
        end
        return Y
end


get_solver_result(o::O) where {O <: QuasiNewtonOptions} = o.x
