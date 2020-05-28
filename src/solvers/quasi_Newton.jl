@doc raw"""
    quasi_Newton(M, F, ∇F, x, H, )
"""
function quasi_Newton(
    M::MT,
    F::Function,
    ∇F::Function,
    x,
    H::Union{Function,Missing};
    )


    if return_options
        return resultO
    else
        return get_solver_result(resultO)
    end
end


function initialize_solver!(p::P,o::O) where {P <: GradientProblem, O <: quasi_Newton_Options}
end


function step_solver!(p::P,o::O,o.current_memory_size) where {P <: GradientProblem, O <: quasi_Newton_Options}
        # Compute BFGS direction
        η = get_quasi_Newton_Direction(p, o)

        # Execute line-search
        α = line_search(p, o, η) # Not sure which Parameters are necessary

        # Compute Step
        x_old = o.x
        retract!(p.M, o.x, o.x, α*η, o.retraction_method)

        # Update the Parameters
        update_Parameters(p, o, η, α, x_old)
end

# Computing the direction

function get_quasi_Newton_Direction(p::GradientProblem, o::QuasiNewtonOptions)
        o.∇ = get_gradient(p,o.x)
        return -o.Hessian_Inverse_Aproximation(p.M, o.x, o.∇)
end

# Limited memory variants

function get_quasi_Newton_Direction(p::GradientProblem, o::RLBFGSOptions{P,T})
        o.∇ = get_gradient(p,o.x)

        q = o.∇
        k = o.current_memory_size

        inner_s_q = zeros(k)

        for i in k : -1 : 1
                inner_s_q[i] = dot(p.M, o.x, o.steps[i], q) / dot(p.M, o.x, o.steps[i], o.stepsgradient_diffrences[i])
                q =  q - inner_s_q[i]*o.stepsgradient_diffrences[i]
        end

        if k == 1
                r = q
        else
                r = (dot(p.M, o.x, o.steps[k-1], o.stepsgradient_diffrences[k-1]) / norm(p.M, o.x, o.stepsgradient_diffrences[k-1])^2) * q
        end

        for i in 1 : k
                omega = dot(p.M, o.x, o.stepsgradient_diffrences[i], r) / dot(p.M, o.x, o.steps[i], o.stepsgradient_diffrences[i])
                r = r  (inner_s_q[i] + omega) * o.steps[i])
        end

        return r
end


# Updating the parameters

function update_Parameters(p::GradientProblem, o::BFGSQuasiNewton{P,T}, α::Float64, η::T, x::p)
        gradf_xold = o.∇
        β = norm(p.M, x, α*η) / norm(p.M, x, vector_transport_to(p.M, x, α*η, o.x, o.vector_transport_method))
        yk = β*get_gradient(p,o.x) - vector_transport_to(p.M, x, gradf_xold, o.x, o.vector_transport_method)
        sk = vector_transport_to(p.M, x, α*η, o.x, o.vector_transport_method)

        o.inverse_hessian_approximation = # Update of the Function / Matrix
end

function update_Parameters(p::GradientProblem, o::CautiuosBFGSQuasiNewton{P,T}, α::Float64, η::T, x::p)
        gradf_xold = o.∇
        β = norm(p.M, x, α*η) / norm(p.M, x, vector_transport_to(p.M, x, α*η, o.x, o.vector_transport_method))
        yk = β*get_gradient(p,o.x) - vector_transport_to(p.M, x, gradf_xold, o.x, o.vector_transport_method)
        sk = vector_transport_to(p.M, x, α*η, o.x, o.vector_transport_method)

        sk_yk = dot(p.M, o.x, sk, yk)
        norm_sk = norm(p.M, o.x, sk)

        bound = o.cautious_Function(norm(p.M, x, gradf_xold))

        if norm_sk != 0 && (sk_yk / norm_sk) >= bound
                o.inverse_hessian_approximation = # Update of the Function / Matrix
        else
                o.inverse_hessian_approximation = # Transport of the old Function / Matrix to the new tagent space
        end

end

# Limited memory variants

function update_Parameters(p::GradientProblem, o:: LimitedMemoryQuasiNewtonOptions, α::Float64, η::T, x::p)
        gradf_xold = o.∇
        β = norm(p.M, x, α*η) / norm(p.M, x, vector_transport_to(p.M, x, α*η, o.x, o.vector_transport_method))
        yk = β*get_gradient(p,o.x) - vector_transport_to(p.M, x, gradf_xold, o.x, o.vector_transport_method)
        sk = vector_transport_to(p.M, x, α*η, o.x, o.vector_transport_method)

        if o.current_memory_size >= o.memory_size
                for  i in 2 : o.memory_size
                        vector_transport_to(p.M, x, o.steps[i], o.x, o.vector_transport_method))
                        o.steps[i] = vector_transport_to(p.M, x, o.steps[i], o.x, o.vector_transport_method))
                        o.stepsgradient_diffrences[i] = vector_transport_to(p.M, x, o.stepsgradient_diffrences[i], o.x, o.vector_transport_method))
                end

                if o.memory_size > 1
                        o.steps = o.steps[2:end]
                        o.stepsgradient_diffrences = o.stepsgradient_diffrences[2:end]
                end

                if o.memory_size > 0
                        o.steps[o.memory_size] = sk
                        o.stepsgradient_diffrences[o.memory_size] = yk
                end
        else

                for i in 1:o.current_memory_size
                        o.steps[i] = vector_transport_to(p.M, x, o.steps[i], o.x, o.vector_transport_method))
                        o.stepsgradient_diffrences[i] = vector_transport_to(p.M, x, o.stepsgradient_diffrences[i], o.x, o.vector_transport_method))
                end

                o.steps[o.current_memory_size+1] = sk
                o.stepsgradient_diffrences[o.current_memory_size+1] = yk

                o.current_memory_size = o.current_memory_size + 1
        end
end


function update_Parameters(p::GradientProblem, o::CautiuosLimitedMemoryQuasiNewtonOptions, α::Float64, η::T, x::p)
        gradf_xold = o.∇
        β = norm(p.M, x, α*η) / norm(p.M, x, vector_transport_to(p.M, x, α*η, o.x, o.vector_transport_method))
        yk = β*get_gradient(p,o.x) - vector_transport_to(p.M, x, gradf_xold, o.x, o.vector_transport_method)
        sk = vector_transport_to(p.M, x, α*η, o.x, o.vector_transport_method)

        sk_yk = dot(p.M, o.x, sk, yk)
        norm_sk = norm(p.M, o.x, sk)
        bound = o.cautious_Function(norm(p.M, x, get_gradient(p,x)))

        if norm_sk != 0 && (sk_yk / norm_sk) >= bound
                if o.current_memory_size >= o.memory_size
                        for  i in 2 : o.memory_size
                                vector_transport_to(p.M, x, o.steps[i], o.x, o.vector_transport_method))
                                o.steps[i] = vector_transport_to(p.M, x, o.steps[i], o.x, o.vector_transport_method))
                                o.stepsgradient_diffrences[i] = vector_transport_to(p.M, x, o.stepsgradient_diffrences[i], o.x, o.vector_transport_method))
                        end

                        if o.memory_size > 1
                                o.steps = o.steps[2:end]
                                o.stepsgradient_diffrences = o.stepsgradient_diffrences[2:end]
                        end

                        if o.memory_size > 0
                                o.steps[o.memory_size] = sk
                                o.stepsgradient_diffrences[o.memory_size] = yk
                        end
                else

                        for i in 1:o.current_memory_size
                                o.steps[i] = vector_transport_to(p.M, x, o.steps[i], o.x, o.vector_transport_method))
                                o.stepsgradient_diffrences[i] = vector_transport_to(p.M, x, o.stepsgradient_diffrences[i], o.x, o.vector_transport_method))
                        end

                        o.steps[o.current_memory_size+1] = sk
                        o.stepsgradient_diffrences[o.current_memory_size+1] = yk

                        o.current_memory_size = o.current_memory_size + 1
                end
        else
                for  i = 1 : min(o.current_memory_size, o.memory_size)
                        o.steps[i] = o.Vector_Transport(p.M, x, o.x, o.steps[i])
                        o.stepsgradient_diffrences[i] = o.Vector_Transport(p.M, x, o.x, o.stepsgradient_diffrences[i])
                end
        end

end


get_solver_result(o::O) where {O <: quasi_Newton_Options} = o.x
