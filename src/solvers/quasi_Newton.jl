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


function step_solver!(p::P,o::O,iter) where {P <: GradientProblem, O <: quasi_Newton_Options}
        # Compute BFGS direction
        η = get_quasi_Newton_Direction(p, o)

        # Execute line-search
        α = line_search(p, o, η)

        # Compute Step
        x_old = o.x
        retract!(p.M, o.x, o.x, α*η, o.retraction_method)

        # Update the Parameters
        update_Parameters(p, o, η, α, x_old)
end


function get_quasi_Newton_Direction(p::GradientProblem, o::Standard_quasi_Newton_Options)
        return -o.Hessian_Inverse_Aproximation(p.M, o.x, get_gradient(p,o.x))
end


function get_quasi_Newton_Direction(p::GradientProblem, o::Limited_Memory_quasi_Newton_Options)

        q = get_gradient(p, o.x)

        inner_s_q = zeros(1, k);

        for i = k : -1 : 1
                inner_s_q(1, i) = rhoHistory[i] * M.inner(xCur, Step_Memory[i], q);
                q = M.lincomb(xCur, 1, q, -inner_s_q(1, i), Gradient_Memory[i]);
        end

        r = M.lincomb(xCur, scaleFactor, q);

        for i = 1 : k
                omega = rhoHistory[i] * M.inner(xCur, Gradient_Memory[i], r);
                r = M.lincomb(xCur, 1, r, inner_s_q(1, i)-omega, Step_Memory[i]);
        end

        dir = M.lincomb(xCur, -1, r);
end


function update_Parameters(p::GradientProblem, o::Standard_quasi_Newton_Options, α::Float64, η::T, x::p)
        gradf_xold = get_gradient(p,x)

        yk = get_gradient(p,o.x) - vector_transport_to(p.M, x, gradf_xold, o.x, o.vector_transport_method)
        sk = o.Vector_Transport(x, o.x, α*η)

        if o.cautious == true
                sk_yk = dot(p.M, o.x, sk, yk)
                norm_sk = norm(p.M, o.x, sk)
                bound = o.cautious_Function(norm(p.M, x, gradf_xold))

                if norm_sk != 0 && (sk_yk / norm_sk) >= bound
                        # Update of the Function / Matrix
                else
                        # Transport of the old Function / Matrix to the new tagent space
                end
        else
                # Update of the Function / Matrix
        end
end


function update_Parameters(p::GradientProblem, o::Limited_Memory_quasi_Newton_Options, α::Float64, η::T, x::p)
        yk = get_gradient(p,o.x) - o.Vector_Transport(p.M, x, o.x, get_gradient(p,x))
        sk = o.Vector_Transport(x, o.x, α*η)

        if o.cautious == true
                sk_yk = dot(p.M, o.x, sk, yk)
                norm_sk = norm(p.M, o.x, sk)
                bound = o.cautious_Function(norm(p.M, x, get_gradient(p,x)))

                if norm_sk != 0 && (sk_yk / norm_sk) >= bound
                        memory = size(o.Step_Memory)

                        if iter >= memory

                                for  i = 2 : memory
                                        Step_Memory[i] = o.Vector_Transport(p.M, x, o.x, Step_Memory[i])
                                        Gradient_Memory[i] = o.Vector_Transport(p.M, x, o.x, Gradient_Memory[i])
                                end

                                if memory > 1
                                        Step_Memory = Step_Memory[2:end]
                                        Gradient_Memory = Gradient_Memory[2:end]
                                end

                                if memory > 0
                                        Step_Memory[memory] = sk
                                        Gradient_Memory[memory] = yk
                                end
                        else

                                for i = 1:iter
                                        Step_Memory[i] = o.Vector_Transport(p.M, x, o.x, Step_Memory[i])
                                        Gradient_Memory[i] = o.Vector_Transport(p.M, x, o.x, Gradient_Memory[i])
                                end

                                Step_Memory[iter+1] = sk
                                Gradient_Memory[iter+1] = yk
                        end

                else
                        for  i = 1 : min(iter, memory)
                                Step_Memory[i] = o.Vector_Transport(p.M, x, o.x, Step_Memory[i])
                                Gradient_Memory[i] = o.Vector_Transport(p.M, x, o.x, Gradient_Memory[i])
                        end
                end
        else
                memory = size(o.Step_Memory)

                if iter >= memory

                        for  i = 2 : memory
                                Step_Memory[i] = o.Vector_Transport(p.M, x, o.x, Step_Memory[i])
                                Gradient_Memory[i] = o.Vector_Transport(p.M, x, o.x, Gradient_Memory[i])
                        end

                        if memory > 1
                                Step_Memory = Step_Memory[2:end]
                                Gradient_Memory = Gradient_Memory[2:end]
                        end

                        if memory > 0
                                Step_Memory[memory] = sk
                                Gradient_Memory[memory] = yk
                        end
                else

                        for i = 1:iter
                                Step_Memory[i] = o.Vector_Transport(p.M, x, o.x, Step_Memory[i])
                                Gradient_Memory[i] = o.Vector_Transport(p.M, x, o.x, Gradient_Memory[i])
                        end

                        Step_Memory[iter+1] = sk
                        Gradient_Memory[iter+1] = yk
                end
        end
end


get_solver_result(o::O) where {O <: quasi_Newton_Options} = o.x
