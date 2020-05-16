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

initialize_solver!(p::P,o::O) where

function step_solver!(p::P,o::O,iter) where
        
end
get_solver_result(o::O) where {O <: TrustRegionsOptions} = o.x
