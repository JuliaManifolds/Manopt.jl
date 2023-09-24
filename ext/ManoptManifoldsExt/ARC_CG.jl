function set_manopt_parameter!(M::TangentSpaceAtPoint, ::Val{:p}, v)
    copyto!(M.fiber.manifold, M.point, v)
    return M
end
function (f::Manopt.AdaptiveRegularizationCubicCost)(M::TangentSpaceAtPoint, X)
    ## (33) in Agarwal et al.
    return get_cost(base_manifold(M), f.mho, M.point) +
           inner(base_manifold(M), M.point, X, f.X) +
           1 / 2 * inner(
               base_manifold(M),
               M.point,
               X,
               get_hessian(base_manifold(M), f.mho, M.point, X),
           ) +
           f.σ / 3 * norm(base_manifold(M), M.point, X)^3
end
function (grad_f::Manopt.AdaptiveRegularizationCubicGrad)(M::TangentSpaceAtPoint, X)
    # (37) in Agarwal et
    return grad_f.X +
           get_hessian(base_manifold(M), grad_f.mho, M.point, X) +
           grad_f.σ * norm(base_manifold(M), M.point, X) * X
end
function (grad_f::Manopt.AdaptiveRegularizationCubicGrad)(M::TangentSpaceAtPoint, Y, X)
    get_hessian!(base_manifold(M), Y, grad_f.mho, M.point, X)
    Y .= Y + grad_f.X + grad_f.σ * norm(base_manifold(M), M.point, X) * X
    return Y
end
function (c::StopWhenFirstOrderProgress)(
    dmp::AbstractManoptProblem{<:TangentSpaceAtPoint},
    ams::AbstractManoptSolverState,
    i::Int,
)
    if (i == 0)
        c.reason = ""
        return false
    end
    #Update Gradient
    TpM = get_manifold(dmp)
    nG = norm(base_manifold(TpM), TpM.point, get_gradient(dmp, ams.p))
    nX = norm(base_manifold(TpM), TpM.point, ams.p)
    if (i > 0) && (nG <= c.θ * nX^2)
        c.reason = "The algorithm has reduced the model grad norm by $(c.θ).\n"
        return true
    end
    return false
end
