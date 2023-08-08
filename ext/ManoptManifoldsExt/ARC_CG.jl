function set_manopt_parameter!(M::TangentSpaceAtPoint, ::Val{:p}, v)
    M.point .= v
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
