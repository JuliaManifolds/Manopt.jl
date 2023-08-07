function set_manopt_parameter!(M::TangentSpaceAtPoint, f::Val{:p}, v)
    return M.point = v
end
function (f::Manopt.AdaptiveRegularizationCubicCost)(M, X)
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
function (grad_f::Manopt.AdaptiveRegularizationCubicGrad)(M, X)
    return grad_f.X +
           get_hessian(base_manifold(M), grad_f.mho, M.point, X) +
           grad_f.σ * norm(base_manifold(M), M.point, X) * X
end
function (grad_f::Manopt.AdaptiveRegularizationCubicGrad)(M, Y, X)
    get_hessian!(base_manifold(M), Y, f.mho, M.point, X)
    Y .= Y + f.X + +f.σ * norm(base_manifold(M), M.point, X)^2 * X
    return Y
end
