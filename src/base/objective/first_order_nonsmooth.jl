# TODO: subgrad, prox

function _ensure_mutating_prox(prox_f, p, evaluation::AbstractEvaluationType)
    return prox_f
end
function _ensure_mutating_prox(prox_f, q::Number, evaluation::AllocatingEvaluation)
    return isnothing(prox_f) ? prox_f : (M, λ, p) -> [prox_f(M, λ, p[])]
end
function _ensure_mutating_prox(prox_f, q::Number, evaluation::InplaceEvaluation)
    return isnothing(prox_f) ? prox_f : (M, q, λ, p) -> (q .= [prox_f(M, λ, p[])])
end

function get_proximal_map end
#TODO: Add a docstring also here?
function get_proximal_map(amp::AbstractManoptProblem, λ, p, i)
    return get_proximal_map(get_manifold(amp), get_objective(amp), λ, p, i)
end
function get_proximal_map!(amp::AbstractManoptProblem, q, λ, p, i)
    return get_proximal_map!(get_manifold(amp), q, get_objective(amp), λ, p, i)
end
function get_proximal_map(amp::AbstractManoptProblem, λ, p)
    return get_proximal_map(get_manifold(amp), get_objective(amp), λ, p)
end
function get_proximal_map!(amp::AbstractManoptProblem, q, λ, p)
    return get_proximal_map!(get_manifold(amp), q, get_objective(amp), λ, p)
end
function get_proximal_map(M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, λ, p)
    return get_proximal_map(M, get_objective(admo, false), λ, p)
end
function get_proximal_map!(
        M::AbstractManifold, q, admo::AbstractDecoratedManifoldObjective, λ, p
    )
    return get_proximal_map!(M, q, get_objective(admo, false), λ, p)
end
