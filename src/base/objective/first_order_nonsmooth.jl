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
