#
# This small part introduces both debug and record for C(k)
# including this file, you can use
# DebugCk() within debug = and RecordCk() within record =
#
# you have to have x_hat defined.
function tilde_x_old(p, o, x_old, ξbar_old)
    return exp(
        p.M,
        x_old,
        vector_transport_to(
            p.M,
            o.m,
            -o.primal_stepsize * adjoint_linearized_operator(p, o.m, o.n, ξbar_old),
            x_old,
        ),
    )
end
function ζk(p, o, x_old, ξbar_old)
    return vector_transport_to(
        p.M,
        x_old,
        log(p.M, x_old, o.x) - vector_transport_to(
            p.M,
            tilde_x_old(p, o, x_old, ξbar_old),
            log(p.M, tilde_x_old(p, o, x_old, ξbar_old), x_hat),
            x_old,
        ),
        o.m,
    ) - log(p.M, o.m, o.x) + log(p.M, o.m, x_hat)
end
function Ck(p, o, x_old, ξ_bar_old)
    return 1 / o.primal_stepsize *
           distance(p.M, x_old, tilde_x_old(p, o, x_old, ξ_bar_old))^2 + inner(
        p.N,
        o.n,
        ξ_bar_old,
        linearized_forward_operator(p, o.m, ζk(p, o, x_old, ξ_bar_old), o.n),
    )
end

struct DebugCk <: DebugAction
    io::IO
    prefix::String
    storage::StoreStateAction
    function DebugCk(a::StoreStateAction=StoreStateAction((:Iterate, :ξbar)), io::IO=stdout)
        return new(io, "C(k): ", a)
    end
end
function (d::DebugCk)(p::P, o::ChambollePockState, i::Int) where {P<:PrimalDualProblem}
    if all(has_storage.(Ref(d.storage), [:Iterate, :ξbar])) && i > 0 # all values stored
        x_old, ξ_bar_old = get_storage.(Ref(d.storage), [:Iterate, :ξbar]) #fetch
        print(d.io, d.prefix * "$(Ck(p, o, x_old,ξ_bar_old))")
    end
    return d.storage(p, o, i)
end

struct RecordCk <: RecordAction
    recorded_values::Array{Float64,1}
    storage::StoreStateAction
    function RecordCk(a::StoreStateAction=StoreStateAction((:Iterate, :ξbar)))
        return new(Array{Float64,1}(), a)
    end
end
function (r::RecordCk)(p::P, o::ChambollePockState, i::Int) where {P<:PrimalDualProblem}
    if all(has_storage.(Ref(r.storage), [:Iterate, :ξbar])) && i > 0 # all values stored
        x_old = get_storage(r.storage, :Iterate)
        ξ_bar_old = get_storage(r.storage, :ξbar)
        Manopt.record_or_reset!(r, Ck(p, o, x_old, ξ_bar_old), i)
    end
    return r.storage(p, o, i)
end
