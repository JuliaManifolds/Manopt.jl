function set_parameter!(TpM::TangentSpace, ::Union{Val{:Basepoint}, Val{:p}}, p)
    copyto!(TpM.manifold, TpM.point, p)
    return TpM
end
