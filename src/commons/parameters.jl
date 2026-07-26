function set_parameter!(TpM::TangentSpace, ::Union{Val{:Basepoint}, Val{:p}}, p)
    copyto!(TpM.manifold, TpM.point, p)
    return TpM
end

"""
    is_tutorial_mode()

A small internal helper to indicate whether tutorial mode is active.

You can set the mode by calling `set_parameter!(:Mode, "Tutorial")` or deactivate it
by `set_parameter!(:Mode, "")`.
"""
is_tutorial_mode() = (get_parameter(:Mode) == "Tutorial")
