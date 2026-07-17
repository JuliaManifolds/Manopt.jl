"""
    dispatch_state_decorator(s::AbstractManoptSolverState)

Indicate internally, whether an [`AbstractManoptSolverState`](@ref) `s` is of decorating type,
and stores (encapsulates) a state in itself, by default in the field `s.state`.

Decorators indicate this by returning `Val{true}` for further dispatch.

The default is `Val{false}`, so by default a state is not decorated.
"""
dispatch_state_decorator(::AbstractManoptSolverState) = Val(false)

"""
    is_state_decorator(s::AbstractManoptSolverState)

Indicate, whether [`AbstractManoptSolverState`](@ref) `s` are of decorator type.
"""
function is_state_decorator(s::AbstractManoptSolverState)
    return _extract_val(dispatch_state_decorator(s))
end
