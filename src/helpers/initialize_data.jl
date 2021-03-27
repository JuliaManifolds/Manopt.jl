"""
    recursive_copyto!(a,b)

copy the values from `b` to `a` by recursively copying the contents.
On the lowest level of `AbstractArray`s this falls back to `copyto!`.
"""
function recursive_copyto!(a::AbstractArray{T}, b::AbstractArray{T}) where {T<:AbstractArray}
    foreach(recursive_copyto!, a, b)
    return a
end
function recursive_copyto!(a::AbstractArray{T}, b::AbstractArray{T}) where {T}
    return copyto!(a, b)
end
