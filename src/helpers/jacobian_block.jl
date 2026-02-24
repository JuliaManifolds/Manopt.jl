@doc raw"""
	JacobianBlock(n_rows, n_cols, row_starts, col_starts, blocks)

Represent a mostly-zero matrix of size `(n_rows, n_cols)` by storing only a tuple of
dense sub-blocks `blocks`, whose top-left entries are located at
`(row_starts[k], col_starts[k])`.

All entries outside these sub-blocks are interpreted as zero.
"""
struct JacobianBlock{T, N, MT <: NTuple{N, AbstractMatrix{T}}} <: AbstractMatrix{T}
    n_rows::Int
    n_cols::Int
    row_starts::NTuple{N, Int}
    col_starts::NTuple{N, Int}
    blocks::MT

    function JacobianBlock{T, N, MT}(
            n_rows::Int,
            n_cols::Int,
            row_starts::NTuple{N, Int},
            col_starts::NTuple{N, Int},
            blocks::MT,
        ) where {T, N, MT <: NTuple{N, AbstractMatrix{T}}}
        n_rows < 1 && throw(ArgumentError("The number of rows has to be positive."))
        n_cols < 1 && throw(ArgumentError("The number of columns has to be positive."))
        N < 1 && throw(ArgumentError("At least one block has to be provided."))

        for i in 1:N
            row_starts[i] < 1 && throw(ArgumentError("The first block row has to be positive."))
            col_starts[i] < 1 &&
                throw(ArgumentError("The first block column has to be positive."))
            size(blocks[i], 1) < 1 &&
                throw(ArgumentError("The block has to have at least one row."))
            size(blocks[i], 2) < 1 &&
                throw(ArgumentError("The block has to have at least one column."))

            row_end = row_starts[i] + size(blocks[i], 1) - 1
            col_end = col_starts[i] + size(blocks[i], 2) - 1

            row_end > n_rows && throw(
                ArgumentError("The stored block exceeds the matrix dimensions in rows."),
            )
            col_end > n_cols && throw(
                ArgumentError("The stored block exceeds the matrix dimensions in columns."),
            )
        end

        return new{T, N, MT}(n_rows, n_cols, row_starts, col_starts, blocks)
    end
end

function JacobianBlock(
        n_rows::Integer,
        n_cols::Integer,
        row_starts::NTuple{N, <:Integer},
        col_starts::NTuple{N, <:Integer},
        blocks::NTuple{N, AbstractMatrix{T}},
    ) where {N, T}
    row_starts_int = ntuple(i -> Int(row_starts[i]), Val(N))
    col_starts_int = ntuple(i -> Int(col_starts[i]), Val(N))
    return JacobianBlock{T, N, typeof(blocks)}(
        Int(n_rows),
        Int(n_cols),
        row_starts_int,
        col_starts_int,
        blocks,
    )
end

Base.size(A::JacobianBlock) = (A.n_rows, A.n_cols)

_row_range(A::JacobianBlock, k::Integer) =
    A.row_starts[k]:(A.row_starts[k] + size(A.blocks[k], 1) - 1)
_col_range(A::JacobianBlock, k::Integer) =
    A.col_starts[k]:(A.col_starts[k] + size(A.blocks[k], 2) - 1)

function LinearAlgebra.mul!(
        C::AbstractMatrix,
        At::Adjoint{<:Any, <:JacobianBlock},
        B::JacobianBlock,
        α::Number,
        β::Number,
    )
    A = parent(At)
    size(A, 1) == size(B, 1) || throw(DimensionMismatch("Matrix dimensions mismatch."))
    size(C, 1) == size(A, 2) || throw(DimensionMismatch("Output matrix has wrong row size."))
    size(C, 2) == size(B, 2) || throw(DimensionMismatch("Output matrix has wrong column size."))

    β == zero(β) ? fill!(C, zero(eltype(C))) : (C .*= β)

    for i in eachindex(A.blocks)
        rows_A = _row_range(A, i)
        cols_A = _col_range(A, i)
        for j in eachindex(B.blocks)
            rows_B = _row_range(B, j)
            cols_B = _col_range(B, j)

            overlap_start = max(first(rows_A), first(rows_B))
            overlap_end = min(last(rows_A), last(rows_B))
            if overlap_start > overlap_end
                continue
            end

            local_rows_A =
                (overlap_start - first(rows_A) + 1):(overlap_end - first(rows_A) + 1)
            local_rows_B =
                (overlap_start - first(rows_B) + 1):(overlap_end - first(rows_B) + 1)

            C[cols_A, cols_B] .+= α .* (
                A.blocks[i][local_rows_A, :]' * B.blocks[j][local_rows_B, :]
            )
        end
    end
    return C
end

function LinearAlgebra.mul!(
        C::AbstractMatrix,
        At::Adjoint{<:Any, <:JacobianBlock},
        B::JacobianBlock,
    )
    T = promote_type(eltype(C), eltype(parent(At)), eltype(B))
    return mul!(C, At, B, one(T), zero(T))
end

function Base.getindex(A::JacobianBlock{T}, i::Integer, j::Integer) where {T}
    @boundscheck checkbounds(A, i, j)
    v = zero(T)
    for k in eachindex(A.blocks)
        row_end = A.row_starts[k] + size(A.blocks[k], 1) - 1
        col_end = A.col_starts[k] + size(A.blocks[k], 2) - 1
        if A.row_starts[k] <= i <= row_end && A.col_starts[k] <= j <= col_end
            v += A.blocks[k][i - A.row_starts[k] + 1, j - A.col_starts[k] + 1]
        end
    end
    return v
end

function Base.Matrix(A::JacobianBlock{T}) where {T}
    result = zeros(T, size(A)...)
    for k in eachindex(A.blocks)
        result[_row_range(A, k), _col_range(A, k)] .+= A.blocks[k]
    end
    return result
end

function Base.:*(A::JacobianBlock{T1}, x::AbstractVector{T2}) where {T1, T2}
    size(A, 2) == length(x) || throw(DimensionMismatch("Matrix and vector dimensions mismatch."))
    T = promote_type(T1, T2)
    y = zeros(T, size(A, 1))
    for k in eachindex(A.blocks)
        y[_row_range(A, k)] .+= A.blocks[k] * view(x, _col_range(A, k))
    end
    return y
end

function Base.:*(A::JacobianBlock{T1}, B::AbstractMatrix{T2}) where {T1, T2}
    size(A, 2) == size(B, 1) || throw(DimensionMismatch("Matrix dimensions mismatch."))
    T = promote_type(T1, T2)
    C = zeros(T, size(A, 1), size(B, 2))
    for k in eachindex(A.blocks)
        C[_row_range(A, k), :] .+= A.blocks[k] * view(B, _col_range(A, k), :)
    end
    return C
end

function Base.:*(A::AbstractMatrix{T1}, B::JacobianBlock{T2}) where {T1, T2}
    size(A, 2) == size(B, 1) || throw(DimensionMismatch("Matrix dimensions mismatch."))
    T = promote_type(T1, T2)
    C = zeros(T, size(A, 1), size(B, 2))
    for k in eachindex(B.blocks)
        C[:, _col_range(B, k)] .+= view(A, :, _row_range(B, k)) * B.blocks[k]
    end
    return C
end

function Base.:*(A::JacobianBlock{T1}, B::JacobianBlock{T2}) where {T1, T2}
    size(A, 2) == size(B, 1) || throw(DimensionMismatch("Matrix dimensions mismatch."))
    return Matrix(A) * Matrix(B)
end

function Base.:*(a::Number, A::JacobianBlock)
    N = length(A.blocks)
    blocks = ntuple(i -> a .* A.blocks[i], Val(N))
    return JacobianBlock(size(A, 1), size(A, 2), A.row_starts, A.col_starts, blocks)
end

function Base.:*(A::JacobianBlock, a::Number)
    N = length(A.blocks)
    blocks = ntuple(i -> A.blocks[i] .* a, Val(N))
    return JacobianBlock(size(A, 1), size(A, 2), A.row_starts, A.col_starts, blocks)
end

function Base.:+(A::JacobianBlock{T1}, B::AbstractMatrix{T2}) where {T1, T2}
    size(A) == size(B) || throw(DimensionMismatch("Matrix dimensions mismatch."))
    C = Matrix{promote_type(T1, T2)}(B)
    for k in eachindex(A.blocks)
        C[_row_range(A, k), _col_range(A, k)] .+= A.blocks[k]
    end
    return C
end

function Base.:+(A::JacobianBlock{T1, N1}, B::JacobianBlock{T2, N2}) where {T1, T2, N1, N2}
    size(A) == size(B) || throw(DimensionMismatch("Matrix dimensions mismatch."))

    if N1 == N2 &&
            A.row_starts == B.row_starts &&
            A.col_starts == B.col_starts &&
            all(size(A.blocks[k]) == size(B.blocks[k]) for k in eachindex(A.blocks))
        blocks = ntuple(k -> A.blocks[k] .+ B.blocks[k], Val(N1))
        return JacobianBlock(size(A, 1), size(A, 2), A.row_starts, A.col_starts, blocks)
    end

    return Matrix(A) + Matrix(B)
end
