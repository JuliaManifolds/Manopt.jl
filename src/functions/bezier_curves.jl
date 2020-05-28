@doc doc"""
    BezierSegment

A type to capture a Bezier segment. With $n$ points, a Beziér segment of degree $n-1$
is stored. On the Euclidean manifold, this yields a polynomial of degree $n-1$.

This type is mainly used to encapsulate the points within a composite Bezier curve, which
consist of an `AbstractVector` of `BezierSegments` where each of the points might
be a nested array on a `PowerManifold` already.

Not that this can also be used to represent tangent vectors on the control points of a segment.

See also: [`de_casteljau`](@ref).

# Constructor
    BezierSegment(pts::AbstractVector)

Given an abstract vector of `pts` generate the corresponding Bézier segment.
"""
struct BezierSegment{T<:AbstractVector{S} where S}
    pts::T
end
#BezierSegment(pts::T) where {T <: AbstractVector{S} where S} = BezierSegment{T}(pts)
Base.show(io::IO, b::BezierSegment) = print(io, "BezierSegment($(b.pts))")

@doc raw"""
    de_casteljau(M::Manifold, b::BezierSegment NTuple{N,P}) -> Function

return the [Bézier curve](https://en.wikipedia.org/wiki/Bézier_curve)
$\beta(\cdot;p_0,\ldots,b_n)\colon [0,1] \to \mathcal M$ defined by the control points
$b_0,\ldots,b_n\in\mathcal M$, $n\in \mathbb N$, as a [`BezierSegment`](@ref).
This function implements de Casteljau's algorithm[^Casteljau1959][^Casteljau1963] gneralized
to manifolds[^PopielNoakes2007]: Let $\gamma_{a,b}(t)$ denote the
shortest geodesic connecting $a,b\in\mathcal M$. Then the curve is defined by the recursion

````math
\begin{aligned}
    \beta(t;b_0,b_1) &= \gamma_{b_0,b_1}(t)\\
    \beta(t;b_0,\ldots,b_n) &= \gamma_{\beta(t;b_0,\ldots,b_{n-1}), \beta(t;b_1,\ldots,b_n)}(t),
\end{aligned}
````
and `P` is the type of a point on the `Manifold` `M`.

````julia
    de_casteljau(M::Manifold, B::AbstractVector{<:BezierSegment}) -> Function
````

Given a vector of Bézier segments, i.e. a vector of control points
$B=\bigl( (b_{0,0},\ldots,b_{n_0,0}),\ldots,(b_{0,m},\ldots b_{n_m,m}) \bigr)$,
where the different segments might be of different degree(s) $n_0,\ldots,n_m$. The resulting
composite Bézier curve $c_B\colon[0,m] \to \mathcal M$ consists of $m$ segments which are
Bézier curves.

````math
c_B(t) :=
    \begin{cases}
        \beta(t; b_{0,0},\ldots,b_{n_0,0}) & \text{ if } t \in [0,1]\\
        \beta(t-i; b_{0,i},\ldots,b_{n_i,i}) & \text{ if }
            t\in (i,i+1], \quad i\in\{1,\ldots,m-1\}.
    \end{cases}
````

````julia
    de_casteljau(M::Manifold, b::BezierSegment, t::Real)
    de_casteljau(M::Manifold, B::AbstractVector{<:BezierSegment}, t::Real)
    de_casteljau(M::Manifold, b::BezierSegment, T::AbstractVector) -> AbstractVector
    de_casteljau(
        M::Manifold,
        B::AbstractVector{<:BezierSegment},
        T::AbstractVector
    ) -> AbstractVector
````

Evaluate the Bézier curve at time `t` or at times `t` in `T`.

[^Casteljau1959]:
    > de Casteljau, P.: Outillage methodes calcul, Enveloppe Soleau 40.040 (1959),
    > Institute National de la Propriété Industrielle, Paris.
[^Casteljau1963]:
    > de Casteljau, P.: Courbes et surfaces à pôles, Microfiche P 4147-1,
    > André Citroën Automobile SA, Paris, (1963).
[^PopielNoakes2007]:
    > Popiel, T. and Noakes, L.: Bézier curves and $C^2$ interpolation in Riemannian
    > manifolds. Journal of Approximation Theory (2007), 148(2), pp. 111–127.-
    > doi: [10.1016/j.jat.2007.03.002](https://doi.org/10.1016/j.jat.2007.03.002).
"""
de_casteljau(M::Manifold, ::Any...)
function de_casteljau(M::Manifold, b::BezierSegment)
    if length(b.pts) == 2
        return t -> shortest_geodesic(M, b.pts[1], b.pts[2], t)
    else
        return t-> shortest_geodesic(
            M,
            de_casteljau(M, BezierSegment(b.pts[1:end-1]),t),
            de_casteljau(M, BezierSegment(b.pts[2:end]),t),
            t,
        )
    end
end
function de_casteljau(M::Manifold, B::AbstractVector{<:BezierSegment})
    length(B) == 1 && return de_casteljau(M, B[1])
    return function (t)
        ((0<t) || (t> length(B))) && DomainError(
            "Parameter $(t) outside of domain of the composite Bézier curve [0,$(length(B))]."
        )
        de_casteljau(M,B[max(ceil(Int,t),1)], ceil(Int,t) == 0 ? 0. : t-ceil(Int,t)+1)
    end
end
de_casteljau(M::Manifold, b, t::Real) = de_casteljau(M,b)(t)
de_casteljau(M::Manifold, b, T::AbstractVector) = map(t -> de_casteljau(M,b,t), T)

@doc raw"""
    get_bezier_junction_tangent_vectors(M::Manifold, B::AbstractVector{<:BezierSegment})
    get_bezier_junction_tangent_vectors(M::Manifold, b::BezierSegment)

returns the tangent vectors at start and end points of the composite Bézier curve
pointing from a junction point to the first and last
inner control points for each segment of the composite Bezier curve specified by
the control points `B`, either a vector of segments of controlpoints.
"""
function get_bezier_junction_tangent_vectors(
    M::Manifold,
    B::AbstractVector{<:BezierSegment},
) where {P}
    return cat(
        [ [log(M, b.pts[1], b.pts[2]), log(M, b.pts[end], b.pts[end-1])] for b in B ]...,
        ;
        dims=1,
    )
end
function get_bezier_junction_tangent_vectors(M::Manifold, b::BezierSegment) where {P,N}
    return get_bezier_junction_tangent_vectors(M,[b])
end

@doc raw"""
    get_bezier_junctions(M::Manifold, B::AbstractVector{<:BezierSegment})
    get_bezier_junctions(M::Manifold, b::BezierSegment)

returns the start and end point(s) of the segments of the composite Bézier curve
specified by the control points `B`. For just one segment `b`, its start and end points
are returned.
"""
function get_bezier_junctions(::Manifold, B::AbstractVector{<:BezierSegment}, double_inner::Bool=false)
    return cat(
        [ double_inner ? [b.pts[[1,end]]...] : [b.pts[1]] for b in B ]...,
        double_inner ? [] : [last(last(B).pts)];
        dims=1,
    )
end
function get_bezier_junctions(::Manifold, b::BezierSegment,::Bool=false)
    return b.pts[[1,end]]
end

@doc raw"""
    get_bezier_inner_points(M::Manifold, B::AbstractVector{<:BezierSegment} )
    get_bezier_inner_points(M::Manifold, b::BezierSegment)

returns the inner (i.e. despite start and end) points of the segments of the
composite Bézier curve specified by the control points `B`. For a single segment `b`,
its inner points are returned
"""
function get_bezier_inner_points(M::Manifold, B::AbstractVector{<:BezierSegment})
    return cat(
        [ [ get_bezier_inner_points(M,b)...] for b in B]...;
        dims=1,
    )
end
function get_bezier_inner_points(::Manifold, b::BezierSegment)
    return b.pts[2:end-1]
end

@doc raw"""
    get_bezier_points(
        M::MAnifold,
        B::AbstractVector{<:BezierSegment},
        reduce::Symbol=:default
    )
    get_bezier_points(M::Manifold, b::BezierSegment, reduce::Symbol=:default)

returns the control points of the segments of the composite Bézier curve
specified by the control points `B`, either a vector of segments of
controlpoints or a.

This method reduces the points depending on the optional `reduce` symbol

* `:default` – no reduction is performed
* `:continuous` – for a continuous function, the junction points are doubled at
  $b_{0,i}=b_{n_{i-1},i-1}$, so only $b_{0,i}$ is in the vector.
* `:differentiable` – for a differentiable function additionally
  $\log_{b_{0,i}}b_{1,i} = -\log_{b_{n_{i-1},i-1}}b_{n_{i-1}-1,i-1}$ holds.
  hence $b_{n_{i-1}-1,i-1}$ is ommited.

If only one segment is given, all points of `b` – i.e. `b.pts` is returned.
"""
function get_bezier_points(
    M::Manifold,
    B::AbstractVector{<:BezierSegment},
    reduce::Symbol=:default
)
    return get_bezier_points(M,B,Val(reduce))
end
function get_bezier_points(::Manifold, B::AbstractVector{<:BezierSegment}, ::Val{:default})
    return cat( [ [b.pts...] for b in B]...; dims=1)
end
function get_bezier_points(::Manifold, B::AbstractVector{<:BezierSegment}, ::Val{:continuous})
    return cat(
        [ [b.pts[1:end-1]...] for b in B]...,
        [last(last(B).pts)];
        dims=1,
    )
end
function get_bezier_points(
    ::Manifold,
    B::AbstractVector{<:BezierSegment},
    ::Val{:differentiable}
)
    return cat(
        [first(B).pts[1]],
        [first(B).pts[2]],
        [ [b.pts[3:end]...] for b in B]...,
        ;
        dims=1,
    )
end
get_bezier_points(::Manifold, b::BezierSegment, ::Symbol=:default) = b.pts

@doc raw"""
    get_bezier_degree(M::Manifold, b::BezierSegment)

return the degree of the Bézier curve represented by the tuple `b` of control points on
the manifold `M`, i.e. the number of points minus 1.
"""
get_bezier_degree(::Manifold, b::BezierSegment) = length(b.pts)-1

@doc raw"""
    get_bezier_degrees(M::Manidold, B::AbstractVector{<:BezierSegment})

return the degrees of the components of a composite Bézier curve represented by tuples
in `B` containing points on the manifold `M`.
"""
function get_bezier_degrees(M::Manifold, B::AbstractVector{<:BezierSegment})
    return get_bezier_degree.(Ref(M),B)
end

@doc raw"""
    get_bezier_segments(M::Manifold, c::AbstractArray{P}, d[, s::Symbol=:default])

returns the array of [`BezierSegment`](@ref)s `B` of a composite Bézier curve reconstructed
from an array `c` of points on the manifold `M` and an array of degrees `d`.

There are a few (reduced) representations that can get extended;
see also [`get_bezier_points`](@ref). For ease of the following, let $c=(c_1,\ldots,c_k)$
and $d=(d_1,\ldots,d_m)$, where $m$ denotes the number of components the composite Bézier
curve consists of. Then

* `:default` – $k = m + \sum_{i=1}^m d_i$ since each component requires one point more than
  its degree. The points are then orderen in tuples, i.e.
  ````math
  B = \bigl[ [c_1,\ldots,c_{d_1+1}], (c_{d_1+2},\ldots,c_{d_1+d_2+2}],\ldots, [c_{k-m+1+d_m},\ldots,c_{k}] \bigr]
  ````
* `:continuous` – $k = 1+ \sum_{i=1}{m} d_i$, since for a continuous curve start and end
  point of sccessive components are the same, so the very first start point and the end
  points are stored.
  ````math
  B = \bigl[ [c_1,\ldots,c_{d_1+1}], [c_{d_1+1},\ldots,c_{d_1+d_2+1}],\ldots, [c_{k-1+d_m},\ldots,b_{k}) \bigr]
  ````
* `:differentiable` – for a differentiable function additionally to the last explanation, also
  the second point of any segment was not stored except for the first segment.
  Hence $k = 2 - m + \sum_{i=1}{m} d_i$ and at a junction point $b_n$ with its given prior
  point $c_{n-1}$, i.e. this is the last inner point of a segment, the first inner point
  in the next segment the junction is computed as
  $b = \exp_{c_n}(-\log_{c_n} c_{n-1})$ such that the assumed differentiability holds
"""
function get_bezier_segments(
    M::Manifold,
    c::Array{P,1},
    d,
    s::Symbol=:default
) where {P}
    ((length(c) == d[1]) && (length(d)==1)) && return Tuple(c)
    return get_bezier_segments(M,c,d,Val(s))
end
function get_bezier_segments(::Manifold, c::Array{P,1}, d, ::Val{:default}) where {P}
    endindices = cumsum(d .+ 1)
    startindices = endindices - d
    return [BezierSegment(c[si:ei]) for (si,ei) in zip(startindices,endindices)]
end
function get_bezier_segments(::Manifold, c::Array{P,1}, d, ::Val{:continuous}) where {P}
    length(c) != (sum(d)+1) && error(
        "The number of control points $(length(c)) does not match (for degrees $(d) expcted $(sum(d)+1) points."
    )
    nums = d .+ [(i==length(d)) ? 1 : 0 for i ∈ 1:length(d)]
    endindices = cumsum(nums)
    startindices = cumsum(nums) - nums .+ 1
    return [
            [ # for all append the start of the new also as last
                BezierSegment([ c[startindices[i]:endindices[i]]..., c[startindices[i+1]] ])
                    for i ∈ 1:length(startindices)-1
            ]..., # despite for the last
            BezierSegment(c[startindices[end]:endindices[end]]),
        ]
end
function get_bezier_segments(M::Manifold, c::Array{P,1}, d, ::Val{:differentiable}) where {P}
    length(c) != (sum(d .-1)+2) && error(
        "The number of control points $(length(c)) does not match (for degrees $(d) expcted $(sum(d.-1)+2) points."
    )
    nums = d .+ [(i==1) ? 1 : -1 for i ∈ 1:length(d)]
    endindices = cumsum(nums)
    startindices = cumsum(nums) - nums .+ 1
    return [ # for all append the start of the new also as last
            BezierSegment(c[startindices[1]:endindices[1]]),
                [   BezierSegment([
                    c[endindices[i-1]],
                    exp(M,c[endindices[i-1]], -log(M,c[endindices[i-1]], c[endindices[i-1]-1])),
                    c[startindices[i]:endindices[i]]...
                ])
                    for i ∈ 2:length(startindices)
            ]..., # despite for the last
        ]
end