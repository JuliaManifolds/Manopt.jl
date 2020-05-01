@doc raw"""
    de_casteljau(M::Manifold, b::NTuple{N,P}) -> Function

return the Bézier curve $\beta(\cdot;p_0,\ldots,b_n)\colon [0,1] \to \mathcal M$ defined
by the control points $b_0,\ldots,b_n\in\mathcal M$, $n\in \mathbb N$, implemented using
the De Casteljau's algorithm[^Casteljau1959][^Casteljau1963] gneralized to
manifolds[^PopielNoakes2007]: Let $\gamma_{a,b}(t)$ denote the
shortest geodesic connecting $a,b\in\mathcal M$. Then the curve is defined by the recursion

````math
\begin{aligned}
    \beta(t;b_0,b_1) &= \gamma_{b_0,b_1}(t)\\
    \beta(t;b_0,\ldots,b_n) &= \gamma_{\beta(t;b_0,\ldots,b_{n-1}), \beta(t;b_1,\ldots,b_n)}(t),
\end{aligned}
````
and `P` is the type of a point on the `Manifold` `M`.

    de_casteljau(M::Manifold, B::Array{NTuple{N,P},1}) -> Function

Given a vector of control points
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

    de_casteljau(M::Manifold, b::NTuple{N,P}, t::Real)
    de_casteljau(M::Manifold, B::Arrray{NTuple{N,P},1}, t::Real)
    de_casteljau(M::Manifold, b::NTuple{N,P}, T::AbstractVector) -> AbstractVector
    de_casteljau(M::Manifold, B::Arrray{NTuple{N,P},1}, T::AbstractVector) -> AbstractVector

Return the point at time `t` or points at times `t` in `T` along the Bézier curve.


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
function de_casteljau(M::Manifold, b::NTuple{2,P}) where {P}
    return t -> shortest_geodesic(M, b[1], b[2], t)
end
function de_casteljau(M::Manifold, b::NTuple{N,P}) where {N,P}
    return t-> shortest_geodesic(
        M,
        de_casteljau(M,b[1:end-1],t),
        de_casteljau(M,b[2:end],t),
        t,
    )
end
function de_casteljau(M::Manifold, B::Array{P,1}) where {P}
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
    get_bezier_junction_tangent_vectors(M::Manifold, B::Array{NTuple{N,P},1})
    get_bezier_junction_tangent_vectors(M::Manifold, b::NTuple{N,P})

returns the tangent vectors at start and end points of the composite Bézier curve
pointing from a junction point to the first and last
inner control points for each segment of the composite Bezier curve specified by
the control points `B`, either a vector of segments of controlpoints.
"""
function get_bezier_junction_tangent_vectors(
    M::Manifold,
    B::Array{P,1}
) where {P}
    return cat(
        [ [log(M, b[1], b[2]), log(M, b[end], b[end-1])] for b in B ]...,
        ;
        dims=1,
    )
end
function get_bezier_junction_tangent_vectors(M,b::NTuple{N,P}) where {P,N}
    return get_bezier_junction_tangent_vectors(M,[b])
end

@doc raw"""
    get_bezier_junctions(M::Manifold, B::Array{NTuple{N,P},1})
    get_bezier_junctions(M::Manifold, b::NTuple{N,P})

returns the start and end point(s) of the segments of the composite Bézier curve
specified by the control points `B`. For just one segment `b`, its start and end points
are returned.
"""
function get_bezier_junctions(M::Manifold, B::Array{P,1}, double_inner=false) where {P}
    return cat(
        [ double_inner ? [b[[1,end]]...] : [b[1]] for b in B ]...,
        double_inner ? [] : [last(last(B))];
        dims=1,
    )
end
function get_bezier_junctions(M::Manifold, b::NTuple{N,P},double_inner=false) where {P,N}
    return get_bezier_junctions(M,[b])
end

@doc raw"""
    get_bezier_inner_points(M::Manifold, B::Array{NTuple{N,P},1} )
    get_bezier_inner_points(M::Manifold, b::NTuple{N,P})

returns the inner (i.e. despite start and end) points of the segments of the
composite Bézier curve specified by the control points `B`. For a single segment `b`,
its inner points are returned
"""
function get_bezier_inner_points(M::Manifold, B::Array{P,1}) where {P}
    return cat(
        [[b[2:end-1]...] for b in B]...;
        dims=1,
    )
end

@doc raw"""
    get_bezier_points(M::MAnifold, B:Array{NTuple{N,P},1}, reduce::Symbol=:none)
    get_bezier_points(M::Manifold, b::NTuple{N,P}, reduce::Symbol=:none)

returns the control points of the segments of the composite Bézier curve
specified by the control points `B`, either a vector of segments of
controlpoints or a.

This method reduces the points depending on the optional `reduce` symbol

* `:none` – no reduction is performed
* `:continuous` – for a continuous function, the junction points are doubled at
  $b_{0,i}=b_{n_{i-1},i-1}$, so only $b_{0,i}$ is in the vector.
* `:differentiable` – for a differentiable function additionally
  $\log_{b_{0,i}}b_{1,i} = -\log_{b_{n_{i-1},i-1}}b_{n_{i-1}-1,i-1}$ holds.
  hence $b_{n_{i-1}-1,i-1}$ is ommited.

If only one segment is given, all points of `b` – i.e. `b` itself is returned.
"""
function get_bezier_points(M::Manifold, B::Array{P,1}, reduce::Symbol=:none) where {P}
    return get_bezier_points(M,B,Val(reduce))
end
function get_bezier_points(M::Manifold, B::Array{P,1}, ::Val{:none}) where {P}
    return cat( [ [b...] for b in B]...; dims=1)
end
function get_bezier_points(M::Manifold, B::Array{P,1}, ::Val{:continuous}) where {P}
    return cat(
        [ [b[1:end-1]...] for b in B]...,
        [last(last(B))];
        dims=1,
    )
end
function get_bezier_points(M::Manifold, B::Array{P,1}, ::Val{:differentiable}) where {P}
    return cat(
        [ [b[1:end-2]...] for b in B]...,
        [last(B)[end-1]],
        [last(B)[end]];
        dims=1,
    )
end
get_bezier_points(M::Manifold, b::NTuple{N,P}, s::Symbol=:none) where {P,N} = b
