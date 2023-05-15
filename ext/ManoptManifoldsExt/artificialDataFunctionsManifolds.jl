
@doc raw"""
    artificial_S2_composite_bezier_curve()

Create the artificial curve in the `Sphere(2)` consisting of 3 segments between the four
points

````math
p_0 = \begin{bmatrix}0&0&1\end{bmatrix}^{\mathrm{T}},
p_1 = \begin{bmatrix}0&-1&0\end{bmatrix}^{\mathrm{T}},
p_2 = \begin{bmatrix}-1&0&0\end{bmatrix}^{\mathrm{T}},
p_3 = \begin{bmatrix}0&0&-1\end{bmatrix}^{\mathrm{T}},
````

where each segment is a cubic Bezér curve, i.e. each point, except $p_3$ has a first point
within the following segment $b_i^+$, $i=0,1,2$ and a last point within the previous
segment, except for $p_0$, which are denoted by $b_i^-$, $i=1,2,3$.
This curve is differentiable by the conditions $b_i^- = \gamma_{b_i^+,p_i}(2)$, $i=1,2$,
where $\gamma_{a,b}$ is the [`shortest_geodesic`](https://juliamanifolds.github.io/Manifolds.jl/stable/interface.html#ManifoldsBase.shortest_geodesic-Tuple{Manifold,Any,Any}) connecting $a$ and $b$.
The remaining points are defined as

````math
\begin{aligned}
    b_0^+ &= \exp_{p_0}\frac{\pi}{8\sqrt{2}}\begin{pmatrix}1&-1&0\end{pmatrix}^{\mathrm{T}},&
    b_1^+ &= \exp_{p_1}-\frac{\pi}{4\sqrt{2}}\begin{pmatrix}-1&0&1\end{pmatrix}^{\mathrm{T}},\\
    b_2^+ &= \exp_{p_2}\frac{\pi}{4\sqrt{2}}\begin{pmatrix}0&1&-1\end{pmatrix}^{\mathrm{T}},&
    b_3^- &= \exp_{p_3}-\frac{\pi}{8\sqrt{2}}\begin{pmatrix}-1&1&0\end{pmatrix}^{\mathrm{T}}.
\end{aligned}
````

This example was used within minimization of acceleration of the paper

> Bergmann, R., Gousenbourger, P.-Y.:
> _A variational model for data fitting on manifolds by minimizing the acceleration of a Bézier curve_,
> Front. Appl. Math. Stat. 12, 2018.
> doi: [10.3389/fams.2018.00059](https://dx.doi.org/10.3389/fams.2018.00059)
> arxiv: [1807.10090](https://arxiv.org/abs/1807.10090)

"""
function Manopt.artificial_S2_composite_bezier_curve()
    M = Sphere(2)
    d0 = [0.0, 0.0, 1.0]
    d1 = [0.0, -1.0, 0.0]
    d2 = [-1.0, 0.0, 0.0]
    d3 = [0.0, 0.0, -1.0]
    #
    # control points - where b1- and b2- are constructed by the C1 condition
    #
    # We define three segments: 1
    b00 = d0 # also known as p0
    ξ0 = π / (8.0 * sqrt(2.0)) .* [1.0, -1.0, 0.0] # staring direction from d0
    b01 = exp(M, d0, ξ0) # b0+
    ξ1 = π / (4.0 * sqrt(2)) .* [1.0, 0.0, 1.0]
    # b02 or b1- and b11 or b1+ are constructed by this vector with opposing sign
    # to achieve a C1 curve
    b02 = exp(M, d1, ξ1)
    b03 = d1
    # 2
    b10 = d1
    b11 = exp(M, d1, -ξ1) # yields c1 condition
    ξ2 = -π / (4 * sqrt(2)) .* [0.0, 1.0, -1.0]
    b12 = exp(M, d2, ξ2)
    b13 = d2
    # 3
    b20 = d2
    b21 = exp(M, d2, -ξ2)
    ξ3 = π / (8.0 * sqrt(2)) .* [-1.0, 1.0, 0.0]
    b22 = exp(M, d3, ξ3)
    b23 = d3
    # hence the matrix of controlpoints for the curve reads
    return [
        BezierSegment([b00, b01, b02, b03]),
        BezierSegment([b10, b11, b12, b13]),
        BezierSegment([b20, b21, b22, b23]),
    ]
end

@doc raw"""
    artificial_S2_rotation_image([pts=64, rotations=(.5,.5)])

Create an image with a rotation on each axis as a parametrization.

# Optional Parameters
* `pts` – (`64`) number of pixels along one dimension
* `rotations` – (`(.5,.5)`) number of total rotations performed on the axes.

This dataset was used in the numerical example of Section 5.1 of

> Bačák, M., Bergmann, R., Steidl, G., Weinmann, A.:
> _A Second Order Non-Smooth Variational Model for Restoring Manifold-Valued Images_
> SIAM J. Sci. Comput. 38(1), A567–A597, 2016.
> doi: [10.1137/15M101988X](https://dx.doi.org/10.1137/15M101988X)
> arxiv: [1506.02409](https://arxiv.org/abs/1506.02409)

"""
function Manopt.artificial_S2_rotation_image(
    pts::Int=64, rotations::Tuple{Float64,Float64}=(0.5, 0.5)
)
    M = Sphere(2)
    img = fill(zeros(3), pts, pts)
    north = [1.0, 0.0, 0.0]
    Rxy(a) = [cos(a) -sin(a) 0.0; sin(a) cos(a) 0.0; 0.0 0.0 1]
    Rxz(a) = [cos(a) 0.0 -sin(a); 0.0 1.0 0.0; sin(a) 0.0 cos(a)]
    for i in 1:pts
        for j in 1:pts
            x = i / pts * 2π * rotations[1]
            y = j / pts * 2π * rotations[2]
            img[i, j] = Rxy(x + y) * Rxz(x - y) * [0, 0, 1]
        end
    end
    return img
end

@doc raw"""
    artificial_S2_lemniscate(p, t::Float64; a::Float64=π/2)

Generate a point from the signal on the [`Sphere`](https://juliamanifolds.github.io/Manifolds.jl/stable/manifolds/sphere.html) $\mathbb S^2$ by
creating the [Lemniscate of Bernoulli](https://en.wikipedia.org/wiki/Lemniscate_of_Bernoulli)
in the tangent space of `p` sampled at `t` and use èxp` to obtain a point on
the [`Sphere`](https://juliamanifolds.github.io/Manifolds.jl/stable/manifolds/sphere.html).

# Input
* `p` – the tangent space the Lemniscate is created in
* `t` – value to sample the Lemniscate at

# Optional Values
 * `a` – (`π/2`) defines a half axis of the Lemniscate to cover a
   half sphere.

This dataset was used in the numerical example of Section 5.1 of

> Bačák, M., Bergmann, R., Steidl, G., Weinmann, A.:
> _A Second Order Non-Smooth Variational Model for Restoring Manifold-Valued Images_
> SIAM J. Sci. Comput. 38(1), A567–A597, 2016.
> doi: [10.1137/15M101988X](https://dx.doi.org/10.1137/15M101988X)
> arxiv: [1506.02409](https://arxiv.org/abs/1506.02409)

 """
function Manopt.artificial_S2_lemniscate(p, t::Float64, a::Float64=π / 2.0)
    M = Sphere(2)
    tP = 2.0 * Float64(p[1] >= 0.0) - 1.0 # Take north or south pole
    base = [0.0, 0.0, tP]
    xc = a * (cos(t) / (sin(t)^2 + 1.0))
    yc = a * (cos(t) * sin(t) / (sin(t)^2 + 1.0))
    tV = vector_transport_to(M, base, [xc, yc, 0.0], p, ParallelTransport())
    return exp(M, p, tV)
end

@doc raw"""
    artificial_S2_whirl_image([pts=64])

Generate an artificial image of data on the 2 sphere,

# Arguments
* `pts` – (`64`) size of the image in `pts`$\times$`pts` pixel.

This example dataset was used in the numerical example in Section 5.5 of

> Laus, F., Nikolova, M., Persch, J., Steidl, G.:
> _A Nonlocal Denoising Algorithm for Manifold-Valued Images Using Second Order Statistics_,
> SIAM J. Imaging Sci., 10(1), 416–448, 2017.
> doi: [ 10.1137/16M1087114](https://dx.doi.org/10.1137/16M1087114)
> arxiv: [1607.08481](https://arxiv.org/abs/1607.08481)

It is based on [`artificial_S2_rotation_image`](@ref) extended by small whirl patches.
"""
function Manopt.artificial_S2_whirl_image(pts::Int=64)
    M = Sphere(2)
    img = artificial_S2_rotation_image(pts, (0.5, 0.5))
    # Set WhirlPatches
    sc = pts / 64
    patchSizes = floor.(sc .* [9, 9, 9, 9, 11, 11, 11, 15, 15, 15, 17, 21])
    patchCenters =
        Integer.(
            floor.(
                sc .*
                [[35, 7] [25, 41] [32, 25] [7, 60] [10, 5] [41, 58] [11, 41] [23, 56] [
                    38, 45
                ] [16, 28] [55, 42] [51, 16]],
            ),
        )
    patchSigns = [1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1]
    for i in 1:length(patchSizes)
        pS = Integer(patchSizes[i])
        pSH = Integer(floor((patchSizes[i] - 1) / 2))
        pC = patchCenters[:, i]
        pC = max.(Ref(1), pC .- pS) .+ pS
        pSgn = patchSigns[i]
        s = pS % 2 == 0 ? 1 : 0
        r = [pC[1] .+ ((-pSH):(pSH + s)), pC[2] .+ ((-pSH):(pSH + s))]
        patch = artificial_S2_whirl_patch(pS)
        if pSgn == -1 # opposite ?
            patch = -patch
        end
        img[r...] = patch
    end
    return img
end
