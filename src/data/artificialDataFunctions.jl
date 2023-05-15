#
# generate Artificial Data on several manifolds
#
#
#
@doc raw"""
    artificialIn_SAR_image([pts=500])
generate an artificial InSAR image, i.e. phase valued data, of size `pts` x
`pts` points.

This data set was introduced for the numerical examples in

> Bergmann, R., Laus, F., Steidl, G., Weinmann, A.:
> _Second Order Differences of Cyclic Data and Applications in Variational Denoising_
> SIAM J. Imaging Sci., 7(4), 2916–2953, 2014.
> doi: [10.1137/140969993](https://dx.doi.org/10.1137/140969993)
> arxiv: [1405.5349](https://arxiv.org/abs/1405.5349)

"""
function artificialIn_SAR_image(pts::Integer)
    # variables
    # rotation of ellipse
    aEll = 35.0
    cosE = cosd(aEll)
    sinE = sind(aEll)
    aStep = 45.0
    cosA = cosd(aStep)
    sinA = sind(aStep)
    # main and minor axis of the ellipse
    axes_inv = [6, 25]
    # values for the hyperboloid
    mid_point = [0.275; 0.275]
    radius = 0.18
    values = [range(-0.5, 0.5; length=pts)...]
    # Steps
    aSteps = 60.0
    cosS = cosd(aSteps)
    sinS = sind(aSteps)
    l = 0.075
    midP = [-0.475, -0.0625]#.125, .55]
    img = zeros(Float64, pts, pts)
    for j in eachindex(values), i in eachindex(values)
        # ellipse
        Xr = cosE * values[i] - sinE * values[j]
        Yr = cosE * values[j] + sinE * values[i]
        v = axes_inv[1] * Xr^2 + axes_inv[2] * Yr^2
        k1 = v <= 1.0 ? 10.0 * pi * Yr : 0.0
        # circle
        Xr = cosA * values[i] - sinA * values[j]
        Yr = cosA * values[j] + sinA * values[i]
        v = ((Xr - mid_point[1])^2 + (Yr - mid_point[2])^2) / radius^2
        k2 = v <= 1.0 ? 4.0 * pi * (1.0 - v) : 0.0
        #
        Xr = cosS * values[i] - sinS * values[j]
        Yr = cosS * values[j] + sinS * values[i]
        k3 = 0.0
        for m in 1:8
            in_range = (abs(Xr + midP[1] + m * l) + abs(Yr + midP[2] + m * l)) ≤ l
            k3 += in_range ? 2 * pi * (m / 8) : 0.0
        end
        img[i, j] = mod(k1 + k2 + k3 - pi, 2 * pi) + pi
    end
    return img
end

@doc raw"""
    artificial_S1_slope_signal([pts=500, slope=4.])

Creates a Signal of (phase-valued) data represented on the
[Circle](https://juliamanifolds.github.io/Manifolds.jl/stable/interface.html#ManifoldsBase.Manifold)` `[Manifold](https://juliamanifolds.github.io/Manifolds.jl/stable/interface.html#ManifoldsBase.Manifold) with increasing slope.

# Optional
* `pts` – (`500`) number of points to sample the function.
* `slope` – (`4.0`) initial slope that gets increased afterwards

This data set was introduced for the numerical examples in

> Bergmann, R., Laus, F., Steidl, G., Weinmann, A.:
> _Second Order Differences of Cyclic Data and Applications in Variational Denoising_
> SIAM J. Imaging Sci., 7(4), 2916–2953, 2014.
> doi: [10.1137/140969993](https://dx.doi.org/10.1137/140969993)
> arxiv: [1405.5349](https://arxiv.org/abs/1405.5349)

"""
function artificial_S1_slope_signal(pts::Integer=500, slope::Float64=4.0)
    t = range(0.0, 1.0; length=pts)
    f = zero(t)
    f[t .<= 1 / 6] .= -π / 2 .+ slope * π / 8 * t[t .<= 1 / 6]
    # In the following terms, the first max
    f[(1 / 6 .< t) .& (t .<= 1 / 3)] .=
        max(f[f .!= 0]...) .- slope * π / 4 * 1 / 6 .+
        slope * π / 4 .* t[(1 / 6 .< t) .& (t .<= 1 / 3)]
    f[(1 / 3 .< t) .& (t .<= 1 / 2)] .=
        max(f[f .!= 0]...) .- slope * π / 2 * 1 / 3 .+
        slope * π / 2 * t[(1 / 3 .< t) .& (t .<= 1 / 2)]
    f[(1 / 2 .< t) .& (t .<= 2 / 3)] .=
        max(f[f .!= 0]...) .- slope * π * 1 / 2 .+
        slope * π * t[(1 / 2 .< t) .& (t .<= 2 / 3)]
    f[(2 / 3 .< t) .& (t .<= 5 / 6)] .=
        max(f[f .!= 0]...) .- slope * 2 * π * 2 / 3 .+
        slope * 2 * π * t[(2 / 3 .< t) .& (t .<= 5 / 6)]
    f[5 / 6 .< t] .=
        max(f[f .!= 0]...) .- slope * 4 * π * 5 / 6 .+ slope * 4 * π * t[5 / 6 .< t]
    return mod.(f .+ Float64(π), Ref(2 * π)) .- Float64(π)
end

@doc raw"""
    artificial_S1_signal([pts=500])

generate a real-valued signal having piecewise constant, linear and quadratic
intervals with jumps in between. If the resulting manifold the data lives on,
is the [Circle](https://juliamanifolds.github.io/Manifolds.jl/stable/interface.html#ManifoldsBase.Manifold) the data is also wrapped
to $[-\pi,\pi)$.

# Optional
* `pts` – (`500`) number of points to sample the function

> Bergmann, R., Laus, F., Steidl, G., Weinmann, A.:
> _Second Order Differences of Cyclic Data and Applications in Variational Denoising_
> SIAM J. Imaging Sci., 7(4), 2916–2953, 2014.
> doi: [10.1137/140969993](https://dx.doi.org/10.1137/140969993)
> arxiv: [1405.5349](https://arxiv.org/abs/1405.5349)

"""
function artificial_S1_signal(pts::Integer=500)
    t = range(0.0, 1.0; length=pts)
    f = artificial_S1_signal.(t)
    return mod.(f .+ Float64(π), Ref(2 * π)) .- Float64(π)
end
@doc raw"""
    artificial_S1_signal(x)
evaluate the example signal $f(x), x ∈  [0,1]$,
of phase-valued data introduces in Sec. 5.1 of

> Bergmann, R., Laus, F., Steidl, G., Weinmann, A.:
> _Second Order Differences of Cyclic Data and Applications in Variational Denoising_
> SIAM J. Imaging Sci., 7(4), 2916–2953, 2014.
> doi: [10.1137/140969993](https://dx.doi.org/10.1137/140969993)
> arxiv: [1405.5349](https://arxiv.org/abs/1405.5349)

for values outside that intervall, this Signal is `missing`.
"""
function artificial_S1_signal(x::Real)
    if x < 0
        y = missing
    elseif x <= 1 / 4
        y = -24 * π * (x - 1 / 4)^2 + 3 / 4 * π
    elseif x <= 3 / 8
        y = 4 * π * x - π / 4
    elseif x <= 1 / 2
        y = -π * x - 3 * π / 8
    elseif x <= (3 * 0 + 19) / 32
        y = -(0 + 7) / 8 * π
    elseif x <= (3 * 1 + 19) / 32
        y = -(1 + 7) / 8 * π
    elseif x <= (3 * 2 + 19) / 32
        y = -(2 + 7) / 8 * π
    elseif x <= (3 * 3 + 19) / 32
        y = -(3 + 7) / 8 * π
    elseif x <= 1
        y = 3 / 2 * π * exp(8 - 1 / (1 - x)) - 3 / 4 * π
    else
        y = missing
    end
    return y
end

function artificial_S2_whirl_image end

function artificial_S2_rotation_image end

@doc raw"""
    artificial_S2_whirl_patch([pts=5])

create a whirl within the `pts`$\times$`pts` patch of
[Sphere](https://juliamanifolds.github.io/Manifolds.jl/stable/manifolds/sphere.html)(@ref)`(2)`-valued image data.

These patches are used within [`artificial_S2_whirl_image`](@ref).

# Optional Parameters
* `pts` – (`5`) size of the patch. If the number is odd, the center is the north
  pole.
"""
function artificial_S2_whirl_patch(pts::Int=5)
    patch = fill([0.0, 0.0, -1.0], pts, pts)
    scaleFactor = sqrt((pts - 1)^2 / 2) * 3 / π
    for i in 1:pts
        for j in 1:pts
            if i != (pts + 1) / 2 || j != (pts + 1) / 2
                α = atan((j - (pts + 1) / 2), (i - (pts + 1) / 2))
                β = sqrt((j - (pts + 1) / 2)^2 + (i - (pts + 1) / 2)^2) / scaleFactor
                patch[i, j] = [
                    sin(α) * sin(π / 2 - β), -cos(α) * sin(π / 2 - β), cos(π / 2 - β)
                ]
            end
        end
    end
    return patch
end

function artificial_S2_composite_bezier_curve end

@doc raw"""
    artificial_SPD_image([pts=64, stepsize=1.5])

create an artificial image of symmetric positive definite matrices of size
`pts`$\times$`pts` pixel with a jump of size `stepsize`.

This dataset was used in the numerical example of Section 5.2 of

> Bačák, M., Bergmann, R., Steidl, G., Weinmann, A.:
> _A Second Order Non-Smooth Variational Model for Restoring Manifold-Valued Images_
> SIAM J. Sci. Comput. 38(1), A567–A597, 2016.
> doi: [10.1137/15M101988X](https://dx.doi.org/10.1137/15M101988X)
> arxiv: [1506.02409](https://arxiv.org/abs/1506.02409)

"""
function artificial_SPD_image(pts::Int=64, stepsize=1.5)
    r = range(0; stop=1 - 1 / pts, length=pts)
    v1 = abs.(2 * pi .* r .- pi)
    v2 = pi .* r
    v3 = range(0; stop=3 * (1 - 1 / pts), length=2 * pts)
    data = fill(Matrix{Float64}(I, 3, 3), pts, pts)
    for row in 1:pts
        for col in 1:pts
            A = [cos(v1[col]) -sin(v1[col]) 0.0; sin(v1[col]) cos(v1[col]) 0.0; 0.0 0.0 1.0]
            B = [1.0 0.0 0.0; 0.0 cos(v2[row]) -sin(v2[row]); 0.0 sin(v2[row]) cos(v2[row])]
            C = [
                cos(v1[mod(col - row, pts) + 1]) 0 -sin(v1[mod(col - row, pts) + 1])
                0.0 1.0 0.0
                sin(v1[mod(col - row, pts) + 1]) 0.0 cos(v1[mod(col - row, pts) + 1])
            ]
            scale = [
                1 + stepsize / 2 * ((row + col) > pts ? 1 : 0)
                1 + v3[row + col] - stepsize * (col > pts / 2 ? 1 : 0)
                4 - v3[row + col] + stepsize * (row > pts / 2 ? 1 : 0)
            ]
            data[row, col] = Matrix(Symmetric(A * B * C * Diagonal(scale) * C' * B' * A'))
        end
    end
    return data
end
@doc raw"""
    artificial_SPD_image2([pts=64, fraction=.66])

create an artificial image of symmetric positive definite matrices of size
`pts`$\times$`pts` pixel with right hand side `fraction` is moved upwards.

This data set was introduced in the numerical examples of Section of

> Bergmann, R., Persch, J., Steidl, G.:
> _A Parallel Douglas Rachford Algorithm for Minimizing ROF-like Functionals on Images with Values in Symmetric Hadamard Manifolds_
> SIAM J. Imaging. Sci. 9(3), pp. 901-937, 2016.
> doi: [10.1137/15M1052858](https://dx.doi.org/10.1137/15M1052858)
> arxiv: [1512.02814](https://arxiv.org/abs/1512.02814)

"""
function artificial_SPD_image2(pts=64, fraction=0.66)
    Zl = 4.0 * Matrix{Float64}(I, 3, 3)
    # create a first matrix
    α = 2.0 * π / 3
    β = π / 3
    B = [1.0 0.0 0.0; 0.0 cos(β) -sin(β); 0.0 sin(β) cos(β)]
    A = [cos(α) -sin(α) 0.0; sin(α) cos(α) 0.0; 0.0 0.0 1.0]
    Zo = Matrix(Symmetric(A * B * Diagonal([2.0, 4.0, 8.0]) * B' * A'))
    # create a second matrix
    α = -4.0 * π / 3
    β = -π / 3
    B = [1.0 0.0 0.0; 0.0 cos(β) -sin(β); 0.0 sin(β) cos(β)]
    A = [cos(α) -sin(α) 0.0; sin(α) cos(α) 0.0; 0.0 0.0 1.0]
    Zt = A * B * Diagonal([8.0 / sqrt(2.0), 8.0, sqrt(2.0)]) * B' * A'
    data = fill(Matrix{Float64}(I, 3, 3), pts, pts)
    M = SymmetricPositiveDefinite(3)
    for row in 1:pts
        for col in 1:pts
            # (a) from Zo a part to Zt
            C = Zo
            if (row > 1) # in X direction
                C = exp(
                    M,
                    C,
                    log(M, C, Zt),
                    (row - 1) / (2 * (pts - 1)) + ((row > fraction * pts) ? 1 / 2 : 0.0),
                )
            end
            if (col > 1) # and then in Y direction
                C = exp(
                    M,
                    C,
                    vector_transport_to(
                        M, Symmetric(Zo), log(M, Zo, Zl), Symmetric(C), ParallelTransport()
                    ),
                    (col - 1.0) / (pts - 1),
                )
            end
            data[row, col] = C
        end
    end
    return data
end

@doc raw"""
    artificial_S2_lemniscate(p [,pts=128,a=π/2,interval=[0,2π])

Generate a Signal on the [`Sphere`](https://juliamanifolds.github.io/Manifolds.jl/stable/manifolds/sphere.html) $\mathbb S^2$ by creating the
[Lemniscate of Bernoulli](https://en.wikipedia.org/wiki/Lemniscate_of_Bernoulli)
in the tangent space of `p` sampled at `pts` points and use `exp` to get a
signal on the [`Sphere`](https://juliamanifolds.github.io/Manifolds.jl/stable/manifolds/sphere.html).

# Input
* `p` – the tangent space the Lemniscate is created in
* `pts` – (`128`) number of points to sample the Lemniscate
* `a` – (`π/2`) defines a half axis of the Lemniscate to cover a
   half sphere.
* `interval` – (`[0,2*π]`) range to sample the lemniscate at, the default value
  refers to one closed curve

This dataset was used in the numerical example of Section 5.1 of

> Bačák, M., Bergmann, R., Steidl, G., Weinmann, A.:
> _A Second Order Non-Smooth Variational Model for Restoring Manifold-Valued Images_
> SIAM J. Sci. Comput. 38(1), A567–A597, 2016.
> doi: [10.1137/15M101988X](https://dx.doi.org/10.1137/15M101988X)
> arxiv: [1506.02409](https://arxiv.org/abs/1506.02409)

"""
function artificial_S2_lemniscate(
    p, pts::Integer=128, a::Float64=π / 2.0, interval::Array{Float64,1}=[0.0, 2.0 * π]
)
    return artificial_S2_lemniscate.(Ref(p), range(interval[1], interval[2]; length=pts), a)
end
