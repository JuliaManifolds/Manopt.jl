@doc raw"""
    check_gradient(M, F, gradF, p=random_point(M), X=random_tangent(M,p))

Check numerivcally whether the gradient `gradF` of `F` is correct.

This implements the method described in Section 4.8 [^Boumal2022].

# Keyword arguments

* `N` (`101`) – number of points to check within the `log_range` default range ``[10^{-8},10^{0}]``
* `error` - (`false`) throw an error message if the gradient is wrong
* `io` – (`nothing`) provide an `IO` to print the check result to
* `limits` (`(10e-8,1)`) specify the limits in the `log_range`
* `log_range` (`range(limits[1], limits[2]; length=N)`) - specify the range of points (in log scale) to sample the gradient line
* `plot`- (`false`) whether to plot the resulting check (if `Plots.jl` is loaded)
* `retraction_method` - (`default_retraction_method(M)`) retraction method to use for the check
* `slope_tol` – (`0.1`) tolerance for the slope (global) of the approximation

[^Boumal2022]:
    > Boumal, N.: _An Introduction to Optimization on Smooth Manifolds_, book in preparation,
    > 2022. url: [http://www.nicolasboumal.net/book](http://www.nicolasboumal.net/book).
"""
function check_gradient(
    M::AbstractManifold,
    F,
    gradF,
    p=random_point(M),
    X=random_tangent(M, p);
    plot=false,
    error=false,
    io::Union{IO,Nothing}=nothing,
    limits=(-8.0, 0.0),
    N=101,
    log_range=range(limits[1], limits[2]; length=N),
    retraction_method=default_retraction_method(M),
    slope_tol=0.1,
    window=nothing,
)
    gradient = gradF(M, p)
    Xn = X ./ norm(M, p, X) # normalize tangent direction
    is_vector(M, p, gradient, error)
    # function for the directional derivative
    df(M, p, Y) = inner(M, p, gradient, Y)
    #
    T = exp10.(log_range)
    # points p_i to evaluate our error function at
    points = map(t -> retract(M, p, Xn, t, retraction_method), T)
    # F(p_i)
    costs = [F(M, pi) for pi in points]
    # linearized
    linearized = map(t -> F(M, p) + t * df(M, p, Xn), T)
    L = abs.(costs .- linearized)
    # global fit a + bx
    x = log_range[L .> 0]
    y = log10.(L[L .> 0])
    (a, b) = find_best_slope_window(x, y, length(x))[1:2]
    if isapprox(b, 2.0; atol=slope_tol)
        plot &&
            plot_slope(T[L .> 0], L[L .> 0]; line_base=costs[1], a=a, b=b, i=1, j=length(y))
        (io !== nothing) && print(
            io,
            "You gradients slope is globally $(@sprintf("%.4f", b)), so within 2 ± $(slope_tol).\n",
        )
        return true
    end
    # otherwise
    # find best contiguous window of length w
    (ab, bb, ib, jb) = find_best_slope_window(log_range, log10.(L), window)
    plot && plot_slope(T[L .> 0], L[L .> 0]; line_base=costs[1], a=ab, b=bb, i=ib, j=jb)
    (io !== nothing) && print(
        io,
        "You gradient fits best on [$(T[ib]),$(T[jb])] with slope  $(@sprintf("%.4f", bb)), but globally your slope $(@sprintf("%.4f", b)) is outside of the tolerance 2 ± $(slope_tol).\n",
    )
    return false
end

"""
    (a,b,i,j) = find_best_slope_window(X,Y,window=nothing; slope=2.0)

Check data X,Y for the best contiguous interval with a regression line closest to `slope`.

If the window is set to `nothing` (default), all window sizes `2,...,length(X)` are checked.
You can also specify a window size or an array of window sizes.

For each window size , all its translates in the data are checked.
For all these (shifted) windows the regression line is computed (i.e. `a,b` in `a + t*b`)
and the best line is computed.

From the best line the following data is returned

* `a`, b` specifying the regression line `a + tb`
* `i`, `j` determining the window, i.e the regression line stems from data `X[i], ..., X[j]`
"""
function find_best_slope_window(X, Y, window=nothing; slope=2.0)
    n = length(X)
    a_best = 0
    b_best = 0
    i_best = 0
    j_best = 0
    for w in (window === nothing ? (2:n) : [window...])
        for j in 1:(n - w + 1)
            x = X[j:(j + w - 1)]
            y = Y[j:(j + w - 1)]
            # fit a line a + bx
            c = cor(x, y)
            b = std(y) / std(x) * c
            a = mean(y) - b * mean(x)
            # look for the best error relative to log scale interval
            r = (maximum(x) - minimum(x))
            if abs(b - slope) < abs(b_best - slope)
                a_best = a
                b_best = b
                i_best = j
                j_best = j + w
            end
        end
    end
    return (a_best, b_best, i_best, j_best)
end
