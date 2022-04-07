"""
plot_slope(x, y; slope=2, line_base=0, a=0, b=2.0, i=1,j=length(x))

plot the result from the [`check_gradient`](@ref)
on data `x,y` with two comparison lines

1) `line_base`+ t`slope`  as the goal the plot should have
2) à`+ `b`t` on the interval [`x[i]`, `x[j]`] for some (best fitting) comparison slope
"""
function plot_slope(x, y; slope=2, line_base=0, a=0, b=2.0, i=1, j=length(x))
    fig = plot(x, y; xaxis=:log, yaxis=:log, label="The error")
    s_line = [line_base + t^slope for t in x]
    plot!(fig, x, s_line; label="slope s=2")
    if (i != 0) && (j != 0)
        best_line = [a + t^b for t in x[i:j]]
        plot!(fig, x[i:j], best_line; label="best global regression (slope $b)")
    end
    gui(fig)
    return nothing
end
