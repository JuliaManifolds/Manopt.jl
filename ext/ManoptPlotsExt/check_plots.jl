"""
plot_slope(x, y; slope=2, line_base=0, a=0, b=2.0, i=1,j=length(x))

plot the result from the error check functions, e.g.
[`check_gradient`](@ref), [`check_differential`](@ref), [`check_Hessian`](@ref)
on data `x,y` with two comparison lines

1) `line_base` + t`slope`  as the global slope the plot should have
2) `a` + `b*t` on the interval [`x[i]`, `x[j]`] for some (best fitting) comparison slope
"""
function plot_slope(x, y; slope=2, line_base=0, a=0, b=2.0, i=1, j=length(x))
    fig = plot(
        x,
        y;
        xaxis=:log,
        yaxis=:log,
        label="\$E(t)\$",
        linewidth=3,
        legend=:topleft,
        color=:lightblue,
    )
    s_line = [exp10(line_base + t * slope) for t in log10.(x)]
    plot!(
        fig, x, s_line; label="slope s=$slope", linestyle=:dash, color=:black, linewidth=2
    )
    if (i != 0) && (j != 0)
        best_line = [exp10(a + t * b) for t in log10.(x[i:j])]
        plot!(
            fig,
            x[i:j],
            best_line;
            label="best slope $(@sprintf("%.4f", b))",
            color=:blue,
            linestyle=:dot,
            linewidth=2,
        )
    end
    return fig
end
