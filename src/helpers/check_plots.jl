function plot_slope(x, y, slope, line_base)
    fig = plot(x, y; xaxis=:log, yaxis=:log, label="The error")
    s_line = [(line_base + t)^slope for t in x]
    plot!(fig, x, s_line; label="slope s=2")
    gui(fig)
    return nothing
end
