#
# SpherePlots.jl –
#
import PyPlot: plot, surf, plot_wireframe, xlim, ylim, zlim, axis, scatter3D
export plot
"""
    plot(M::Sphere, signals) – plot signals on the two sphere
"""
function plot(M::Sphere,signals::Array{Array{SnPoint,1},1})
    if manifoldDimension(M) > 2
        error("Plot only works for functions on S2")
    end
    # TODO: Sphere resolution as options
    n = 180
    u = linspace(0,2*π,n);
    v = linspace(0,π,n);
    x = cos.(u) * sin.(v)';
    y = sin.(u) * sin.(v)';
    z = ones(n) * cos.(v)';
    surf(x,y,z, color=[.9,.9,.9], alpha=0.2)
    xlim(-1,1); ylim(-1,1); zlim(-1,1); axis("off");
    plot_wireframe(x,y,z, rstride=6,cstride=6,colors=[0,0,0],linewidth=0.1, alpha=0.3)
    # TODO: Signal styles as options
    # including the scale for visibility
    pts_scale = 1.01;
    for signal in signals
        if any(manifoldDimension.(signal) .> 2)
            error("This plot method only works for points on S2.");
        end
        xS = [pts_scale*p.value[1] for p in signal]
        yS = [pts_scale*p.value[2] for p in signal]
        zS = [pts_scale*p.value[3] for p in signal]
        scatter3D(xS,yS,zS,alpha=1.0)
    end
end
plot(M::Sphere, signal::Array{SnPoint,1}) = plot(M,[signal])
