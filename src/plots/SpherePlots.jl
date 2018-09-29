#
# SpherePlots.jl –
#
# import RecipesBase: plot
# import Plots: scatter3d, scatter3d!, surface!, surface, current
# export plot
# """
#     plot(M::Sphere, signals) – plot signals on the two sphere
# """
# function plot(M::Sphere,signals::Array{Array{SnPoint,1},1})
#     if manifoldDimension(M) > 2
#         error("Plot only works for functions on S2")
#     end
#     plotlyjs()
#     n = 180
#     u = linspace(0,2*π,n);
#     v = linspace(0,π,n);
#     x = cos.(u) * sin.(v)';
#     y = sin.(u) * sin.(v)';
#     z = ones(n) * cos.(v)';
#     surface(x,y,z,opacity=.95,
#     f = :white,
#     axis=nothing,colorbar=nothing,legend=nothing,border=:none)
#     for signal in signals
#         if any(manifoldDimension.(signal) .> 2)
#             error("This plot method only works for points on S2.");
#         end
#         xS = [ getValue(p)[1] for p in signal]
#         yS = [ getValue(p)[2] for p in signal]
#         zS = [ getValue(p)[3] for p in signal]
#         scatter3d!(xS,yS,zS)
#     end
#     return current()
# end
# plot(M::Sphere, signal::Array{SnPoint,1}) = plot(M,[signal])
