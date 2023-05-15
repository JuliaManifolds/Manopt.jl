module ManoptPlotsExt

if isdefined(Base, :get_extension)
    using Plots
else
    # imports need to be relative for Requires.jl-based workflows:
    # https://github.com/JuliaArrays/ArrayInterface.jl/pull/387
    using ..Plots
end
    include("check_plots.jl")
end