using ColorTypes, Colors
export renderAsymptote, asyExportS2
"""
    renderAsymptote(filename, exportFct; render=4, format="png", ...)
render an exported `asy`.

# INPUT
* `filename` : filename of the exported `asy` and rendered image
* `exportFct`: a function creating an `asy` file with `kwargs` as optional
  arguments and the `filename` string as its only mandatory argument

# Keyword Arguments
the default values are given in brackets
* `render`   : (4) render level of asymptote, i.e. its `-render` option
* `format`   : (`"png"`) final rendered format, i.e. asymptote's `-f` option
all further keyword arguments are passed down to the `exportFct` call.
"""
function renderAsymptote(filename, exportFct; render::Int=4, format="png", kwargs...)
    renderCmd = `asy -render $(render) -f $(format) $(filename)`
    exportFct(filename; kwargs...)
    run(renderCmd)
end
function asyExportS2(filename::String;
    points::Array{Array{SnPoint,1},1} = Array{Array{SnPoint,1},1}(undef,0),
    curves::Array{Array{SnPoint,1},1} = Array{Array{SnPoint,1},1}(undef,0),
    tVectors::Array{Array{TVectorE{SnTVector,SnPoint},1},1} = Array{Array{TVectorE{SnTVector,SnPoint},1},1}(undef,0),
    colors::Dict{Symbol, Array{T,1} where T}  = Dict{Symbol,Array{RGBA{Float},1}}(),
    arrowHeadSize::Float64 = 6.,
    cameraPosition::Tuple{Float64,Float64,Float64} = (1., 1., 0.),
    dotSize::Float64 = 1.0,
    dotSizes::Array{Float64,1} = fill(dotSize,size(points)),
    target::Tuple{Float64,Float64,Float64} = (0.,0.,0.),
    )
    io = open(filename,"w")
    try
        #
        # Header
        # ---
        write(io,string("import settings;\nimport three;\nimport solids;",
                    "unitsize(4cm);\n\n",
                    "currentprojection=perspective( ",
                    "camera = $(cameraPosition), ",
                    "target = $(target) );\n",
                    "currentlight=nolight;\n\n",
                    "revolution S=sphere(O,1);\n",
                    "draw(surface(S), surfacepen=lightgrey+opacity(.6), ",
                    "meshpen=0.6*white+linewidth(.5pt));\n")
        );
        write(io,"\n/*\n  Colors\n*/\n")
        j=0
        for (key,value) in colors
            penPrefix = "$(j)"
            penPrefix = "$(j)"
            if key==:points
                penPrefix="point"
            elseif key==:curves
                penPrefix="curve"
            elseif key==:tvectors
                penPrefix="tVector"
            end
            i=0
            for c in value
                i=i+1;
                write(io,string("pen $(penPrefix)Style$(i) = ",
                    "rgb($(red(c)),$(green(c)),$(blue(c)))",
                    (key==:points) ? "+linewidth($(dotSizes[i])pt)" : "",
                    "+opacity($(alpha(c)));\n"));
            end
        end
        write(io,"\n/*\n  Exported Points\n*/\n")
        i=0
        for pSet in points
            i=i+1
            for point in pSet
                write(io,string("dot( (",string([string(v,",") for v in getValue(point)]...)[1:end-1],"), pointStyle$(i));\n"));
            end
        end
        i=0
        for curve in curves
            i=i+1
            write(io,"path3 p$(i) = ");
            j=0
            for point in pSet
                j=j+1
                if j>1
                    write( io," .. (",[string(v,",") for v in getValue(point)]...,")" )
                else
                    write( io,"(",[string(v,",") for v in getValue(point)]...,")" )
                end
                write( io,string(";\n draw(p$(i), curveStyle$(i));\n") );
            end
        end
        i=0
        for tVecs in tVectors
            i=i+1
            j=0
            for vector in tVecs
                j=j+1
                write(io,string("draw( (",
                    string( [string(v,",") for v in getBase(vector)]...)[1:end-1],")--(",
                    string( [string(v,",") for v in getValue(vectort)]...)[1:end-1],
                    "), tVectorStyle$(j),Arrow3($(arrowHeadSize)));\n"));
            end
        end
    finally
        close(io)
    end
end
