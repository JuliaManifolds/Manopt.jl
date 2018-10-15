using ColorTypes, Colors
export renderAsymptote, asyExportS2
"""
    renderAsymptote(filename, exportFct; render=4, format="png", ...)
render an exported `asy`.

# Input
* `filename` : filename of the exported `asy` and rendered image
* `exportFct`: a function creating an `asy` file with `kwargs` as optional
  arguments and the `filename` string as its only mandatory argument

# Keyword Arguments
the default values are given in brackets
* `render`   : (4) render level of asymptote, i.e. its `-render` option
* `format`   : (`"png"`) final rendered format, i.e. asymptote's `-f` option
all further keyword arguments are passed down to the `exportFct` call.

# See also
[`asyExportS2`](@ref)
"""
function renderAsymptote(filename, exportFct; render::Int=4, format="png", kwargs...)
    renderCmd = `asy -render $(render) -f $(format) $(filename)`
    exportFct(filename; kwargs...)
    run(renderCmd)
end
@doc doc"""
    asyExportS2(filename; points, curves, tVectors, colors, options...)
Export given `points`, `curves`, and `tVectors` on the sphere $\mathbb S^2$
to Asymptote.

# Input
* `filename` – a file to store the Asymptote code in.

# Optional Arguments (Data)
* `colors` - dictionary of color arrays (indexed by symbols `:points`, `:curves`
  and `:tvector`) where each entry has to provide as least as many colors as
  the length of the corresponding sets.
* `curves` – an `Array` of `Arrays` of `SnPoints` where each inner array is
  interpreted as a curve and is accompanied by an entry within `colors`
* `points` – an `Array` of `Arrays` of `SnPoints` where each inner array is
  itnerpreted as a set of points and is accompanied by an entry within `colors`
* `tVectors` – an `Array` of `Arrays` of extended `SnTVector`s
  (`TVectorE{SnTVector}`) where each set of vectors is accompanied by an entry
  from within `colors`

# Optional Arguments (Asymptote)
* `arrowHeadSize` - (`6.0`) size of the arrowheads of the tangent vectors
* `cameraPosition` - (`(1., 1., 0.)`) position of the camera in the Asymptote
  szene
* `lineWidth` – (`1.0`) size of the lines used to draw the curves.
* `lineWidths` – overrides the previous value to specify a value per curve.
* `dotSize` – (`1.0`) size of the dots used to draw the points.
* `dotSizes` – overrides the previous value to specify a value per point set.
* `target` - (`(0.,0.,0.)`) position the camera points at.
"""
function asyExportS2(filename::String;
    points::Array{Array{SnPoint,1},1} = Array{Array{SnPoint,1},1}(undef,0),
    curves::Array{Array{SnPoint,1},1} = Array{Array{SnPoint,1},1}(undef,0),
    tVectors::Array{Array{TVectorE{SnTVector,SnPoint},1},1} = Array{Array{TVectorE{SnTVector,SnPoint},1},1}(undef,0),
    colors::Dict{Symbol, Array{RGBA{Float64},1} }  = Dict{Symbol,Array{RGBA{Float64},1}}(),
    arrowHeadSize::Float64 = 6.,
    cameraPosition::Tuple{Float64,Float64,Float64} = (1., 1., 0.),
    lineWidth::Float64 = 1.0,
    lineWidths::Array{Float64,1} = fill(lineWidth,size(curves)),
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
                    (key==:curves) ? "+linewidth($(lineWidths[i])pt)" : "",
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
            for point in curve
                j=j+1
                pString = string("(",
                  string([string(v,",") for v in getValue(point)]...)[1:end-1],
                    ")");
                if j>1
                    write( io," .. $(pString)")
                else
                    write( io, pString )
                end
            end
            write( io,string(";\n draw(p$(i), curveStyle$(i));\n") );
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
