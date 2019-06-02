using ColorTypes, Colors, ColorSchemes
import LinearAlgebra: eigen, eigvals, tril
export renderAsymptote, asyExportS2Signals, asyExportS2Data, asyExportSPDData
"""
    renderAsymptote(filename, exportFct=missing; render=4, format="png", ...)
render an exported `asy`. The export function can be provided but might also
be infered by the optional arguments, i.e. type of `data`
(or `curves`, `points` and `tVectors`) provided.

# Input

* `filename` – filename of the exported `asy` and rendered image
* `exportFct`: (`missing`) a function creating an `asy` file with `kwargs` as optional
  arguments and the `filename` string as its only mandatory argument. If this
  function is not given, we infere it from kwargs, i.e. `data` indicates that
  `asyExportXData` is used. `points`, `curves` or `tVectors` indicate, that
  `asyExportS2Signals` is infered.

# Keyword Arguments

the default values are given in brackets

* `render` – (`4`) render level of asymptote, i.e. its `-render` option
* `format` – (`"png"`) final rendered format, i.e. asymptote's `-f` option

all further keyword arguments are passed down to the `exportFct` call.

# See also
[`asyExportS2Signals`](@ref)
"""
function renderAsymptote(filename, exportFunction::Union{Function,Missing}=missing; render::Int=4, format="png",
    exportFolder = string( filename[1:( [findlast(".",filename)...][1])], format), kwargs...)
    if ismissing(exportFunction) # try to determine type automatically
        kwargs_dict = Dict(kwargs)
        if haskey(kwargs_dict,:data) # asyExport$MData? check whether PowPoint Data is a SPD point
            if isa( first(getValue(kwargs_dict[:data])), SPDPoint)
                exportFunction = asyExportSPDData
            elseif isa( first(getValue(kwargs_dict[:data])), SnPoint)
                exportFunction = asyExportS2Data
            end
        elseif haskey(kwargs_dict,:points) || haskey(kwargs_dict,:curves) || haskey(kwargs_dict, :tVectors) # asyExportS2Signals
            exportFunction = asyExportS2Signals
        else
            throw(ErrorException("Could not find a suitable export function (automatically), please provide an exportFunction."))
        end
    end # end if ismissing
    renderCmd = `asy -render $(render) -f $(format) -globalwrite  -o "$(relpath(exportFolder))" $(filename)`
    exportFunction(filename; kwargs...)
    run(renderCmd)
end
@doc doc"""
    asyExportS2Signals(filename; points, curves, tVectors, colors, options...)
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
* `lineWidths` – overrides the previous value to specify a value per curve and tVector set.
* `dotSize` – (`1.0`) size of the dots used to draw the points.
* `dotSizes` – overrides the previous value to specify a value per point set.
* `sphereColor` – (`RGBA{Float64}(0.85, 0.85, 0.85, 0.6)`) color of the sphere
  the data is drawn on
* `sphereLineColor` –  (`RGBA{Float64}(0.75, 0.75, 0.75, 0.6)`) color of the lines on the sphere
* `sphereLineWidth` – (`0.5`) line width of the lines on the sphere
* `target` – (`(0.,0.,0.)`) position the camera points at
"""
function asyExportS2Signals(filename::String;
    points::Array{Array{SnPoint{T},1},1} where T = Array{Array{SnPoint{Float64},1},1}(undef,0),
    curves::Array{Array{SnPoint{T},1},1} where T = Array{Array{SnPoint{Float64},1},1}(undef,0),
    tVectors::Array{Array{TVectorE{SnTVector{T},SnPoint{T}},1},1} where T= Array{Array{TVectorE{SnTVector{Float64},SnPoint{Float64}},1},1}(undef,0),
    colors::Dict{Symbol, Array{RGBA{Float64},1} }  = Dict{Symbol,Array{RGBA{Float64},1}}(),
    arrowHeadSize::Float64 = 6.,
    cameraPosition::Tuple{Float64,Float64,Float64} = (1., 1., 0.),
    lineWidth::Float64 = 1.0,
    lineWidths::Array{Float64,1} = fill(lineWidth,size(curves)+size(tVectors)),
    dotSize::Float64 = 1.0,
    dotSizes::Array{Float64,1} = fill(dotSize,size(points)),
    sphereColor::RGBA{Float64} = RGBA{Float64}(0.85, 0.85, 0.85, 0.6),
    sphereLineColor::RGBA{Float64} = RGBA{Float64}(0.75, 0.75, 0.75, 0.6),
    sphereLineWidth::Float64 = 0.5,
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
            "pen SpherePen = rgb($(red(sphereColor)),",
            "$(green(sphereColor)),$(blue(sphereColor)))",
            "+opacity($(alpha(sphereColor)));\n",
            "pen SphereLinePen = rgb($(red(sphereLineColor)),",
            "$(green(sphereLineColor)),$(blue(sphereLineColor)))",
            "+opacity($(alpha(sphereLineColor)))+linewidth($(sphereLineWidth)pt);\n",
            "draw(surface(S), surfacepen=SpherePen, meshpen=SphereLinePen);\n")
        );
        write(io,"\n/*\n  Colors\n*/\n")
        j=0
        for (key,value) in colors # colors for all keys
            penPrefix = "$(j)"
            sets = 0
            if key==:points
                penPrefix="point"
                sets = length(points)
            elseif key==:curves
                penPrefix="curve"
                sets = length(curves)
            elseif key==:tvectors
                penPrefix="tVector"
                sets = length(tVectors)
            end
            if length(value) < sets
                throw( ErrorException(
                    "Not enough colors ($(length(value))) provided for $(sets) sets in $(key)."
                ))
            end
            i=0
            # export all colors
            for c in value
                i=i+1;
                if i>sets
                    # avoid access errors in lineWidth or dotSizes if more colors then sets are given
                    break
                end
                write(io,string("pen $(penPrefix)Style$(i) = ",
                    "rgb($(red(c)),$(green(c)),$(blue(c)))",
                    (key==:curves || key==:tvectors) ? "+linewidth($(lineWidths[i])pt)" : "",
                    (key==:tvectors) ? "+linewidth($(lineWidths[length(curves)+i])pt)" : "",
                    (key==:points) ? "+linewidth($(dotSizes[i])pt)" : "",
                    "+opacity($(alpha(c)));\n"));
            end
        end
        if length(points)>0
            write(io,"\n/*\n  Exported Points\n*/\n")
        end
        i=0
        for pSet in points
            i=i+1
            for point in pSet
                write(io,string("dot( (",string([string(v,",") for v in getValue(point)]...)[1:end-1],"), pointStyle$(i));\n"));
            end
        end
        i=0
        if length(curves)>0
            write(io,"\n/*\n  Exported Curves\n*/\n")
        end
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
        if length(tVectors)>0
            write(io,"\n/*\n  Exported tangent vectors\n*/\n")
        end
        for tVecs in tVectors
            i=i+1
            j=0
            for vector in tVecs
                j=j+1
                base = getValue(getBasePoint(vector))
                endPoints = base + getValue(vector)
                write(io,string("draw( (",
                    string( [string(v,",") for v in base]...)[1:end-1],")--(",
                    string( [string(v,",") for v in endPoints]...)[1:end-1],
                    "), tVectorStyle$(i),Arrow3($(arrowHeadSize)));\n"));
            end
        end
    finally
        close(io)
    end
end
@doc doc"""
    asyExportS2Data(filename)
Export given `data` as a point on a `Power{SnPoint}` manifold, i.e. one-, two-
or three-dimensional data with points on the [`Sphere`](@ref)`(2)` $\mathbb S^2$.

# Input
* `filename` – a file to store the Asymptote code in.

# Optional Arguments (Data)
* `data` – a `PowPoint` representing the 1-,2-, or 3-D array of `SnPoints`
* `elevationColorScheme` - A `ColorScheme` for elevation
* `scaleAxes` - (`(1/3,1/3,1/3)`) move spheres closer to each other by a factor
  per direction

# Optional Arguments (Asymptote)
* `arrowHeadSize` - (`1.8`) size of the arrowheads of the vectors (in mm)
* `cameraPosition` - position of the camrea (default: centered above xy-plane)
  szene
* `target` - position the camera points at (default: center of xy-plane within
  data).
"""
function asyExportS2Data(filename::String;
    data::PowPoint = PowPoint(fill(SnPoint([0.,0.,1.]),0,0)),
    arrowHeadSize::Float64 = 1.8,
    scaleAxes = (1/3.,1/3.,1/3.),
    cameraPosition::Tuple{Float64,Float64,Float64} = scaleAxes.*( (size(data,1)-1)/2 ,(size(data,2)-1)/2, max(size(data,3),0)+10),
    target::Tuple{Float64,Float64,Float64} = scaleAxes.*( (size(data,1)-1)/2 ,(size(data,2)-1)/2, 0.),
    elevationColorScheme = ColorSchemes.viridis,
    )
    io = open(filename,"w")
    try
      write(io,string("import settings;\nimport three;\n",
        "size(7cm);\n",
        "DefaultHead.size=new real(pen p=currentpen) {return $(arrowHeadSize)mm;};\n",
        "currentprojection=perspective( ",
        "camera = $(cameraPosition), up=Y,",
        "target = $(target) );\n\n"));
      dims = [size(data,i) for i=[1,2,3]]
      for x=1:dims[1]
        for y=1:dims[2]
          for z=1:dims[3]
            v = Tuple(getValue(data[x,y,z])) #extract value
            el = asin( min(1,max(-1,v[3])) ); # since 3 is between -1 and 1 this yields a value between 0 and pi
            # map elevation to colormap
            c = get(elevationColorScheme,el+π/2, (0.,Float64(π)) );
            # write arrow in this colormap
            # transpose image to comply with image adresses (first index column downwards, second rows)
            write(io,string("draw( $(scaleAxes.*(x-1,y-1,z-1))",
              "--$(scaleAxes.*(x-1,y-1,z-1).+v),",
              " rgb($(red(c)),$(green(c)),$(blue(c))), Arrow3);\n"));
          end
        end
      end
    finally
      close(io)
    end
end
@doc doc"""
    asyExportSPDData(filename)

export given `data` as a point on a `Power{SPDPoint}` manifold, i.e. one-, two-
or three-dimensional data with points on the manifold of symmetric positive
definite matrices.

# Input
* `filename` – a file to store the Asymptote code in.

# Optional Arguments (Data)
* `data` – a `PowPoint` representing the 1-,2-, or 3-D array of `SPDPoints`
* `colorScheme` - A `ColorScheme` for Geometric Anisotropy Index
* `scaleAxes` - (`(1/3,1/3,1/3)`) move symmetric positive definite matrices
  closer to each other by a factor per direction compared to the distance
  esimated by the maximal eigenvalue of all involved SPD points

# Optional Arguments (Asymptote)
* `cameraPosition` - position of the camrea (default: centered above xy-plane)
  szene.
* `target` - position the camera points at (default: center of xy-plane within data).

Both values `cameraPosition` and `target` are scaled by `scaledAxes*EW`, where
`EW` is the maximal eigenvalue in the `data`.
"""
function asyExportSPDData(filename::String;
    data::PowPoint = PowPoint(fill(SPDPoint(Matrix{Float64}(I,3,3)),0,0)),
    scaleAxes = (1/3.,1/3.,1/3.) #multiplied with the maximal eigenvalue if data is present
      .* ( length(getValue(data)) > 0 ? maximum(maximum(eigvals.( getValue.(getValue(data)) ))) : 1 ),
    cameraPosition::Tuple{Float64,Float64,Float64} = ( (size(data,1)-1)/2 ,(size(data,2)-1)/2, max(size(data,3),0.)+10.),
    target::Tuple{Float64,Float64,Float64} = ( (size(data,1)-1)/2 ,(size(data,2)-1)/2, 0.),
    colorScheme = ColorSchemes.viridis,
    )
    io = open(filename,"w")
    try
      write(io,string("import settings;\nimport three;\n",
      "surface ellipsoid(triple v1,triple v2,triple v3,real l1,real l2, real l3, triple pos=O) {\n",
      "  transform3 T = identity(4);\n",
      "  T[0][0] = l1*v1.x;\n  T[1][0] = l1*v1.y;\n  T[2][0] = l1*v1.z;\n",
      "  T[0][1] = l2*v2.x;\n  T[1][1] = l2*v2.y;\n  T[2][1] = l2*v2.z;\n",
      "  T[0][2] = l3*v3.x;\n  T[1][2] = l3*v3.y;\n  T[2][2] = l3*v3.z;\n",
      "  T[0][3] = pos.x;\n  T[1][3] = pos.y;\n  T[2][3] = pos.z;\n",
      "  return T*unitsphere;\n}\n\n",
      "size(200);\n\n",
      "real gDx=$(scaleAxes[1]);\n",
      "real gDy=$(scaleAxes[2]);\n",
      "real gDz=$(scaleAxes[3]);\n\n",
      "currentprojection=perspective(up=Y, ",
      "camera = (gDx*$(cameraPosition[1]),gDy*$(cameraPosition[2]),gDz*$(cameraPosition[3])), ",
      "target = (gDx*$(target[1]),gDy*$(target[2]),gDz*$(target[3])) );\n",
      "currentlight=Viewport;\n\n",
      ));
      dims = [size(data,1) size(data,2) size(data,3) ];
      for x=1:dims[1]
        for y=1:dims[2]
          for z=1:dims[3]
            A = getValue(data[x,y,z]) #extract matrix
            F = eigen(A)
            if maximum(abs.(A)) > 0. # a nonzero matrix (exclude several pixel
              # Following Moakher & Batchelor: Geometric Anisotropic Index:
              λ = F.values
              V = F.vectors
              Lλ = log.(λ)
              GAI = sqrt( 2/3*sum( Lλ.^2 )  - 2/3 * sum(sum( tril( Lλ * Lλ',-1), dims=1), dims=2)[1])
              c = get(colorScheme, GAI/(1+GAI), (0,1) )
              write(io,string("  draw(  ellipsoid( ($(V[1,1]),$(V[2,1]),$(V[3,1])),",
              " ($(V[1,2]),$(V[2,2]),$(V[3,2])), ($(V[1,3]),$(V[2,3]),$(V[3,3])),",
              " $(λ[1]), $(λ[2]), $(λ[3]), ",
              " (gDx*$(x-1), gDy*$(y-1), gDz*$(z-1))),",
              " rgb($(red(c)),$(green(c)),$(blue(c)))  );\n"));
            end
          end
        end
      end
    finally
      close(io)
    end
end
