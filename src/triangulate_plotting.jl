"""
    MeshPlotter(VX,VY,EToV)
    MeshPlotter(triout::TriangulateIO)

Plot recipe to plot a mesh. Usage: plot(MeshPlotter(triout))
"""
struct MeshPlotter{Tv,Ti}
    VX::Vector{Tv}
    VY::Vector{Tv}
    EToV::Matrix{Ti}
end
MeshPlotter(triout::TriangulateIO) = MeshPlotter(triangulateIO_to_VXYEToV(triout)...)

"""
    BoundaryTagPlotter(triout::TriangulateIO)    

Plot recipe to visualize boundary tags by color. Usage: plot(BoundaryTagPlotter(triout))
"""
struct BoundaryTagPlotter
    triout::TriangulateIO
end

@recipe function f(m::MeshPlotter)
    @unpack VX,VY,EToV = m

    linecolor --> :black
    legend --> false
    aspect_ratio --> 1
    title --> "$(size(EToV,1)) elements"

    xmesh = Float64[]
    ymesh = Float64[]
    for vertex_ids in eachrow(EToV)
        ids = vcat(vertex_ids, vertex_ids[1])
        append!(xmesh,[VX[ids];NaN])
        append!(ymesh,[VY[ids];NaN])
    end
    return xmesh,ymesh
end

@recipe function f(m::BoundaryTagPlotter)
    triout = m.triout
    tags = unique(triout.segmentmarkerlist)
    num_colors = length(tags)
    colors = distinguishable_colors(num_colors)
    xseg = zeros(2,size(triout.segmentlist,2))
    yseg = zeros(2,size(triout.segmentlist,2))
    segcolor = HSV{Float32}[]
    for (col,segment) in enumerate(eachcol(triout.segmentlist))
        xseg[:,col] .= triout.pointlist[1,segment]
        yseg[:,col] .= triout.pointlist[2,segment]
        push!(segcolor,colors[triout.segmentmarkerlist[col]])
    end
    for i = 1:num_colors
        color_ids = findall(triout.segmentmarkerlist .== tags[i])

        # hack to get around issues with multiple legend labels appearing when plotting multiple series
        x_i = vec([xseg[:,color_ids]; fill(NaN,length(color_ids))']) 
        y_i = vec([yseg[:,color_ids]; fill(NaN,length(color_ids))']) 

        @series begin
            marker --> :circle
            seriescolor --> permutedims(segcolor[color_ids]),
            ratio --> 1
            label --> string(tags[i])
            x_i,y_i
        end
    end
    # we already added all shapes in @series so we don't want to return a series
    # here (we are returning an empty series which is not added to the legend.)
    primary := false
    ()
end
