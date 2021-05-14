using Plots
using Triangulate

# TODO: convert to recipes!
# ========= plotting routines for mesh visualization ============

"""
    plot_mesh(VX,VY,EToV)
    plot_mesh!(VX,VY,EToV)
    plot_mesh(triout::TriangulateIO)
    plot_mesh!(triout::TriangulateIO)
    
Plots a mesh given vertices + connectivity or a TriangulateIO object.
"""
function plot_mesh(VX,VY,EToV)
    Plots.plot()
    plot_mesh!(VX,VY,EToV)
end
function plot_mesh!(VX,VY,EToV)
    xmesh = Float64[]
    ymesh = Float64[]
    for vertex_ids in eachrow(EToV)
        ids = vcat(vertex_ids, vertex_ids[1])
        append!(xmesh,[VX[ids];NaN])
        append!(ymesh,[VY[ids];NaN])
    end
    display(Plots.plot!(xmesh,ymesh,linecolor=:black,legend=false,ratio=1,title="$(size(EToV,1)) elements"))
end

plot_mesh(triout::TriangulateIO) = plot_mesh(triangulateIO_to_VXYEToV(triout)...)
plot_mesh!(triout::TriangulateIO) = plot_mesh!(triangulateIO_to_VXYEToV(triout)...)

"""
    plot_segment_tags(triout::TriangulateIO)    
    plot_segment_tags!(triout::TriangulateIO)    

Plot boundary segments in colored according to the tag number. 
"""
function plot_segment_tags(triout::TriangulateIO)    
    Plots.plot()
    plot_segment_tags!(triout) 
end
function plot_segment_tags!(triout::TriangulateIO)    
    tags = unique(triout.segmentmarkerlist)
    num_colors = length(tags)
    if num_colors>1
        colors = range(HSV(0,1,1), stop=HSV(360-360÷num_colors,1,1), length=num_colors)
    else
        colors = [HSV(0,1,1)]
    end
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

        Plots.plot!(x_i,y_i,mark=:circle,color=permutedims(segcolor[color_ids]),
                    ratio = 1,label=string(tags[i])) 
    end
    display(plot!())
end