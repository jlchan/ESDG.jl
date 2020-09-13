using Plots
push!(LOAD_PATH, "./src")
using Basis2DTri
using Basis1D
theme(:wong)
gr(aspect_ratio=1,legend=:bottomright,axis=nothing,border=nothing,ticks=nothing,legendfontsize=12,xlims=[-3,7],
   markerstrokewidth=0,markersize=1)


trans_x = 1.2
trans_y = 1.6

plot([-2;-2].+2*4/8,[-2;2],seriescolor=:lightgrey,lw=3,label = nothing)
plot!([-2;-2+trans_x].+2*4/8,[2;2+trans_y],seriescolor=:lightgrey,lw=3,label = nothing)
plot!([-2+trans_x;-2+trans_x].+2*4/8,[2+trans_y;-2+trans_y],seriescolor=:lightgrey,lw=3,ls=:dash,label = nothing)
plot!([-2;-2+trans_x].+2*4/8,[-2;-2+trans_y],seriescolor=:lightgrey,lw=3,ls=:dash,label = nothing)

plot!([-2;-2].+3*4/8,[-2;2],seriescolor=:lightgrey,lw=3,label = nothing)
plot!([-2;-2+trans_x].+3*4/8,[2;2+trans_y],seriescolor=:lightgrey,lw=3,label = nothing)
plot!([-2+trans_x;-2+trans_x].+3*4/8,[2+trans_y;-2+trans_y],seriescolor=:lightgrey,lw=3,ls=:dash,label = nothing)
plot!([-2;-2+trans_x].+3*4/8,[-2;-2+trans_y],seriescolor=:lightgrey,lw=3,ls=:dash,label = nothing)


plot!([-2;-2].+5*4/8,[-2;2],seriescolor=:lightgrey,lw=3,label = nothing)
plot!([-2;-2+trans_x].+5*4/8,[2;2+trans_y],seriescolor=:lightgrey,lw=3,label = nothing)
plot!([-2+trans_x;-2+trans_x].+5*4/8,[2+trans_y;-2+trans_y],seriescolor=:lightgrey,lw=3,ls=:dash,label = nothing)
plot!([-2;-2+trans_x].+5*4/8,[-2;-2+trans_y],seriescolor=:lightgrey,lw=3,ls=:dash,label = nothing)

plot!([-2;-2].+6*4/8,[-2;2],seriescolor=:lightgrey,lw=3,label = nothing)
plot!([-2;-2+trans_x].+6*4/8,[2;2+trans_y],seriescolor=:lightgrey,lw=3,label = nothing)
plot!([-2+trans_x;-2+trans_x].+6*4/8,[2+trans_y;-2+trans_y],seriescolor=:lightgrey,lw=3,ls=:dash,label = nothing)
plot!([-2;-2+trans_x].+6*4/8,[-2;-2+trans_y],seriescolor=:lightgrey,lw=3,ls=:dash,label = nothing)

plot!([-2;-2].+7*4/8,[-2;2],seriescolor=:lightgrey,lw=3,label = nothing)
plot!([-2;-2+trans_x].+7*4/8,[2;2+trans_y],seriescolor=:lightgrey,lw=3,label = nothing)
plot!([-2+trans_x;-2+trans_x].+7*4/8,[2+trans_y;-2+trans_y],seriescolor=:lightgrey,lw=3,ls=:dash,label = nothing)
plot!([-2;-2+trans_x].+7*4/8,[-2;-2+trans_y],seriescolor=:lightgrey,lw=3,ls=:dash,label = nothing)


plot!([-2;-2],[-2;2],seriescolor=:dimgrey,lw=3,label = nothing)
plot!([-2;2],[2;2],seriescolor=:dimgrey,lw=3,label = nothing)
plot!([2;-2],[-2;-2],seriescolor=:dimgrey,lw=3,label = nothing)

plot!([-2;-2].+trans_x,[-2;2].+trans_y,seriescolor=:dimgrey,lw=3,ls=:dash,label = nothing)
plot!([-2;2].+trans_x,[2;2].+trans_y,seriescolor=:dimgrey,lw=3,label = nothing)
plot!([2;-2].+trans_x,[-2;-2].+trans_y,seriescolor=:dimgrey,lw=3,ls=:dash,label = nothing)

plot!([-2;-2+trans_x],[-2;-2+trans_y],seriescolor=:dimgrey,lw=3,ls=:dash,label = nothing)
plot!([-2;-2+trans_x],[2;2+trans_y],seriescolor=:dimgrey,lw=3,label = nothing)

# color1 = :royalblue
# color2 = :deepskyblue
# color3 = :lightskyblue1

color1 = :firebrick4
color2 = :firebrick2
color3 = :coral

# First Fourier slice
plot!([-2;-2].+4/8,[-2;2],seriescolor=color1,lw=3,label = nothing)
plot!([-2;-2+trans_x].+4/8,[2;2+trans_y],seriescolor=color1,lw=3,label = "Fourier mode 1")
plot!([-2+trans_x;-2+trans_x].+4/8,[2+trans_y;-2+trans_y],seriescolor=color1,lw=3,ls=:dash,label = nothing)
plot!([-2;-2+trans_x].+4/8,[-2;-2+trans_y],seriescolor=color1,lw=3,ls=:dash,label = nothing)

# Second Fourier slice
plot!([-2;-2].+4*4/8,[-2;2],seriescolor=color2,lw=3,label = nothing)
plot!([-2;-2+trans_x].+4*4/8,[2;2+trans_y],seriescolor=color2,lw=3,label = "Fourier mode 4")
plot!([-2+trans_x;-2+trans_x].+4*4/8,[2+trans_y;-2+trans_y],seriescolor=color2,lw=3,ls=:dash,label = nothing)
plot!([-2;-2+trans_x].+4*4/8,[-2;-2+trans_y],seriescolor=color2,lw=3,ls=:dash,label = nothing)

# Third Fourier slice
plot!([2;2+trans_x],[-2;-2+trans_y],seriescolor=color3,lw=3,label = nothing)
plot!([2;2+trans_x],[2;2+trans_y],seriescolor=color3,lw=3,label = nothing)
plot!([2;2],[2;-2],seriescolor=color3,lw=3,label = nothing)
plot!([2;2].+trans_x,[2;-2].+trans_y,seriescolor=color3,lw=3,label = "Fourier mode 8")

plot!([-1.5,1.5],[-2.2,-2.2],arrow=0.4,seriescolor=:black,lw=2.5,label=nothing)
annotate!(0,-2.5,"z direction")
savefig("domain.png")
