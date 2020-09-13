using Plots
gr(xlims=[-1,6],ylims=[-4,6],border=:none)

plot([0,1],-[4,4],fill=(0,:royalblue1),fillalpha=0.7)
plot!([1,2],-[4,4],fill=(0,:darkorange1),fillalpha=0.7)
plot!([4,5],[3,3],fill=(0,:darkorange1),fillalpha=0.7)
plot!([4,5],[2,2],fill=(0,:white),fillalpha=1)
plot!([-0.1,2.1],-[2,2],fill=(0,:white),lw=0.1,fillalpha=1)
plot!([0,5],[0,0],seriescolor=:black,lw=3,legend=nothing)

for i = 1:5
    plot!([0,5],[i,i],seriescolor=:black,lw=3,legend=nothing)
end
for i = 0:5
    plot!([i,i],[0,5],seriescolor=:black,lw=3,legend=nothing)
end

for i = 0:5
    plot!([i,i],[-2,-4],seriescolor=:black,lw=3,legend=nothing)
end
for i = 2:4
    plot!([0,5],[-i,-i],seriescolor=:black,lw=3,legend=nothing)
end


current()



savefig("GPU_3.png")
