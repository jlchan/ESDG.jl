rt = [4.348e-06
1.465e-05
5.446e-05
1.976e-04
7.916e-04
3.825e-03]

rt_trixi = [3.21e-05
3.15e-05
4.97e-05
1.21e-04
4.00e-04
1.51e-03]

plot(elems,rt_trixi,marker=:dot,label="Trixi")
plot!(elems,rt,marker=:dot,label="JC code")
plot!(elems,1e-7*elems,linestyle=:dash,label="Linear")
plot!(xaxis=:log,yaxis=:log,xlabel="Elements",ylabel="Runtime (seconds)",legend=:bottomright)
