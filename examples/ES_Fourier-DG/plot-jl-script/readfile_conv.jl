using DelimitedFiles
using Plots

#open("./examples/ES_Fourier-DG/array1.txt","r") do io
convarr = readdlm("./examples/ES_Fourier-DG/conv/trans_vortex_y/array_conv.txt",'\t',Float64,'\n')
convarr = reshape(convarr,4,1,5)

N_P_arr = [1;2;3;4;5]
Np_F_arr = [8]
K1D_arr = [2;4;10;20]
