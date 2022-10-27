#------------------------------------------------------------------------------
# Header: load module
#------------------------------------------------------------------------------
# ATTENTION: make sure that your present working directory pwd() is set to the folder
# containing script.jl and BASEforHANK.jl. Otherwise adjust the load path.

# pre-process user inputs for model setup
include("3_NumericalBasics/PreprocessInputs.jl")
using BenchmarkTools, LinearAlgebra

push!(LOAD_PATH, pwd())
using BASEforHANK   

# set BLAS threads to the number of Julia threads.
# prevents BLAS from grabbing all threads on a machine
BLAS.set_num_threads(Threads.nthreads())

#------------------------------------------------------------------------------
# initialize parameters to priors to select coefficients of DCTs of Vm, Vk]
# that are retained 
#------------------------------------------------------------------------------
m_par = ModelParameters()
priors = collect(metaflatten(m_par, prior)) # model parameters
par_prior = mode.(priors)
m_par = BASEforHANK.Flatten.reconstruct(m_par, par_prior)
e_set = BASEforHANK.e_set;
# alternatively, load estimated parameters by running, e.g.,
# @load BASEforHANK.e_set.save_posterior_file par_final e_set
# m_par = BASEforHANK.Flatten.reconstruct(m_par, par_final[1:length(par_final)-length(e_set.meas_error_input)])

################################################################################
# Comment in the following block to be able to go straight to plotting (comment out lines 40-80)
################################################################################
# @load "7_Saves/steadystate.jld2" sr_full
# @load "7_Saves/linearresults.jld2" lr_full
# @load "7_Saves/reduction.jld2" sr_reduc lr_reduc
# @load BASEforHANK.e_set.save_posterior_file sr_mc lr_mc er_mc m_par_mc draws_raw
# @set! e_set.estimate_model = false 

# Calculate Steady State at prior mode to find further compressed representation of Vm, Vk
sr_full = compute_steadystate(m_par)
jldsave("7_Saves/steadystate.jld2", true; sr_full) # true enables compression
# @load "7_Saves/steadystate.jld2" sr_full

#------------------------------------------------------------------------------
# compute and display steady-state moments
#------------------------------------------------------------------------------
K       = exp.(sr_full.XSS[sr_full.indexes.KSS])
B       = exp.(sr_full.XSS[sr_full.indexes.BSS])
Bgov    = exp.(sr_full.XSS[sr_full.indexes.BgovSS])
Y       = exp.(sr_full.XSS[sr_full.indexes.YSS])
T10W    = exp(sr_full.XSS[sr_full.indexes.TOP10WshareSS])
G       = exp.(sr_full.XSS[sr_full.indexes.GSS])
distr_m = sum(sr_full.distrSS,dims=(2,3))[:]
fr_borr = sum(distr_m[sr_full.n_par.grid_m.<0])

println("Steady State Moments:")
println("Liquid to Illiquid Assets Ratio:", B/K)
println("Capital to Output Ratio:", K/Y)
println("Government Debt to Output Ratio:", Bgov/Y)
println("Government spending to Output Ratio:", G/Y)
println("TOP 10 Wealth Share:", T10W)
println("Fraction of Borrower:", fr_borr)

# linearize the full model
lr_full = linearize_full_model(sr_full, m_par)
jldsave("7_Saves/linearresults.jld2", true; lr_full)
# @load "7_Saves/linearresults.jld2" lr_full

# Find sparse state-space representation
sr_reduc = model_reduction(sr_full, lr_full, m_par);
lr_reduc = update_model(sr_reduc, lr_full, m_par)
jldsave("7_Saves/reduction.jld2", true; sr_reduc, lr_reduc)
# @load "7_Saves/reduction.jld2" sr_reduc lr_reduc

# model timing
println("One model solution takes")
@set! sr_reduc.n_par.verbose = false
@btime lr_reduc = update_model(sr_reduc, lr_full, m_par)
@set! sr_reduc.n_par.verbose = true;


if e_set.estimate_model == true

        # warning: estimation might take a long time!
        er_mode, posterior_mode, smoother_mode, sr_mode, lr_mode, m_par_mode =
                find_mode(sr_reduc, lr_reduc, m_par)

        # Stores results in file e_set.save_mode_file 
        jldsave(BASEforHANK.e_set.save_mode_file, true;
                posterior_mode, smoother_mode, sr_mode, lr_mode, er_mode, m_par_mode, e_set)
        # @load BASEforHANK.e_set.save_mode_file posterior_mode sr_mode lr_mode er_mode m_par_mode smoother_mode e_set

        if e_set.estimation_type == :likelihoodbased
            sr_mc, lr_mc, er_mc, m_par_mc, draws_raw, posterior, accept_rate,
            par_final, hessian_sym, smoother_output = montecarlo(sr_mode, lr_mode, er_mode, m_par_mode)
        else
            sr_mc, lr_mc, er_mc, m_par_mc, draws_raw, posterior, accept_rate,
            par_final, hessian_sym = montecarlo(sr_mode, lr_mode, er_mode, m_par_mode)
        end
        
        # # Stores results in file e_set.save_posterior_file 
        jldsave(BASEforHANK.e_set.save_posterior_file, true;
                sr_mc, lr_mc, er_mc, m_par_mc, draws_raw, posterior, accept_rate,
                par_final, hessian_sym, e_set)
        # !! The following file is not provided !!
        #  @load BASEforHANK.e_set.save_posterior_file sr_mc lr_mc er_mc  m_par_mc draws_raw posterior accept_rate par_final hessian_sym e_set

end

using Flatten, Distributions, DataFrames, CSV, Plots

x0      = zeros(size(lr_mc.LOMstate,1), 1)
x0[sr_mc.indexes_r.Gshock] =1.0*m_par_mc.σ_Gshock

MX = [I; lr_mc.State2Control]
nlag=15#e_set.irf_horizon
x = x0*ones(1,nlag+2)
IRF_state_sparse_mode = zeros(sr_mc.indexes_r.profits, nlag+1)
for t = 1:nlag+1
        IRF_state_sparse_mode[:, t]= (MX*x[:,t])'
        x[:,t+1] = lr_mc.LOMstate * x[:,t]
end

println(     IRF_state_sparse_mode[sr_mc.indexes_r.RB,end])


# observed_vars_input::Array{Symbol,1} = [:G, :Y, :B, :I, :C, :RB, :LP]
Data_temp = DataFrame(CSV.File("irf_data_0706_inclPC.csv"; missingstring = "NaN"))
data_names_temp = propertynames(Data_temp)

# Rename observables that do not have matching model names
for i in data_names_temp
    name_temp = get(e_set.data_rename, i, :none)
    if name_temp != :none
        rename!(Data_temp, Dict(i => name_temp))
    end
end

styles=[:solid :dashdot :dot :dashdotdot :dot :dash]
colorlist=[:black, :blue, :red, :green, :orange]

select_variables=e_set.observed_vars_input


IRFs = HPDI_IRF([:Gshock], select_variables, nlag+1, sr_mc, lr_mc, m_par_mc, draws_raw;n_replic = 300)[1]
IRF_lower  = dropdims(quantile.(IRFs, 0.05), dims=3)
IRF_upper  = dropdims(quantile.(IRFs, 0.95), dims=3)
count = 1
for j in select_variables
        h = collect(range(1.0, stop=nlag, length = nlag))
        d = Data_temp[(Data_temp[:, :pointdum] .== 1), j]./maximum(Data_temp[:,:B])
        l = Data_temp[(Data_temp[:, :pointdum] .== 1), j]./maximum(Data_temp[:,:B]) .- 1.6449*Data_temp[(Data_temp[:, :pointdum] .== 0), j]./maximum(Data_temp[:,:B])
        u = Data_temp[(Data_temp[:, :pointdum] .== 1), j]./maximum(Data_temp[:,:B]) .+ 1.6449*Data_temp[(Data_temp[:, :pointdum] .== 0), j]./maximum(Data_temp[:,:B])
        if j==:B
            d_model = IRF_state_sparse_mode[getfield(sr_mc.indexes_r,j), 2:end]
            l_model = IRF_lower[count,2:end]
            u_model = IRF_upper[count,2:end]
        else
            d_model = IRF_state_sparse_mode[getfield(sr_mc.indexes_r,j), 1:end-1]
            l_model = IRF_lower[count,1:end-1]
            u_model = IRF_upper[count,1:end-1]
        end
        if j==:G
                l[1]=d[1] 
                u[1]=d[1] 
        end
    
        plt1 = plot(l, fillrange = u, alpha = 0.2, label="Data", reuse=false, titlefontsize = 16,color=colorlist[1],
                    legend = :topright,  size = (250,250), legendfontsize = 10, xtickfontsize = 10, ytickfontsize = 10, tickfontvalign=:top, linewidth=0)
        plt1 = plot!(d, label="Data", reuse=false, titlefontsize = 16,color=colorlist[1],linewidth=2, linestyle=styles[6])

        plt1 = plot!(l_model, fillrange = u_model, alpha = 0.2, label="Model", reuse=false, titlefontsize = 16,color=colorlist[2],
                legend = :topright,  size = (250,250), legendfontsize = 10, xtickfontsize = 10, ytickfontsize = 10, tickfontvalign=:top, linewidth=0)
        plt1 = plot!(d_model, label="Model", reuse=false, legend=false,linestyle = styles[1],color = colorlist[2], linewidth=2, fontfamily="Computer Modern")

        display(plt1)
        savefig(string("./8_PostEstimation/Figures/IRFs_on_Gshock_",j,"_BRB_bands.pdf"))
        global count += 1
end