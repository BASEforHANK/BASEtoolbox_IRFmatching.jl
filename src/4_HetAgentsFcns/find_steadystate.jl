@doc raw"""
    find_steadystate(m_par)

Find the stationary equilibrium capital stock.

# Returns
- `KSS`: steady-state capital stock
- `VmSS`, `VkSS`: marginal value functions
- `distrSS::Array{Float64,3}`: steady-state distribution of idiosyncratic states, computed by [`Ksupply()`](@ref)
- `n_par::NumericalParameters`,`m_par::ModelParameters`
"""
function find_steadystate(m_par)

# -------------------------------------------------------------------------------
## STEP 1: Find the stationary equilibrium for coarse grid
# -------------------------------------------------------------------------------
#-------------------------------------------------------
# Income Process and Income Grids
#-------------------------------------------------------
# Read out numerical parameters for starting guess solution with reduced income grid.

# Numerical parameters 
n_par                   = NumericalParameters(m_par = m_par, naggrstates = length(state_names), naggrcontrols = length(control_names),
                                              aggr_names  = aggr_names, distr_names = distr_names)

# Capital stock guesses

# Capital stock guesses
rmin   = 0.00001
rmax   = (1.0 .- m_par.β)./m_par.β - 0.005

capital_intensity(r) = ((r + m_par.δ_0) ./ m_par.α .* m_par.μ)^(1.0 ./ (m_par.α .- 1))
labor_supply(w) = ((1.0 .- m_par.τ_prog) .* m_par.τ_lev)^(1.0 ./ (m_par.γ .+ m_par.τ_prog)) .*
                    w^((1.0 .- m_par.τ_prog) ./ (m_par.γ .+ m_par.τ_prog))

Kmax = capital_intensity(rmin) .* labor_supply(wage(capital_intensity(rmin), 1.0 ./ m_par.μ, 1.0, m_par) ./ m_par.μw)
Kmin = capital_intensity(rmax) .* labor_supply(wage(capital_intensity(rmax), 1.0 ./ m_par.μ, 1.0, m_par) ./ m_par.μw)

Kmax=51.5
Kmin=3.0
println("Kmin: ", Kmin)
println("Kmax: ", Kmax)

# a.) Define excess demand function
d(  K, 
    initial::Bool=true, 
    Vm_guess = zeros(1,1,1), 
    Vk_guess = zeros(1,1,1), 
    distr_guess = n_par.dist_guess) = Kdiff(K, n_par, m_par, initial, Vm_guess, Vk_guess, distr_guess)


    # KN_HANDLE(KN) =  fnc_kndiff(KN, n_par, m_par)[1]

# K_guess=35.0

# K, fval, iter, distF = broyden(d,K_guess,1e-7,1e-10,100,0.005,0.05)

# -----
# Find stationary equilibrium for refined economy
BrentOut                = CustomBrent(d, Kmin, Kmax;tol = n_par.ϵ)
KSS                     = BrentOut[1]
VmSS                    = BrentOut[3][2]
VkSS                    = BrentOut[3][3]
distrSS                 = BrentOut[3][4]
if n_par.verbose
    println("Capital stock is")
    println(KSS)
end
return KSS, VmSS, VkSS, distrSS, n_par, m_par

end

