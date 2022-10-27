# Set aggregate steady state variabel values
ASS       = m_par.ASHIFT
ZSS       = 1.0
ZISS      = 1.0
μSS       = m_par.μ
μwSS      = m_par.μw
τprogSS   = m_par.τ_prog
τlevSS    = m_par.τ_lev

σSS          = 1.0
τprog_obsSS  = 1.0
GshockSS     = 1.0
RshockSS     = 1.0
TprogshockSS = 1.0
Gshock2SS  = 1.0

SshockSS  = 1.0
RBSS      = m_par.RB
LPSS      = (rSS / RBSS)^4
LPXASS    = (rSS / RBSS)^4
ISS       = m_par.δ_0 * KSS

πSS       = 1.0
πwSS      = 1.0
rRBSS = (RBSS./πSS)^4

BgovSS = BSS
BtargetSS = BSS
BDSS      = -sum(distr_mSS.*(n_par.grid_m.<0).*n_par.grid_m)
# Calculate taxes and government expenditures
TSS       = dot(distr_ySS, taxrev) + av_tax_rateSS*((1.0 .- 1.0 ./ m_par.μw).*wSS.*NSS)
GSS       = TSS - (m_par.RB-1.0)*BSS

CSS       = (YSS - m_par.δ_0 * KSS - GSS - m_par.Rbar*BDSS + (ASS .- 1.0) .* RBSS .* BSS) 

qBSS = 1.0 ./ (RBSS .- 1.0 + m_par.δ_B)
bankprofitsSS = 1.0
qBlagSS  = qBSS

qSS       = 1.0
mcSS      = 1.0 ./ m_par.μ
mcwSS     = 1.0 ./ m_par.μw
mcwwSS    = wSS * mcwSS
uSS       = 1.0
profitsSS = (1.0 - mcSS).*YSS
unionprofitsSS = (1.0 - mcwSS) .* wSS .* NSS
tauLSSS = 1.0

BYSS   = BSS / YSS
TYSS   = TSS / YSS
TlagSS = TSS

YlagSS = YSS
BgovlagSS = BgovSS
GlagSS = GSS
IlagSS = ISS
wlagSS = wSS
qlagSS = qSS
ClagSS = CSS
av_tax_ratelagSS = av_tax_rateSS
τproglagSS       = τprogSS
GlagSS = GSS
Glag2SS = GSS

YgrowthSS = 1.0
BgovgrowthSS = 1.0
IgrowthSS = 1.0
wgrowthSS = 1.0
CgrowthSS = 1.0
TgrowthSS = 1.0
HtSS      = 1.0

KGSS =1.0
