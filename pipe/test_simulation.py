import sys
import os
import numpy as np

sys.path.append(os.getcwd())
import pressure_drop as pdp

# 입력 변수 고정 (웹 UI에서 넘겨받는다고 가정)
fluid_key = "Water"
temp_c = 20.0
Q_m3s = 0.005 # 300 L/min
L = 100.0 # 100m 직선관
epsilon = pdp.ROUGHNESS["Stainless Steel (스테인리스 강관)"]
sum_K_fit = 1.5
sum_K_valve = 2.0

rho, mu = pdp.get_fluid_properties(fluid_key, temp_c)

# KS 규격 기반 경제 분석 세팅
eco_elec = 150.0  # 원/kWh
eco_ir = 2.50     # 금리 2.5%
eco_years = 20    # 20년 상환
eco_inf_e = 3.0   # 인플레이션 3%
eco_carbon_price = 15000.0
eco_maint_rate = 2.0
eco_hours = 8000
eco_carbon_factor = 0.4594

eco_pump_base = 1000000
eco_pump_kw = 300000
eco_material_k = 500000
eco_fitting_pct = 35.0
eco_labor_pct = 150.0
eco_sch = "SCH 40"

# 지상 노출 (토목 굴착 페널티 +0.0)
install_factor = (eco_labor_pct / 100.0) 

print("==================================================")
print(f"배관 길이: {L}m, 유량: {Q_m3s * 60000:.0f} L/min")
print("==================================================")

ir = eco_ir / 100.0
inf_e = eco_inf_e / 100.0
N = eco_years

crf = (ir * (1 + ir)**N) / ((1 + ir)**N - 1)
pvf_std = (1.0 - (1.0 + ir)**-N) / ir
x = (1.0 + inf_e) / (1.0 + ir)
pvf_energy = (1.0 / (1.0 + inf_e)) * x * (1.0 - x**N) / (1.0 - x)
lvl_energy_factor = pvf_energy * crf
lvl_maint_factor = pvf_std * crf

ks_candidates = []
for nps, sch_data in pdp.KS_PIPE_DATA.items():
    if eco_sch in sch_data:
        t = sch_data[eco_sch]
        od = sch_data["OD"]
        id_m = (od - 2*t) / 1000.0
        if id_m > 0:
            ks_candidates.append({"Name": nps, "D_m": id_m})

min_cost = float('inf')
opt_name = ""

for cand in ks_candidates:
    t_name = cand["Name"]
    test_d = cand["D_m"]
    
    t_v = pdp.calc_velocity(Q_m3s, test_d)
    t_Re = pdp.calc_reynolds(rho, t_v, test_d, mu)
    t_f, _ = pdp.calc_friction_factor(t_Re, test_d, epsilon)
    t_dp_fric, t_dp_fit, t_dp_valve, t_dp_total = pdp.calc_pressure_dp(t_f, L, test_d, rho, t_v, sum_K_fit, sum_K_valve)
    
    H_m = t_dp_total / (rho * 9.81)
    q_m3h = Q_m3s * 3600.0
    base_eff = 45.0 + (15.0 * np.log10(max(q_m3h, 1.0)))
    base_eff = min(base_eff, 85.0)
    head_penalty = max(0.0, (H_m - 30.0) * 0.15)
    curr_eff = max(base_eff - head_penalty, 30.0)
    
    t_kw = pdp.calc_pump_power(t_dp_total, Q_m3s, curr_eff)
    pump_capex = eco_pump_base + (t_kw * eco_pump_kw)
    
    t_capex_material = eco_material_k * test_d * L
    t_capex_fittings = t_capex_material * (eco_fitting_pct / 100.0)
    t_capex_base = t_capex_material + t_capex_fittings
    
    pipe_capex = t_capex_base * (1.0 + install_factor)
    
    capex = pipe_capex + pump_capex
    euac_capex = capex * crf
    
    e1 = t_kw * eco_hours * eco_elec
    euac_energy = e1 * lvl_energy_factor
    
    carbon1 = (t_kw * eco_hours / 1000.0) * eco_carbon_factor * eco_carbon_price
    euac_carbon = carbon1 * lvl_energy_factor
    
    maint1 = capex * (eco_maint_rate / 100.0)
    euac_maint = maint1 * lvl_maint_factor
    
    total_annual = euac_capex + euac_energy + euac_carbon + euac_maint
    
    print(f"[{t_name} (ID: {test_d*1000:5.1f}mm)] 총 연간비용: {total_annual:>10,.0f} 원 "
          f"(유속:{t_v:4.1f} m/s, 펌프:{t_kw:5.1f}kW, 전기세:{euac_energy:>8,.0f})")
    
    if total_annual < min_cost:
        min_cost = total_annual
        opt_name = t_name

print("==================================================")
print(f"✨ 최종 KS 규격 최적 경제관경(D_opt) : {opt_name}")
print(f"👉 이 때의 최저 연간 통합 비용(EUAC) : {min_cost:,.0f} 원/년")
print("==================================================")
