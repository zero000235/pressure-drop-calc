# =============================================================================
# 배관 압력강하 계산기 (Darcy-Weisbach 방정식 및 고급 기능)
# =============================================================================
# 실행 방법:
#   streamlit run pressure_drop.py
# =============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import CoolProp.CoolProp as CP

# ─────────────────────────────────────────────
# [모듈 1] 계산 로직 함수 모음
# ─────────────────────────────────────────────

# 💧 지원 유체 목록
FLUID_OPTIONS = {
    "Water"            : "물 (Water)",
    "Methanol"         : "메탄올 (Methanol)",
    "Ethanol"          : "에탄올 (Ethanol)",
    "INCOMP::MEG[0.5]" : "에틸렌 글리콜 (부동액 50% 수용액)",
    "INCOMP::MPG[0.5]" : "프로필렌 글리콜 (부동액 50% 수용액)",
    "Acetone"          : "아세톤 (Acetone)",
    "Benzene"          : "벤젠 (Benzene)",
    "Toluene"          : "톨루엔 (Toluene)",
}

def get_fluid_properties(fluid: str, temp_c: float) -> tuple:
    """
    CoolProp을 이용해 유체의 밀도(ρ)와 동적 점성계수(μ)를 반환합니다.
    입력된 온도에서 유체가 기화되지 않도록 강제로 액체(Liquid) 물성치를 반환합니다.
    """
    T_K = temp_c + 273.15
    P = 101325.0
    
    if fluid.startswith("INCOMP::"):
        # 비압축성 부동액 등은 항상 액체이므로 P 베이스로 정상 계산
        rho = CP.PropsSI("D", "T", T_K, "P", P, fluid)
        mu  = CP.PropsSI("V", "T", T_K, "P", P, fluid)
    else:
        try:
            # 순수 물질(Water 등)은 포화 액체 건도(Q=0, Saturated Liquid) 라인을 항상 보장하도록 강제
            rho = CP.PropsSI("D", "T", T_K, "Q", 0, fluid)
            mu  = CP.PropsSI("V", "T", T_K, "Q", 0, fluid)
        except ValueError:
            # 임계 온도를 넘어가 Q=0이 불가능한 극한의 경우 1atm 베이스 폴백
            rho = CP.PropsSI("D", "T", T_K, "P", P, fluid)
            mu  = CP.PropsSI("V", "T", T_K, "P", P, fluid)
            
    return rho, mu

# 배관 재질별 절대 조도 ε (단위: m)
ROUGHNESS = {
    "Smooth Pipe (초매끈한 관, ε=0)": 0.0,
    "PVC (일반 플라스틱 관)": 1.5e-6,
    "Commercial Steel (상업용 강관)": 4.6e-5,
    "Galvanized Steel (아연도금 강관)": 1.5e-4,
    "Cast Iron (주철관)": 2.6e-4,
    "Concrete (콘크리트관)": 1.5e-3,
    "Drawn Tubing (인발 튜브)": 1.5e-6,
    "Stainless Steel (스테인리스 강관)": 1.5e-5,
}

# 관부속품 K-factor (피팅류 계수)
FITTING_LOSSES = {
    "90도 엘보우 (Standard)": 0.75,
    "45도 엘보우 (Standard)": 0.40,
    "티 (Straight run)": 0.60,
    "티 (Branch flow)": 1.80,
}
# 밸브류 K-factor (밸브 저항 계수)
VALVE_LOSSES = {
    "게이트 밸브 (Fully open)": 0.15,
    "글로브 밸브 (Fully open)": 10.0,
    "스윙 체크 밸브": 2.0,
}

# KS 배관 규격 (KS D 3576 / JIS G 3459 스테인리스 강관) 제원 데이터
# 형식: {"OD": 바깥지름(mm), "SCH 5S": 두께(mm), ...}
KS_PIPE_DATA = {
    "15A (1/2B)":   {"OD": 21.7, "SCH 5S": 1.65, "SCH 10S": 2.1, "SCH 20S": 2.5, "SCH 40": 2.8, "SCH 80": 3.7},
    "20A (3/4B)":   {"OD": 27.2, "SCH 5S": 1.65, "SCH 10S": 2.1, "SCH 20S": 2.5, "SCH 40": 2.9, "SCH 80": 3.9},
    "25A (1B)":     {"OD": 34.0, "SCH 5S": 1.65, "SCH 10S": 2.8, "SCH 20S": 3.0, "SCH 40": 3.4, "SCH 80": 4.5},
    "32A (1 1/4B)": {"OD": 42.7, "SCH 5S": 1.65, "SCH 10S": 2.8, "SCH 20S": 3.0, "SCH 40": 3.6, "SCH 80": 4.9},
    "40A (1 1/2B)": {"OD": 48.6, "SCH 5S": 1.65, "SCH 10S": 2.8, "SCH 20S": 3.0, "SCH 40": 3.7, "SCH 80": 5.1},
    "50A (2B)":     {"OD": 60.5, "SCH 5S": 1.65, "SCH 10S": 2.8, "SCH 20S": 3.5, "SCH 40": 3.9, "SCH 80": 5.5},
    "65A (2 1/2B)": {"OD": 76.3, "SCH 5S": 2.1,  "SCH 10S": 3.0, "SCH 20S": 3.5, "SCH 40": 5.2, "SCH 80": 7.0},
    "80A (3B)":     {"OD": 89.1, "SCH 5S": 2.1,  "SCH 10S": 3.0, "SCH 20S": 4.0, "SCH 40": 5.5, "SCH 80": 7.6},
    "90A (3 1/2B)": {"OD": 101.6,"SCH 5S": 2.1,  "SCH 10S": 3.0, "SCH 20S": 4.0, "SCH 40": 5.7, "SCH 80": 8.1},
    "100A (4B)":    {"OD": 114.3,"SCH 5S": 2.1,  "SCH 10S": 3.0, "SCH 20S": 4.0, "SCH 40": 6.0, "SCH 80": 8.6},
    "125A (5B)":    {"OD": 139.8,"SCH 5S": 2.8,  "SCH 10S": 3.4, "SCH 20S": 5.0, "SCH 40": 6.6, "SCH 80": 9.5},
    "150A (6B)":    {"OD": 165.2,"SCH 5S": 2.8,  "SCH 10S": 3.4, "SCH 20S": 5.0, "SCH 40": 7.1, "SCH 80": 11.0},
    "200A (8B)":    {"OD": 216.3,"SCH 5S": 2.8,  "SCH 10S": 4.0, "SCH 20S": 6.5, "SCH 40": 8.2, "SCH 80": 12.7},
    "250A (10B)":   {"OD": 267.4,"SCH 5S": 3.4,  "SCH 10S": 4.0, "SCH 20S": 6.5, "SCH 40": 9.3, "SCH 80": 15.1},
    "300A (12B)":   {"OD": 318.5,"SCH 5S": 4.0,  "SCH 10S": 4.5, "SCH 20S": 6.5, "SCH 40": 10.3,"SCH 80": 17.4},
}

def calc_velocity(Q_m3s: float, D: float) -> float:
    """유속(v) 계산: Q / A"""
    # 내경이 0인 경우 방어를 위해 아주 작은 값 추가
    D = max(D, 1e-9)
    A = np.pi / 4.0 * D**2
    return Q_m3s / A

def calc_reynolds(rho: float, v: float, D: float, mu: float) -> float:
    """레이놀즈수(Re) 계산: ρ·v·D / μ"""
    if mu <= 0: return float('inf')
    return (rho * v * D) / mu

def calc_friction_factor(Re: float, D: float, epsilon: float) -> tuple:
    """
    마찰계수(f) 계산:
    - 층류 (Re < 2300): 64/Re
    - 난류 (Re >= 2300): Swamee-Jain 방정식 (1976) 적용
    """
    if Re < 1e-6:
        return 0.0, "정지 (No Flow)"
    if Re < 2300:
        f = 64.0 / Re
        regime = "층류 (Laminar)"
    else:
        D = max(D, 1e-9)
        relative_roughness = epsilon / D
        # Swamee-Jain 방정식
        denom = np.log10(relative_roughness / 3.7 + 5.74 / (Re**0.9))
        f = 0.25 / denom**2
        regime = "전이구간 (Transitional)" if Re < 4000 else "난류 (Turbulent)"
    return f, regime

def calc_pressure_dp(f: float, L: float, D: float, rho: float, v: float, sum_K_fit: float, sum_K_valve: float) -> tuple:
    """
    압력강하 (마찰손실 + 피팅/밸브 손실) 계산 (Pa 단위)
    마찰 손실 (Darcy-Weisbach): ΔP_fric = f * (L/D) * (ρ*v²/2)
    피팅 손실: ΔP_fit = ΣK_fit * (ρ*v²/2)
    밸브 손실: ΔP_valve = ΣK_valve * (ρ*v²/2)
    """
    D = max(D, 1e-9)
    dynamic_pressure = rho * v**2 / 2.0
    dp_fric = f * (L / D) * dynamic_pressure
    dp_fit = sum_K_fit * dynamic_pressure
    dp_valve = sum_K_valve * dynamic_pressure
    dp_total = dp_fric + dp_fit + dp_valve
    return dp_fric, dp_fit, dp_valve, dp_total

def calc_pump_power(dp_pa: float, Q_m3s: float, eff: float) -> float:
    """
    펌프 소요 동력(kW) 계산
    W = (ΔP * Q) / η
    """
    if eff <= 0: return 0.0
    power_watts = (dp_pa * Q_m3s) / (eff / 100.0)
    return power_watts / 1000.0


# ─────────────────────────────────────────────
# [모듈 2] Streamlit UI 구성
# ─────────────────────────────────────────────

def build_ui():
    st.set_page_config(page_title="배관 설계 시뮬레이터", page_icon="🚀", layout="wide")
    
    # ── 커스텀 CSS (색상·다이나믹 디자인) ─────────
    st.markdown("""
    <style>
        .stApp { background-color: #F8F9FA; }
        .main-header {
            background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
            padding: 2.5rem; border-radius: 16px;
            color: white; margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }
        .main-header h1 { margin: 0; font-size: 2.5rem; color: white; font-weight: 800;}
        .main-header p  { margin-top: 0.5rem; opacity: 0.9; font-size: 1.1rem; }
        .section-header { 
            font-size: 1.25rem; font-weight: 700; color: #1E3A8A; 
            margin-top: 1.5rem; margin-bottom: 0.8rem; 
            border-bottom: 2px solid #E5E7EB; padding-bottom: 0.3rem;
        }
        .res-card { 
            background: white; padding: 1.5rem; border-radius: 12px; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 1.5rem;
            transition: transform 0.2s ease-in-out;
        }
        .res-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 15px rgba(0,0,0,0.1);
        }
        .badge { 
            display: inline-flex; align-items: center; justify-content: center;
            padding: 0.35rem 1rem; border-radius: 999px; 
            font-weight: 700; font-size: 1rem; margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .badge-laminar { background:#D1FAE5; color:#065F46; border: 1px solid #34D399; }
        .badge-transitional { background:#FEF3C7; color:#92400E; border: 1px solid #FBBF24; }
        .badge-turbulent { background:#DBEAFE; color:#1E3A8A; border: 1px solid #60A5FA; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-header">
        <h1>🚀 프리미엄 배관 시스템 설계 계산기</h1>
        <p>Darcy-Weisbach 직관 손실 &nbsp;|&nbsp; 피팅 및 밸브 손실 계산 &nbsp;|&nbsp; 필요 펌프 동력</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ══════════════════════════════════════════
    # 사이드바 : 입력 패널
    # ══════════════════════════════════════════
    with st.sidebar:
        st.header("⚙️ 시스템 조건 입력")
        
        # ── 1. 유체 정보 ───────────────────────
        st.subheader("💧 1. 유체(Fluid) 특성")
        
        fluid_options_list = list(FLUID_OPTIONS.values()) + ["직접 입력 (Manual)"]
        fluid_display = st.selectbox("유체 종류", fluid_options_list, index=0)
        
        if fluid_display == "직접 입력 (Manual)":
            rho = st.number_input("밀도 ρ (kg/m³)", value=1000.0, step=10.0, format="%.2f")
            mu = st.number_input("동적 점성계수 μ (Pa·s)", value=0.001, step=0.0001, format="%.6f")
        else:
            fluid_key = [k for k, v in FLUID_OPTIONS.items() if v == fluid_display][0]
            temp_c = st.number_input("유체 온도 (°C)", min_value=-50.0, max_value=500.0, value=20.0, step=1.0)
            
            try:
                rho, mu = get_fluid_properties(fluid_key, temp_c)
                st.info(f"**밀도:** {rho:.2f} kg/m³  \n**점성계수:** {mu:.4e} Pa·s")
            except Exception as e:
                st.error(f"⚠️ 물성치 계산 오류: {e}")
                rho, mu = 1000.0, 0.001

        st.markdown("---")
        # ── 2. 배관 제원 ───────────────────────
        st.subheader("🔩 2. 배관(Pipe) 및 유량 제원")
        
        q_unit = st.radio("유량 단위 선택", ["m³/s", "L/min"], horizontal=True)
        if q_unit == "L/min":
            Q_input = st.number_input("설계 체적 유량 (L/min)", min_value=0.1, value=100.0, step=10.0)
            Q_m3s = Q_input / 60000.0
        else:
            Q_input = st.number_input("설계 체적 유량 (m³/s)", min_value=0.0001, value=0.005, step=0.001, format="%.4f")
            Q_m3s = Q_input

        d_input_method = st.radio("배관 내경 입력 방식", ["직접 입력", "규격표에서 선택 (KS/JIS 배관)"], horizontal=True)
        
        if d_input_method == "직접 입력":
            d_unit = st.radio("내경 단위 선택", ["mm", "m"], horizontal=True)
            if d_unit == "mm":
                D_input = st.number_input("배관 내경 (mm)", min_value=1.0, value=50.0, step=5.0)
                D = D_input / 1000.0
            else:
                D_input = st.number_input("배관 내경 (m)", min_value=0.001, value=0.05, step=0.01)
                D = D_input
        else:
            col_sc1, col_sc2 = st.columns(2)
            with col_sc1:
                nps_select = st.selectbox("호칭 지름", list(KS_PIPE_DATA.keys()), index=5) # 기본 50A
            with col_sc2:
                sch_select = st.selectbox("호칭 두께 (Schedule)", ["SCH 5S", "SCH 10S", "SCH 20S", "SCH 40", "SCH 80"], index=3)
                
            pipe_od = KS_PIPE_DATA[nps_select]["OD"]
            pipe_t = KS_PIPE_DATA[nps_select][sch_select]
            selected_inner_mm = pipe_od - (2 * pipe_t)
            
            st.info(f"📏 외경: **{pipe_od}** mm / 두께: **{pipe_t}** mm\n\n👉 계산된 실제 내경: **{selected_inner_mm:.2f} mm**")
            D = selected_inner_mm / 1000.0
            
        L = st.number_input("배관 총 직관 길이 (m)", min_value=0.1, value=50.0, step=5.0)
        
        material = st.selectbox("배관 재질 (절대조도)", list(ROUGHNESS.keys()))
        epsilon = ROUGHNESS[material]
        
        st.markdown("---")
        # ── 3. 부속품 및 밸브 손실 ───────────────────────
        st.subheader("🌟 3. 부속품 및 밸브 손실")
        st.caption("배관 시스템 내 설치된 밸브 및 피팅류 갯수를 입력하세요.")
        
        col_f, col_v = st.columns(2)
        fitting_counts, valve_counts = {}, {}
        with col_f:
            st.markdown("**[ 피팅류 ]**")
            for item in FITTING_LOSSES.keys():
                fitting_counts[item] = st.number_input(f"{item}", min_value=0, value=0, step=1, key=f"f_{item}")
        with col_v:
            st.markdown("**[ 밸브류 ]**")
            for item in VALVE_LOSSES.keys():
                valve_counts[item] = st.number_input(f"{item}", min_value=0, value=0, step=1, key=f"v_{item}")
            
        sum_K_fit = sum(count * FITTING_LOSSES[item] for item, count in fitting_counts.items())
        sum_K_valve = sum(count * VALVE_LOSSES[item] for item, count in valve_counts.items())
        if (sum_K_fit + sum_K_valve) > 0:
            st.success(f"**피팅 저항 (ΣK) = {sum_K_fit:.2f} / 밸브 저항 (ΣK) = {sum_K_valve:.2f}**")

        st.markdown("---")
        # ── 4. 펌프 동력 ───────────────────────
        # 🌟 기능 2: 펌프 동력 (Pump Power) 계산 추가
        st.subheader("⚡ 4. 펌프 동력 (Pump Power)")
        pump_eff = st.slider("펌프 모터 종합 효율 (%)", min_value=30, max_value=100, value=70, step=1)
        
        st.markdown("<br>", unsafe_allow_html=True)
        calc_btn = st.button("🚀 전체 시스템 시뮬레이션 결과 보기", type="primary", use_container_width=True)

    if 'calc_run' not in st.session_state:
        st.session_state['calc_run'] = False
        
    if calc_btn:
        st.session_state['calc_run'] = True

    # ══════════════════════════════════════════
    # 메인 화면 : 결과 출력
    # ══════════════════════════════════════════
    if st.session_state['calc_run']:
        # 단일 설계점 포인트 계산 실행
        v = calc_velocity(Q_m3s, D)
        Re = calc_reynolds(rho, v, D, mu)
        f, regime = calc_friction_factor(Re, D, epsilon)
        dp_fric, dp_fit, dp_valve, dp_total = calc_pressure_dp(f, L, D, rho, v, sum_K_fit, sum_K_valve)
        power_kw = calc_pump_power(dp_total, Q_m3s, pump_eff)
        
        tab1, tab2 = st.tabs(["📊 기본 시뮬레이션 결과", "⚖️ 단일 조건 비교 분석"])
        
        with tab1:
            # ── 흐름 상태 뱃지 출력 ───────────────
            if "층류" in regime: 
                regime_class = "badge-laminar"
                icon = "🟢"
            elif "전이" in regime: 
                regime_class = "badge-transitional"
                icon = "🟠"
            else: 
                regime_class = "badge-turbulent"
                icon = "🔵"
            
            st.markdown(
                f"<div><span class='badge {regime_class}'>{icon} 유체 흐름 체제: <b>{regime}</b></span></div>", 
                unsafe_allow_html=True
            )
            
            col1, col2 = st.columns([1.6, 1])
            with col1:
                st.markdown("<div class='res-card'>", unsafe_allow_html=True)
                st.markdown("<div class='section-header'>📊 주요 성능 지표 (KPI)</div>", unsafe_allow_html=True)
                
                # 최종 결과 매트릭 출력
                c1, c2, c3 = st.columns(3)
                c1.metric("총 시스템 압력강하", f"{dp_total / 1e5:.4f} bar")
                c2.metric("총 시스템 압력강하", f"{dp_total:,.1f} Pa")
                # 필요 펌프 동력 출력
                c3.metric("필요 펌프 동력 (전력)", f"{power_kw:.3f} kW", help=f"시스템 효율 {pump_eff}% 기준 소요 동력")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # 중간 계산 결과 출력
                c4, c5, c6 = st.columns(3)
                c4.metric("평균 유속 (v)", f"{v:.3f} m/s")
                c5.metric("레이놀즈수 (Re)", f"{Re:,.0f}")
                c6.metric("최종 마찰계수 (f)", f"{f:.5f}")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col2:
                st.markdown("<div class='res-card'>", unsafe_allow_html=True)
                st.markdown("<div class='section-header'>분석 요약 보고서</div>", unsafe_allow_html=True)
                
                # 0으로 나누기 방지
                safe_total = dp_total if dp_total > 0 else 1e-9
                
                st.write(f"🔹 **직관 마찰 손실**: {dp_fric/1e5:.4f} bar ({dp_fric/safe_total*100:.1f}%)")
                st.write(f"🔸 **피팅류 손실**: {dp_fit/1e5:.4f} bar ({dp_fit/safe_total*100:.1f}%)")
                st.write(f"🔺 **밸브류 손실**: {dp_valve/1e5:.4f} bar ({dp_valve/safe_total*100:.1f}%)")
                st.write(f"📏 **단위 길이당 직관 압력구배**: {dp_fric/L:.1f} Pa/m")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.progress(min(dp_fric/safe_total, 1.0), text="🟦 직관 마찰 손실 (Friction Loss)")
                st.progress(min(dp_fit/safe_total, 1.0), text="🟧 피팅류 손실 (Fitting Loss)")
                st.progress(min(dp_valve/safe_total, 1.0), text="🟥 밸브류 손실 (Valve Loss)")
                st.markdown("</div>", unsafe_allow_html=True)

        with tab2:
            st.markdown("<div class='res-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>⚖️ 대안 설계 (Alternative Design) 비교</div>", unsafe_allow_html=True)
            st.info("좌측 사이드바의 기본 조건(Base Case)을 유지한 채, **아래에서 선택한 변수 1개만 변경** 시 결과를 비교합니다.")
            
            compare_var = st.selectbox("변경할 변수 선택", [
                "1. 유체 온도 변경 (물성치 변화 적용)",
                "2. 체적 유량 변경",
                "2. 배관 내경 강제 변경",
                "2. 배관 총 직관 길이 변경",
                "3. 수동 저항계수(K) 추가",
                "4. 펌프 종합 효율 변경"
            ])
            
            alt_temp_c = temp_c if 'temp_c' in locals() else 20.0
            alt_Q_m3s = Q_m3s
            alt_D = D
            alt_L = L
            alt_K_add = 0.0
            alt_pump_eff = pump_eff
            
            st.markdown("---")
            if "온도" in compare_var:
                if fluid_display == "직접 입력 (Manual)":
                    st.warning("⚠️ 직접 입력하신 유체는 온도 변화에 따른 물성치 자동 계산이 불가합니다.")
                else:
                    alt_temp_c = st.number_input("대안 유체 온도 (°C)", value=alt_temp_c + 10.0, step=1.0)
            elif "유량" in compare_var:
                if q_unit == "L/min":
                    alt_Q_input = st.number_input("대안 체적 유량 (L/min)", value=Q_input + 20.0, step=10.0)
                    alt_Q_m3s = alt_Q_input / 60000.0
                else:
                    alt_Q_input = st.number_input("대안 체적 유량 (m³/s)", value=Q_input + 0.001, step=0.001, format="%.4f")
                    alt_Q_m3s = alt_Q_input
            elif "내경" in compare_var:
                alt_D_input = st.number_input("대안 배관 내경 (mm)", value=(D * 1000.0) + 10.0, step=5.0)
                alt_D = alt_D_input / 1000.0
            elif "길이" in compare_var:
                alt_L = st.number_input("대안 배관 길이 (m)", value=L + 10.0, step=5.0)
            elif "저항계수" in compare_var:
                alt_K_add = st.number_input("추가할 저항계수 K (단위 무차원, 필터류/밸브 추가 가정)", value=5.0, step=1.0)
            elif "효율" in compare_var:
                alt_pump_eff = st.slider("대안 펌프 효율 (%)", min_value=30, max_value=100, value=min(pump_eff + 10, 100), step=1)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # 대안 재계산 로직
            c_rho, c_mu = rho, mu
            if "온도" in compare_var and fluid_display != "직접 입력 (Manual)":
                try:
                    c_rho, c_mu = get_fluid_properties(fluid_key, alt_temp_c)
                except:
                    pass
                    
            alt_sum_K_valve = sum_K_valve + alt_K_add
            
            a_v = calc_velocity(alt_Q_m3s, alt_D)
            a_Re = calc_reynolds(c_rho, a_v, alt_D, c_mu)
            a_f, a_regime = calc_friction_factor(a_Re, alt_D, epsilon)
            a_dp_fric, a_dp_fit, a_dp_valve, a_dp_total = calc_pressure_dp(a_f, alt_L, alt_D, c_rho, a_v, sum_K_fit, alt_sum_K_valve)
            a_power_kw = calc_pump_power(a_dp_total, alt_Q_m3s, alt_pump_eff)
            
            # 비교 UI 출력
            col_b, col_a = st.columns(2)
            
            diff_dp = a_dp_total - dp_total
            diff_pow = a_power_kw - power_kw
            diff_v = a_v - v
            
            with col_b:
                st.markdown("<div style='background-color:#F8FAFC; padding:1.5rem; border-radius:12px; border-top: 4px solid #94A3B8;'>", unsafe_allow_html=True)
                st.markdown("<h3 style='color:#334155; margin-top:0;'>🔵 기본 조건 (Base)</h3>", unsafe_allow_html=True)
                st.write(f"**평균 유속**: {v:.3f} m/s")
                st.write(f"**총 압력강하**: {dp_total/1e5:.4f} bar")
                st.write(f"**필요 펌프 동력**: {power_kw:.3f} kW")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col_a:
                st.markdown("<div style='background-color:#EFF6FF; padding:1.5rem; border-radius:12px; border-top: 4px solid #3B82F6; box-shadow:0 4px 6px rgba(0,0,0,0.05);'>", unsafe_allow_html=True)
                st.markdown("<h3 style='color:#1E3A8A; margin-top:0;'>🟠 대안 조건 (Alternative)</h3>", unsafe_allow_html=True)
                st.metric("평균 유속 (m/s)", f"{a_v:.3f}", delta=f"{diff_v:.3f}", delta_color="normal")
                st.metric("총 압력강하 (bar)", f"{a_dp_total/1e5:.4f}", delta=f"{diff_dp/1e5:.4f}", delta_color="inverse")
                st.metric("필요 펌프 동력 (kW)", f"{a_power_kw:.3f}", delta=f"{diff_pow:.3f}", delta_color="inverse")
                st.markdown("</div>", unsafe_allow_html=True)
                
            st.markdown("</div>", unsafe_allow_html=True)
        
    else:
        # 계산 전 초기화면 안내
        st.markdown("""
        <div style='text-align:center; padding: 3rem 2rem; color:#5D6D7E; background:white; border-radius:12px; box-shadow:0 4px 6px rgba(0,0,0,0.05);'>
            <div style='font-size:4rem; margin-bottom:1rem;'>🛠️</div>
            <h3 style='color:#1E3A8A;'>좌측 사이드바에서 유체 및 시스템 조건을 입력하고<br>
            <b>[🚀 전체 시스템 시뮬레이션 결과 보기]</b> 버튼을 누르세요.</h3>
            <p style='margin-top:1rem; font-size:1rem; line-height: 1.6;'>
                본 프로그램은 단순한 직선 배관 마찰뿐만 아니라 <b>피팅 및 밸브에 의한 손실(Minor & Valve Losses)</b>을 명확하게 분리하여 분석하고,<br>
                유체 이송에 필요한 <b>펌프 요구 동력(Pump Power)</b>을 전력(kW) 단위로 정교하게 도출해 드립니다.
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    build_ui()