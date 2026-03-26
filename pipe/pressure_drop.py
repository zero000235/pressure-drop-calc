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
    """CoolProp을 이용해 유체의 밀도(ρ)와 동적 점성계수(μ)를 반환합니다."""
    T_K = temp_c + 273.15
    P = 101325.0
    rho = CP.PropsSI("D", "T", T_K, "P", P, fluid)
    mu  = CP.PropsSI("V", "T", T_K, "P", P, fluid)
    return rho, mu

# 배관 재질별 절대 조도 ε (단위: m)
ROUGHNESS = {
    "Commercial Steel (상업용 강관)": 4.6e-5,
    "Galvanized Steel (아연도금 강관)": 1.5e-4,
    "Cast Iron (주철관)": 2.6e-4,
    "Concrete (콘크리트관)": 1.5e-3,
    "PVC / Smooth Pipe (매끈한 관)": 1.5e-6,
    "Drawn Tubing (인발 튜브)": 1.5e-6,
    "Stainless Steel (스테인리스 강관)": 1.5e-5,
}

# 관부속품 K-factor (국부 손실 계수)
MINOR_LOSSES = {
    "90도 엘보우 (Standard)": 0.75,
    "45도 엘보우 (Standard)": 0.40,
    "게이트 밸브 (Fully open)": 0.15,
    "글로브 밸브 (Fully open)": 10.0,
    "스윙 체크 밸브": 2.0,
    "티 (Straight run)": 0.60,
    "티 (Branch flow)": 1.80,
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

def calc_pressure_dp(f: float, L: float, D: float, rho: float, v: float, sum_K: float) -> tuple:
    """
    압력강하 (마찰손실 + 국부손실) 계산 (Pa 단위)
    마찰 손실 (Darcy-Weisbach): ΔP_fric = f * (L/D) * (ρ*v²/2)
    국부 손실: ΔP_minor = ΣK * (ρ*v²/2)
    """
    D = max(D, 1e-9)
    dynamic_pressure = rho * v**2 / 2.0
    dp_fric = f * (L / D) * dynamic_pressure
    dp_minor = sum_K * dynamic_pressure
    dp_total = dp_fric + dp_minor
    return dp_fric, dp_minor, dp_total

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
        <p>Darcy-Weisbach 직관 손실 &nbsp;|&nbsp; 국부 손실 계산 &nbsp;|&nbsp; 필요 펌프 동력 &nbsp;|&nbsp; 시스템 성능 곡선 시각화</p>
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

        d_unit = st.radio("내경 단위 선택", ["mm", "m"], horizontal=True)
        if d_unit == "mm":
            D_input = st.number_input("배관 내경 (mm)", min_value=5.0, value=50.0, step=5.0)
            D = D_input / 1000.0
        else:
            D_input = st.number_input("배관 내경 (m)", min_value=0.01, value=0.05, step=0.01)
            D = D_input
            
        L = st.number_input("배관 총 직관 길이 (m)", min_value=0.1, value=50.0, step=5.0)
        
        material = st.selectbox("배관 재질 (절대조도)", list(ROUGHNESS.keys()))
        epsilon = ROUGHNESS[material]
        
        st.markdown("---")
        # ── 3. 국부 손실 ───────────────────────
        # 🌟 기능 1: 국부 손실 (Minor Losses) 계산 추가
        st.subheader("🌟 3. 국부 손실 (Minor Losses)")
        st.caption("배관 시스템 내 설치된 밸브 및 피팅류 갯수를 입력하세요.")
        minor_loss_counts = {}
        for item in MINOR_LOSSES.keys():
            minor_loss_counts[item] = st.number_input(f"{item} 수량", min_value=0, value=0, step=1)
            
        sum_K = sum(count * MINOR_LOSSES[item] for item, count in minor_loss_counts.items())
        if sum_K > 0:
            st.success(f"**총 저항계수 (ΣK) = {sum_K:.2f}**")

        st.markdown("---")
        # ── 4. 펌프 동력 ───────────────────────
        # 🌟 기능 2: 펌프 동력 (Pump Power) 계산 추가
        st.subheader("⚡ 4. 펌프 동력 (Pump Power)")
        pump_eff = st.slider("펌프 모터 종합 효율 (%)", min_value=30, max_value=100, value=70, step=1)
        
        st.markdown("<br>", unsafe_allow_html=True)
        calc_btn = st.button("🚀 전체 시스템 시뮬레이션 및 차트 업데이트", type="primary", use_container_width=True)


    # ══════════════════════════════════════════
    # 메인 화면 : 결과 출력
    # ══════════════════════════════════════════
    if calc_btn:
        # 단일 설계점 포인트 계산 실행
        v = calc_velocity(Q_m3s, D)
        Re = calc_reynolds(rho, v, D, mu)
        f, regime = calc_friction_factor(Re, D, epsilon)
        dp_fric, dp_minor, dp_total = calc_pressure_dp(f, L, D, rho, v, sum_K)
        power_kw = calc_pump_power(dp_total, Q_m3s, pump_eff)
        
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
            st.write(f"🔸 **국부(부속품) 손실**: {dp_minor/1e5:.4f} bar ({dp_minor/safe_total*100:.1f}%)")
            st.write(f"📏 **단위 길이당 직관 압력구배**: {dp_fric/L:.1f} Pa/m")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.progress(min(dp_fric/safe_total, 1.0), text="🟦 직관 마찰 손실 (Friction Loss) 비율")
            st.progress(min(dp_minor/safe_total, 1.0), text="🟧 국부 손실 (Minor Loss) 비율")
            st.markdown("</div>", unsafe_allow_html=True)
            
        # 🌟 기능 3: 유량-압력강하 시스템 곡선 시각화
        st.markdown("<div class='res-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>📈 유량-압력강하 시스템 곡선 (System Curve)</div>", unsafe_allow_html=True)
        
        # 시스템 곡선 데이터 생성 (0 부터 현재 입력 유량의 1.5배 범위까지)
        sim_q_m3s = np.linspace(0, max(Q_m3s, 0.0001) * 1.5, 30)
        dp_list_bar = []
        
        for q_val in sim_q_m3s:
            if q_val <= 0:
                dp_list_bar.append(0.0)
                continue
            v_i = calc_velocity(q_val, D)
            re_i = calc_reynolds(rho, v_i, D, mu)
            f_i, _ = calc_friction_factor(re_i, D, epsilon)
            _, _, dp_tot_i = calc_pressure_dp(f_i, L, D, rho, v_i, sum_K)
            dp_list_bar.append(dp_tot_i / 1e5) # bar 단위 변환
            
        # x축 단위를 사용자가 선택한 단위에 맞춰서 변환 표시
        if q_unit == "L/min":
            sim_q_plot = sim_q_m3s * 60000.0
            x_label = "Flow Rate (L/min)"
        else:
            sim_q_plot = sim_q_m3s
            x_label = "Flow Rate (m³/s)"
            
        df_curve = pd.DataFrame({
            x_label: sim_q_plot,
            "Total Pressure Drop (bar)": dp_list_bar
        })
        
        df_curve.set_index(x_label, inplace=True)
        
        # Streamlit 차트 시각화
        st.line_chart(df_curve, y="Total Pressure Drop (bar)", color="#2563EB")
        
        st.caption(f"💡 이 곡선(System Curve)은 현재 입력된 배관 직경, 길이, 재질 및 국부 요소들이 유량 변화에 어떠한 압력 손실(저항)을 일으키는지를 나타냅니다. 2차 곡선의 가파름은 유속 제곱에 비례하는 저항의 특징을 보여줍니다. (목표 설계 유량: **{Q_input} {q_unit}**)")
        st.markdown("</div>", unsafe_allow_html=True)
        
    else:
        # 계산 전 초기화면 안내
        st.markdown("""
        <div style='text-align:center; padding: 3rem 2rem; color:#5D6D7E; background:white; border-radius:12px; box-shadow:0 4px 6px rgba(0,0,0,0.05);'>
            <div style='font-size:4rem; margin-bottom:1rem;'>🛠️</div>
            <h3 style='color:#1E3A8A;'>좌측 사이드바에서 유체 및 시스템 조건을 입력하고<br>
            <b>[🚀 전체 시스템 시뮬레이션 및 차트 업데이트]</b> 버튼을 누르세요.</h3>
            <p style='margin-top:1rem; font-size:1rem; line-height: 1.6;'>
                본 프로그램은 단순한 직선 배관 마찰뿐만 아니라 엘보나 밸브에 의한 <b>국부 손실(Minor Losses)</b>을 함께 분석하고,<br>
                유체 이송에 필요한 <b>펌프 요구 동력(Pump Power)</b>을 전력(kW) 단위로 도출해 드립니다.<br>
                또한 유량 변화에 따른 <b>시스템 저항 곡선(System Curve)</b>을 시각화하여 설계 최적화에 기여합니다.
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    build_ui()