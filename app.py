import streamlit as st
import pandas as pd
import joblib
import numpy as np

# CONFIGURAÇÃO DA PÁGINA
st.set_page_config(
    page_title="HealthGuard AI",
    page_icon="🩺",
    layout="wide"
)

# CARREGAR O CÉREBRO DA IA
@st.cache_resource # Isso faz o app carregar rápido sem reler o arquivo toda hora
def carregar_modelo():
    try:
        modelo = joblib.load('modelo_healthguard.pkl')
        scaler = joblib.load('scaler.pkl')
        return modelo, scaler
    except:
        return None, None

model, scaler = carregar_modelo()

# TÍTULO E CABEÇALHO
st.title("🩺 HealthGuard AI")
st.markdown("### Sistema de Alerta Precoce para Risco Hepático e Cardiovascular")
st.markdown("---")

# SE O MODELO NÃO CARREGAR
if model is None:
    st.error("❌ Erro: Arquivos do modelo não encontrados. Rode o 'train_model.py' primeiro!")
    st.stop()

# --- BARRA LATERAL (ENTRADA DE DADOS) ---
st.sidebar.header("📝 Dados do Paciente")

def user_input_features():
    # Dados Demográficos
    st.sidebar.subheader("Perfil")
    age = st.sidebar.slider("Idade", 18, 90, 40)
    sex = st.sidebar.selectbox("Sexo Biológico", ["Masculino", "Feminino"])
    
    # Medidas
    st.sidebar.subheader("Antropometria")
    weight = st.sidebar.number_input("Peso (kg)", 40, 150, 75)
    waistline = st.sidebar.number_input("Cintura (cm)", 50, 150, 85)
    
    # Hábitos
    st.sidebar.subheader("Hábitos")
    smoke_opt = st.sidebar.selectbox("Tabagismo", 
                                   ["Nunca fumou", "Ex-fumante", "Fumante Atual"])
    drink_opt = st.sidebar.selectbox("Consome Álcool?", ["Não", "Sim"])

    # Sinais Vitais
    st.sidebar.subheader("Sinais Vitais")
    sbp = st.sidebar.slider("Pressão Sistólica (Alta)", 80, 200, 120)
    dbp = st.sidebar.slider("Pressão Diastólica (Baixa)", 50, 120, 80)

    # Exames de Sangue (Lipídios)
    st.sidebar.subheader("Perfil Lipídico (Colesterol)")
    tot_chole = st.sidebar.number_input("Colesterol Total", 100, 400, 190)
    ldl = st.sidebar.number_input("LDL (Ruim)", 50, 300, 110)
    triglyceride = st.sidebar.number_input("Triglicerídeos", 50, 500, 130)

    # Exames de Sangue (Fígado e Outros)
    st.sidebar.subheader("Marcadores Hepáticos/Sangue")
    hemoglobin = st.sidebar.number_input("Hemoglobina", 10.0, 20.0, 15.0)
    gamma_gtp = st.sidebar.number_input("Gamma GTP (Fígado)", 10, 500, 40)
    sgot_alt = st.sidebar.number_input("ALT (TGP)", 10, 500, 30)
    sgot_ast = st.sidebar.number_input("AST (TGO)", 10, 500, 30)

    # TRADUÇÃO DOS DADOS PARA A LINGUAGEM DA IA
    # Precisamos converter texto para números igual fizemos no treino
    
    sex_num = 0 if sex == "Masculino" else 1
    drink_num = 0 if drink_opt == "Não" else 1
    
    # Mapeamento Fumo: 1(Nunca), 2(Ex), 3(Atual)
    smoke_map = {"Nunca fumou": 1, "Ex-fumante": 2, "Fumante Atual": 3}
    smoke_num = smoke_map[smoke_opt]

    # Criar o dicionário de dados na MESMA ORDEM do treino
    data = {
        'age': age,
        'sex': sex_num,
        'weight': weight,
        'waistline': waistline,
        'SBP': sbp,
        'DBP': dbp,
        'tot_chole': tot_chole,
        'LDL_chole': ldl,  # Nome corrigido
        'triglyceride': triglyceride,
        'hemoglobin': hemoglobin,
        'gamma_GTP': gamma_gtp,
        'SGOT_ALT': sgot_alt,
        'SGOT_AST': sgot_ast,
        'SMK_stat_type_cd': smoke_num,
        'DRK_YN': drink_num
    }
    
    return pd.DataFrame(data, index=[0])

# Captura os dados
input_df = user_input_features()

# --- ÁREA PRINCIPAL (DASHBOARD) ---

# Botão para processar
if st.button("🔍 ANALISAR RISCO AGORA", use_container_width=True):
    
    # 1. Normalizar os dados (usando a mesma régua do treino)
    input_scaled = scaler.transform(input_df)
    
    # 2. Fazer a previsão
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1] # Chance de ser Risco (0 a 1)

    # 3. Exibir Resultados
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Diagnóstico da IA")
        if prediction[0] == 1:
            st.error("⚠️ ALERTA: RISCO DETECTADO")
            st.metric(label="Probabilidade de Risco", value=f"{probability:.1%}")
        else:
            st.success("✅ BAIXO RISCO APARENTE")
            st.metric(label="Segurança", value=f"{(1-probability):.1%}")

    with col2:
        st.subheader("Análise de Fatores")
        # Lógica simples para explicar o porquê (Explainable AI simplificado)
        
        # Checa Fígado
        if input_df['gamma_GTP'][0] > 50 or input_df['SGOT_ALT'][0] > 45:
            st.warning("🚨 **Atenção Hepática:** Suas enzimas (GTP/ALT) estão elevadas. Se houver consumo de álcool, o risco de dano hepático é alto.")
        else:
            st.info("🔹 Fígado: Biomarcadores dentro do esperado.")

        # Checa Coração