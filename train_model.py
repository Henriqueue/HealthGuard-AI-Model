import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

# 1. CARREGAMENTO DOS DADOS
print("🔄 Carregando dataset...")
try:
    df = pd.read_csv('smoking_drinking_dataset.csv')
    print(f"✅ Dados carregados! {df.shape[0]} pacientes encontrados.")
except FileNotFoundError:
    print("❌ Erro: Arquivo 'smoking_drinking_dataset.csv' não encontrado. Verifique a pasta.")
    exit()

# 2. ENGENHARIA DE DADOS (CRIANDO O "ALVO")
# Aqui definimos a regra médica: Quem é considerado "Risco" (1) vs "Controle" (0)?
# Regra: Bebe E tem enzimas alteradas OU Fuma E tem pressão/gordura alterada.

print("⚙️ Criando indicadores de risco...")

# Limites de referência (simplificados para o modelo)
LIMIT_GTP = 50
LIMIT_ALT = 45
LIMIT_PRESSAO = 140
LIMIT_TRIGLIC = 200 # Triglicerídeos alto

def definir_risco(row):
    # Critério 1: Risco Hepático (Bebe + Enzimas Altas)
    risco_hepatico = (row['DRK_YN'] == 'Y') and (
        row['gamma_GTP'] > LIMIT_GTP or 
        row['SGOT_ALT'] > LIMIT_ALT
    )
    
    # Critério 2: Risco Cardiovascular (Fuma + Pressão ou Gordura Alta)
    # SMK_stat_type_cd: 1(Nunca), 2(Ex), 3(Fumante)
    risco_cardio = (row['SMK_stat_type_cd'] == 3) and (
        row['SBP'] > LIMIT_PRESSAO or 
        row['triglyceride'] > LIMIT_TRIGLIC
    )
    
    if risco_hepatico or risco_cardio:
        return 1 # ALERTA VERMELHO
    return 0 # BAIXO RISCO

# Aplica a função linha a linha
df['Risk_Flag'] = df.apply(definir_risco, axis=1)

print(f"📊 Distribuição de Risco:\n{df['Risk_Flag'].value_counts(normalize=True)}")

# 3. PRÉ-PROCESSAMENTO
# Transformar texto em números para a IA entender

# Mapeamento de Sexo: Male -> 0, Female -> 1
df['sex'] = df['sex'].map({'Male': 0, 'Female': 1})

# Mapeamento de Bebida: N -> 0, Y -> 1
df['DRK_YN'] = df['DRK_YN'].map({'N': 0, 'Y': 1})

# Seleção de Features (O que a IA vai olhar para decidir)(CORRIGIDA)
features = [
    'age', 'sex', 'weight', 'waistline', # Perfil
    'SBP', 'DBP', # Pressão
    'tot_chole', 'LDL_chole', 'triglyceride', # Gorduras (Corrigido aqui: LDL -> LDL_chole)
    'hemoglobin', 'gamma_GTP', 'SGOT_ALT', 'SGOT_AST', # Sangue/Fígado
    'SMK_stat_type_cd', 'DRK_YN' # Hábitos
]

X = df[features] # Dados de entrada
y = df['Risk_Flag'] # O que queremos prever

# Divisão Treino (80%) e Teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalização (Colocar tudo na mesma escala numérica)
# Ex: Idade (50) e GTP (200) ficam em escalas comparáveis
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. TREINAMENTO DO MODELO
# Usaremos Random Forest: Robusto e explicável
print("\n🧠 Treinando a Inteligência Artificial (Isso pode levar alguns segundos)...")
model = RandomForestClassifier(
    n_estimators=100, # Número de árvores de decisão
    max_depth=15,     # Profundidade máxima (evita decorar)
    class_weight='balanced', # Força a IA a prestar atenção nos casos de Risco (minoria)
    random_state=42,
    n_jobs=-1 # Usa todos os núcleos do processador
)

model.fit(X_train_scaled, y_train)

# 5. AVALIAÇÃO
print("\n✅ Treinamento concluído! Avaliando performance...")
y_pred = model.predict(X_test_scaled)

# Relatório focado em Recall (Sensibilidade) - Importante para saúde!
print(classification_report(y_test, y_pred))
print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred))

# 6. SALVAMENTO (EXPORTAÇÃO)
# Salvamos o Modelo e o Scaler para o App usar depois
print("\n💾 Salvando o cérebro da IA...")
joblib.dump(model, 'modelo_healthguard.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("🎉 Arquivos 'modelo_healthguard.pkl' e 'scaler.pkl' criados com sucesso!")