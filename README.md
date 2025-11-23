# 🩺 HealthGuard AI - Monitor de Risco Hepático e Cardiovascular

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)
![Status](https://img.shields.io/badge/Status-Prototype-green)

## 📖 Sobre o Projeto

O **HealthGuard AI** é uma solução de Inteligência Artificial voltada para a medicina preventiva. Diferente de modelos tradicionais que apenas classificam hábitos (como "Fumante" ou "Bebedor"), este sistema foca em identificar **sinais silenciosos de estresse metabólico**.

O modelo cruza dados demográficos, antropométricos e exames laboratoriais para detectar padrões que indicam risco iminente de doenças hepáticas ou cardiovasculares, muitas vezes antes do aparecimento de sintomas clínicos graves.

### 🎯 Objetivo
Democratizar a triagem de saúde, oferecendo um "radar" que alerta pacientes sobre a necessidade de intervenção médica baseada na combinação complexa de seus biomarcadores.

## ⚙️ Funcionalidades

* **Análise de Risco em Tempo Real:** Previsão instantânea baseada em dados do usuário.
* **Engenharia de Dados Médicos:** Criação de *targets* sintéticos baseados em literatura médica (ex: enzimas hepáticas altas + consumo de álcool).
* **Interface Amigável:** Dashboard interativo construído com Streamlit para uso em clínicas ou por pacientes.
* **Foco em Sensibilidade (Recall):** O modelo foi otimizado para minimizar falsos negativos, priorizando a segurança do paciente.

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python
* **Manipulação de Dados:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Random Forest / XGBoost)
* **Persistência do Modelo:** Joblib
* **Frontend/Web App:** Streamlit
* **Dataset:** [Smoking and Drinking Dataset (Kaggle)](https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset)

## 🚀 Como Rodar o Projeto

### Pré-requisitos
Certifique-se de ter o Python instalado.

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/SEU-USUARIO/healthguard-ai.git](https://github.com/SEU-USUARIO/healthguard-ai.git)
   cd healthguard-ai

2. **Instale as dependências:**
   ```bash
   pip install pandas scikit-learn joblib matplotlib seaborn streamlit
   
3. **Treine a Inteligência Artificial: Execute o script que processa os dados e gera o modelo (.pkl):**
      ```bash
   python train_model.py

4. **Inicie o Aplicativo:**
      ```bash
   python -m streamlit run app.py

## 📊 Metodologia e Resultados
### O projeto seguiu um fluxo rigoroso de Data Science:

    EDA (Análise Exploratória): Validação da integridade dos dados e correlações (ex: Idade vs Pressão, Enzimas vs Álcool).

    Pré-processamento: Normalização (StandardScaler) e Codificação de variáveis categóricas.

    Modelagem: Treinamento supervisionado com foco na métrica de Recall para reduzir riscos de saúde não detectados.

## 🔮 Próximos Passos (Roadmap)
    [ ] Implementar Triagem em Duas Etapas (Modelo simplificado para quem não tem exames de sangue).
    [ ] Adicionar leitura de exames via OCR (upload de PDF/Foto).
    [ ] Integração com bibliotecas de Explainable AI (SHAP) para detalhar o porquê do risco.

##🤝 Contribuição
### Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.
