import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from intradaypricepredictor import IntradayPricePredictor

# Configurações de simulação
B = 200
T_sizes = [2000, 4000, 8000, 16000, 32000]
T_sizes = [4000, 16000]
seed = 1
np.random.seed(seed)

# Função para calcular π_t
def calculate_pi(y_lags, beta):
    eta = beta['Intercept']
    coef_keys = ['lag1', 'lag2', 'lag3', 'lag4', 'lag5']
    for i, key in enumerate(coef_keys):
        if len(y_lags) > i:
            eta += beta[key] * y_lags[i]
    return np.exp(eta) / (1 + np.exp(eta))

# Simular múltiplas séries
def simulate_multiple_series(B, T_sizes, beta):
    series_data = {}
    for T in T_sizes:
        data = np.zeros((B, T), dtype=int)
        for b in range(B):
            for t in range(5, T):
                y_lags = data[b, t-5:t][::-1]
                data[b, t] = np.random.binomial(1, calculate_pi(y_lags, beta))
        series_data[T] = data
    return series_data

# Criar DataFrame com defasagens
def create_lagged_df(series, lags):
    data = {'Y': series[lags:]}
    for i in range(1, lags + 1):
        data[f'lag{i}'] = series[lags - i:-i]
    return pd.DataFrame(data)

# Ajustar modelos com diferentes números de defasagens
def fit_models_formula(series, lags):
    df = create_lagged_df(series, lags)
    formula = 'Y ~ ' + ' + '.join([f'lag{i}' for i in range(1, lags + 1)])
    return smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

# Aplicar modelos nas séries simuladas
def apply_models_formula(simulated_data):
    model_results = []
    for T, data in simulated_data.items():
        for series in data:
            series_clean = series[5:]
            results = {}
            results['true_model'] = fit_models_formula(series_clean, 5)  # Modelo verdadeiro
            results['parsimonious_model'] = fit_models_formula(series_clean, 3)  # Modelo parcimonioso
            results['non_parsimonious_model'] = fit_models_formula(series_clean, 7)  # Modelo não parcimonioso
            model_results.append(results)
    return model_results

# Organizar estimativas dos coeficientes por modelo e coeficiente
def extract_coefficients(models, coef_names):
    coef_dict = {coef: [] for coef in coef_names}
    for model in models:
        for coef in coef_names:
            coef_dict[coef].append(model.params.get(coef, None))
    return coef_dict

# Calcular estatísticas-resumo
def calculate_statistics(models, coef_names):
    coef_dict = extract_coefficients(models, coef_names)
    stats = {}
    for coef, estimates in coef_dict.items():
        valid_estimates = np.array([est for est in estimates if not np.isnan(est)])  
        stats[coef] = {
            'mean': np.mean(valid_estimates),
            'variance': np.var(valid_estimates, ddof=1)
        }
    return stats

# Apresentar estatísticas-resumo
def display_statistics(true_values, T, stats):
    print(f"Estatísticas para T = {T}")
    for coef, values in stats.items():
        print(f"{coef} {true_values[coef]}, estimações: Média = {values['mean']:.4f}, Variância = {values['variance']:.4f}, diff = {values['mean'] - true_values[coef]:.4f}")
    print()

# Criar boxplots para cada coeficiente
def plot_boxplots(models, coef_names, true_values, title):
    coef_dict = extract_coefficients(models, coef_names)
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.Set3.colors
    
    for i, (coef, color) in enumerate(zip(coef_names, colors)):
        bp = ax.boxplot(coef_dict[coef], positions=[i+1], widths=0.6, patch_artist=True, boxprops=dict(facecolor=color))
        ax.axhline(y=true_values[coef], color=color, linestyle='--', label=f'{coef} (Valor Verdadeiro)')
    
    ax.set_title(title)
    ax.set_xlabel('Coeficiente')
    ax.set_ylabel('Estimativa')
    ax.legend()
    plt.savefig(f'boxplot_{np.random.randint(1000)}.png')
    plt.show()
    
def plot_boxplots_by_coefficient(models_by_T, coef_name, true_value, T_sizes, title_prefix):
    # Preparando os dados para o plot
    data_to_plot = []
    for T in T_sizes:
        models = models_by_T[T]
        estimates = [model.params.get(coef_name, None) for model in models]
        data_to_plot.append(estimates)

    # Criando o boxplot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data_to_plot, labels=T_sizes, patch_artist=True)
    ax.axhline(y=true_value, color='red', linestyle='--', label=f'Valor Verdadeiro ({coef_name})')

    # Configurações de visualização
    ax.set_title(f"{title_prefix} - {coef_name}")
    ax.set_xlabel('Tamanho da Série (T)')
    ax.set_ylabel('Estimativas')
    ax.legend()
    plt.savefig(f'boxplot_{coef_name}.png')
    plt.show()

def normalize_estimates(estimates):
    mean_est = np.mean(estimates)
    std_est = np.std(estimates, ddof=1)
    normalized = [(x - mean_est) / std_est for x in estimates]
    return normalized

def generate_qq_plot(normalized_data, coef_name, T):
    fig, ax = plt.subplots(figsize=(6, 6))
    sm.qqplot(np.array(normalized_data), line ='45', ax=ax)
    ax.set_title(f'QQ Plot de {coef_name} para T={T}')
    plt.savefig(f'{T}_qqplot_{coef_name}.png')
    plt.show()
    
def compare_models_by_criteria(models_by_T, T_target):
    
    results_aic = {'3_lags': 0, '5_lags': 0, '7_lags': 0}
    results_bic = {'3_lags': 0, '5_lags': 0, '7_lags': 0}
    
    models = models_by_T[T_target]
    for model_set in models:
        
        aic_values = {
            3: model_set['parsimonious_model'].aic, 
            5: model_set['true_model'].aic, 
            7: model_set['non_parsimonious_model'].aic
        }
        
        bic_values = {
            3: model_set['parsimonious_model'].bic, 
            5: model_set['true_model'].bic, 
            7: model_set['non_parsimonious_model'].bic
        }

        # Identificar o modelo com menor AIC e BIC
        best_aic = min(aic_values, key=aic_values.get)
        best_bic = min(bic_values, key=bic_values.get)

        # Incrementar contagem para o modelo com melhor AIC e BIC
        results_aic[f'{best_aic}_lags'] += 1
        results_bic[f'{best_bic}_lags'] += 1

    print(f"Melhor modelo por AIC e BIC para T = {T_target}:")
    print("Contagem de melhores modelos por AIC:", results_aic)
    print("Contagem de melhores modelos por BIC:", results_bic)


# Função principal
def main():
    coef_names_true = ['Intercept', 'lag1', 'lag2', 'lag3', 'lag4', 'lag5']
    true_values = {
        'Intercept': -2.8702,
        'lag1': 0.8484,
        'lag2': 0.5779,
        'lag3': 0.6372,
        'lag4': 0.7393,
        'lag5': 0.7402
    }
    
    # Simulando séries e aplicando modelos 
    # models_by_T carrega todos modelos verdadeiros para as analises de comparacao
    models_by_T = {}
    all_models = {}
    for T in T_sizes:
        simulated_all_series = simulate_multiple_series(B, [T], true_values)
        simulated_models = apply_models_formula(simulated_all_series)
        all_models[T] = simulated_models
        
        models_true = [result['true_model'] for result in simulated_models]
        stats = calculate_statistics(models_true, coef_names_true)
        display_statistics(true_values, T, stats)
        models_by_T[T] = models_true
        
    # Gerando boxplots para cada coeficiente em todos os Tamanhos de Série
    # So quero gerar boxplot quando tiver analisando todos os Tamanhos de Série
    if T_sizes != [4000, 16000]:
        for coef_name, true_value in true_values.items():
            plot_boxplots_by_coefficient(models_by_T, coef_name, true_value, T_sizes, "Boxplot das Estimativas")
        
    # Normalizando o coeficiente para um tamanho de série e verificando se é proximo de uma normal
    for T_target in [4000, 16000]:
        target_models = models_by_T[T_target]
        for coef_target in coef_names_true:
            target_estimates = [model.params.get(coef_target, None) for model in target_models]
            normalized_estimates = normalize_estimates(target_estimates)
            generate_qq_plot(normalized_estimates, coef_target, T_target)
    
        # Até então fizemos análises apenas dentro dos modelos verdadeiros, vamos agora cruzar os modelos.
        compare_models_by_criteria(all_models, T_target)
    
        non_significant_count = { 'lag6': 0, 'lag7': 0 }
        for model_set in all_models[T_target]:
            non_parsimonious_model = model_set['non_parsimonious_model']
            if 'lag6' in non_parsimonious_model.params and non_parsimonious_model.pvalues['lag6'] > 0.05:
                non_significant_count['lag6'] += 1
            if 'lag7' in non_parsimonious_model.params and non_parsimonious_model.pvalues['lag7'] > 0.05:
                non_significant_count['lag7'] += 1

        print(f"Para T = {T_target}:")
        print(f"Lag 6 não significativo em {non_significant_count['lag6']} de {len(all_models[T_target])} simulações.")
        print(f"Lag 7 não significativo em {non_significant_count['lag7']} de {len(all_models[T_target])} simulações.")
        print("")
        
if __name__ == "__main__":
    main()
