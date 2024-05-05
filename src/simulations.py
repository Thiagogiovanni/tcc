import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Configurações de simulação
B = 200
T_sizes = [2000, 4000, 8000, 16000, 32000]
seed = 1
np.random.seed(seed)

# Coeficientes do modelo
beta_base = {
    'Intercept': -3.4108,
    'VIOL_LAG1': 0.4040,
    'VIOL_LAG3': 0.1967,
    'VIOL_LAG4': 0.3123,
    'VIOL_LAG5': 0.3124,
    'VIOL_LAG6': 0.3276
}

# Função para calcular π_t
def calculate_pi(y_lags, beta):
    eta = beta['Intercept']
    coef_keys = ['VIOL_LAG1', 'VIOL_LAG3', 'VIOL_LAG4', 'VIOL_LAG5', 'VIOL_LAG6']
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
def display_statistics(T, stats):
    print(f"Estatísticas para T = {T}")
    for coef, values in stats.items():
        print(f"{coef}: Média = {values['mean']:.4f}, Variância = {values['variance']:.4f}")
    print()

# Criar boxplots para cada coeficiente
def plot_boxplots(models, coef_names, true_values, title):
    coef_dict = extract_coefficients(models, coef_names)
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.Set3.colors  # Conjunto de cores pré-definido
    
    for i, (coef, color) in enumerate(zip(coef_names, colors)):
        bp = ax.boxplot(coef_dict[coef], positions=[i+1], widths=0.6, patch_artist=True, boxprops=dict(facecolor=color))
        ax.axhline(y=true_values[coef], color=color, linestyle='--', label=f'{coef} (Valor Verdadeiro)')
    
    ax.set_title(title)
    ax.set_xlabel('Coeficiente')
    ax.set_ylabel('Estimativa')
    ax.legend()
    plt.show()

# Função principal
def main():
    coef_names_true = ['Intercept', 'lag1', 'lag2', 'lag3', 'lag4', 'lag5']
    true_values = {
        'Intercept': -3.4108,
        'lag1': 0.4040,
        'lag2': 0,
        'lag3': 0.1967,
        'lag4': 0.3123,
        'lag5': 0.3124
    }

    for T in T_sizes:
        simulated_all_series = simulate_multiple_series(B, [T], beta_base)
        simulated_models = apply_models_formula(simulated_all_series)

        models_true = [result['true_model'] for result in simulated_models]
        stats = calculate_statistics(models_true, coef_names_true)
        display_statistics(T, stats)
        plot_boxplots(models_true, coef_names_true, true_values, f"Boxplot das Estimativas - T = {T}")

if __name__ == "__main__":
    main()
