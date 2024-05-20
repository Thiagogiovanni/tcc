from intradaypricepredictor import IntradayPricePredictor
from utils import config # Antes eu definia no config, porém deixando na main acho que fica mais fácil p ver o que quer usar
import numpy as np
from colorama import Fore, Style
import os
import itertools

def main():
    urls = URLS()
    retornos_definidos = definir_retornos()  # Renomeado para evitar conflito
    return_type = 'log_return'

    predictor = IntradayPricePredictor(urls, retornos_definidos, return_type)

    models_defasagens = {
        "modelo_simulacao": {
            "lagged_viol": 5
        },
        "aic_score": {
            "lagged_returns": 20,
            "lagged_viol": 20,
            "sq_lagged_returns": 20,
            "lagged_volume": 5
        },
        "bic_score": {
            "lagged_viol": 20,
            "lagged_volume": 5
        },
        "fn_test_using_train_threshold": {
            "sq_lagged_returns": 20,
            "lagged_volume": 5
        },
        "fp_test_using_train_threshold": {
            "lagged_returns": 45,
            "sq_lagged_returns": 5
        },
        "mean_fp_fn_test_using_train_threshold": {
            "lagged_viol": 60,
            "sq_lagged_returns": 5,
            "lagged_volume": 10
        },
        "auc_score" : {
            "lagged_returns": 60,
            "lagged_viol": 60,
            "sq_lagged_returns": 10,
            "lagged_volume": 30
        },
        "min_max_fn_fp" : {
            "lagged_returns": 30,
            "lagged_viol": 30,
            "lagged_volume": 20
        }
    }        
    
    
    model_to_run = 'modelo_simulacao'
    
    color = Fore.CYAN
    ticker = 'TSLA'
    
    predictor.load_and_prepare_data(ticker)
    
    for model in models_defasagens if model_to_run == 'all' else [model_to_run]:
        print(f"Modelo {model}")
    
        chosen_regressors = list(models_defasagens[model].keys())
        quantile_value = 0.05
        lags_per_regressor = models_defasagens[model]
        
        folder_id = f'./results/{ticker}_{return_type}_' + "_".join(f"{reg}{lags}" for reg, lags in lags_per_regressor.items())

        predictor.construct_data_matrix(ticker, chosen_regressors, lags_per_regressor, quantile_value)
        
        
        if not os.path.exists(folder_id):
            os.makedirs(folder_id)
            os.makedirs(folder_id + '/train')
            os.makedirs(folder_id + '/test')
            os.makedirs(folder_id + '/test_with_optimal_threshold_train')
            
        optimal_threshold_train, auc_score_train, optimal_threshold_test, auc_score_test, aic_score, bic_score, fnr_train, fn_test_using_train_threshold, fnr_test, fp_train, fp_using_train, fp_test = predictor.fit_and_evaluate_model(ticker, folder_id, train_start_date='2023-01-01', train_end_date='2023-01-31',test_end_date='2023-03-01', columns_to_drop = None, refine_model = False)
        
        print(color + f'-'*50)
        print(color + f'Ticker: {ticker}')
        print(color + f'Regressors: {chosen_regressors}')
        print(color + f'Return type: {return_type}')
        print(color + f'Lags: {lags_per_regressor}')
        print(color + f'Results:')
        print(color + f'Optimal threshold for train: {optimal_threshold_train:3f}, AUC score for train: {auc_score_train:3f}')
        print(color + f'Optimal threshold for test: {optimal_threshold_test:3f}, AUC score for test: {auc_score_test:3f}')
        print(color + f'AIC Score: {aic_score}')
        print(color + f'BIC Score: {bic_score}')
        print(color + f'FN Train: {fnr_train}')
        print(color + f'FP Train: {fp_train}')
        print(color + f'FN Test: {fnr_test}')
        print(color + f'FP Test: {fp_test}')
        print(color + f'FN Test using optimal threshold for train: {fn_test_using_train_threshold}')
        print(color + f'FP Test using optimal threshold for train: {fp_using_train}')
        print(color + f'Mean FN and FP test using optimal threshold for train: {(fn_test_using_train_threshold+fp_using_train)/2}')
        print(color + f'Mean FN and FP test: {(fnr_test+fp_test)/2}')
        print(Style.RESET_ALL + f'-'*50)

    # final_model = refine_model(predictor, ticker, train_start_date='2023-01-01', train_end_date='2023-01-31', test_end_date='2023-03-01')    
    # print(final_model)
    
    
def refine_model(predictor, ticker, train_start_date, train_end_date, test_end_date, significance_level=0.05):
    
    folder_id = f'./results/{ticker}_refined_model_5_percent_significance_level'
    
    if not os.path.exists(folder_id):
            os.makedirs(folder_id)
            os.makedirs(folder_id + '/train')
            os.makedirs(folder_id + '/test')
            os.makedirs(folder_id + '/test_with_optimal_threshold_train')
            
    model = predictor.fit_and_evaluate_model(ticker, folder_id, train_start_date, train_end_date, test_end_date, columns_to_drop = None, refine_model = True)
    all_non_significant_regressors = []
    color = Fore.CYAN
    
    while True:
        
        p_values = model.pvalues
        max_p_value = p_values.max()
        if max_p_value < significance_level:
            break

        non_significant_regressor = p_values.idxmax()
        all_non_significant_regressors.append(non_significant_regressor)
        model = predictor.fit_and_evaluate_model(ticker, folder_id, train_start_date, train_end_date, test_end_date, columns_to_drop = all_non_significant_regressors, refine_model = True)


    # Agora com o modelo refinado vamos avaliar o modelo
    optimal_threshold_train, auc_score_train, optimal_threshold_test, auc_score_test, aic_score, bic_score, fnr_train, fn_test_using_train_threshold, fnr_test, fp_train, fp_using_train, fp_test = predictor.fit_and_evaluate_model(ticker, folder_id, train_start_date='2023-01-01', train_end_date='2023-01-31',test_end_date='2023-03-01', columns_to_drop = all_non_significant_regressors, refine_model = False)
    
    print(color + f'-'*50)
    print(color + f'Ticker: {ticker}')
    print(color + f'Regressors: {model.pvalues.index.tolist}')
    print(color + f'Results:')
    print(color + f'Optimal threshold for train: {optimal_threshold_train:3f}, AUC score for train: {auc_score_train:3f}')
    print(color + f'Optimal threshold for test: {optimal_threshold_test:3f}, AUC score for test: {auc_score_test:3f}')
    print(color + f'AIC Score: {aic_score}')
    print(color + f'BIC Score: {bic_score}')
    print(color + f'FN Train: {fnr_train}')
    print(color + f'FP Train: {fp_train}')
    print(color + f'FN Test: {fnr_test}')
    print(color + f'FP Test: {fp_test}')
    print(color + f'FN Test using optimal threshold for train: {fn_test_using_train_threshold}')
    print(color + f'FP Test using optimal threshold for train: {fp_using_train}')
    print(color + f'Mean FN and FP test using optimal threshold for train: {(fn_test_using_train_threshold+fp_using_train)/2}')
    print(color + f'Mean FN and FP test: {(fnr_test+fp_test)/2}')
    print(Style.RESET_ALL + f'-'*50)
        
    final_regressors = model.pvalues.index.tolist()

    return final_regressors

def URLS():
    return {
        # 'AAPL': 'https://frd001.s3-us-east-2.amazonaws.com/AAPL_1min_sample_firstratedata.zip',
        # 'AMZN': 'https://frd001.s3-us-east-2.amazonaws.com/AMZN_1min_sample_firstratedata.zip',
        # 'MSFT': 'https://frd001.s3-us-east-2.amazonaws.com/MSFT_1min_sample_firstratedata.zip',
        # 'META': 'https://frd001.s3-us-east-2.amazonaws.com/META_1min_sample_firstratedata.zip',
        'TSLA': 'https://frd001.s3-us-east-2.amazonaws.com/TSLA_1min_sample_firstratedata.zip'
        
    }

def definir_retornos():  # Renomeado para evitar conflito
    return {
        'log_return': lambda df: np.log(df['close']).diff(),
        'log_return_squared': lambda df: np.log(df['close']).diff() ** 2
        # Para testar novos retornos tem que definir aqui o nome e como se calcular
    }

def all_regressors():
    return [
        'lagged_returns',
        'lagged_viol',
        'sq_lagged_returns',
        'lagged_volume'
        # Para adicionar aqui novos regressores tem que definir em "construct_data_matrix"
    ]
    
if __name__ == '__main__':
    main()