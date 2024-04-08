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
    
    best_results = []
    results = []
    predictor = IntradayPricePredictor(urls, retornos_definidos, return_type)
    lags_values = [5, 10, 20, 30, 45, 60]
    
    colors = [Fore.CYAN, Fore.GREEN, Fore.MAGENTA, Fore.YELLOW, Fore.BLUE, Fore.RED]
    permutacoes = []
    for idx, ticker in enumerate(urls.keys()):
        predictor.load_and_prepare_data(ticker)
        for i in range(2, len(all_regressors())+1):
            permutacoes = [list(p) for p in list(itertools.combinations(all_regressors(), i))]
            for permutation in permutacoes:
                 for lags_combination in itertools.product(lags_values, repeat=len(permutation)):
                    try:
                        chosen_regressors = permutation
                        
                        quantile_value = 0.05
                        lags_per_regressor = dict(zip(permutation, lags_combination))
                        
                        color = colors[idx % len(colors)] 
                        folder_id = f'./results/{ticker}_{return_type}_' + "_".join(f"{reg}{lags}" for reg, lags in lags_per_regressor.items())

                        predictor.construct_data_matrix(ticker, chosen_regressors, lags_per_regressor, quantile_value)
                        
                        
                        if not os.path.exists(folder_id):
                            os.makedirs(folder_id)
                            os.makedirs(folder_id + '/train')
                            os.makedirs(folder_id + '/test')
                            os.makedirs(folder_id + '/test_with_optimal_threshold_train')
                            
                        optimal_threshold_train, auc_score_train, optimal_threshold_test, auc_score_test, aic_score, bic_score, fnr_train, fn_test_using_train_threshold, fnr_test, fp_train, fp_using_train, fp_test = predictor.fit_and_evaluate_model(ticker, folder_id, train_start_date='2023-01-01', train_end_date='2023-01-31',test_end_date='2023-03-01')
                        
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
                        
                        results.append(
                            {
                                'folder_id': folder_id,
                                'ticker': ticker,
                                'regressors': chosen_regressors,
                                'lags': lags_per_regressor,
                                'optimal_threshold_train': optimal_threshold_train,
                                'auc_score_train': auc_score_train,                            
                                'auc_score_test': auc_score_test,
                                'auc_score_mean': (auc_score_train+auc_score_test)/2,
                                'optimal_threshold_test': optimal_threshold_test,
                                'aic_score': aic_score,
                                'bic_score': bic_score,
                                'fn_test_using_train_threshold': fn_test_using_train_threshold,
                                'fp_test_using_train_threshold': fp_using_train,
                                'mean_fp_fn_test_using_train_threshold': (fn_test_using_train_threshold+fp_using_train)/2,
                                'max_fn_and_fp_test_using_train_threshold': max(fn_test_using_train_threshold, fp_using_train),
                            }
                        )
                    except Exception as e:
                        print(f'Erro: {e}')
                        continue
                    
        best_models = filter_best_models(results)
    
        best_results.append({
            'ticker': ticker,
            'results': best_models
        })
    
    for best_result in best_results:
        ticker = best_result['ticker']
        best_models = best_result['results']
        for criteria, model in best_models.items():
            print(f'Melhores modelos para {ticker}')
            print(f"Melhor modelo por {criteria}: {model['folder_id']} com {criteria} de {model[criteria]}")

def filter_best_models(results):
    
    best_auc_model = max(results, key=lambda x: (x['auc_score_mean']))
    best_aic_model = min(results, key=lambda x: x['aic_score'])
    best_bic_model = min(results, key=lambda x: x['bic_score'])
    best_fn_model = min(results, key=lambda x: x['fn_test_using_train_threshold'])
    best_fp_mdoel = min(results, key=lambda x: x['fp_test_using_train_threshold'])
    best_mean_fp_fn_model = min(results, key=lambda x: x['mean_fp_fn_test_using_train_threshold'])
    best_max_max_fn_and_fp_test_using_train_threshold = min(results, key=lambda x: x['max_fn_and_fp_test_using_train_threshold'])

    return {
        'auc_score_mean': best_auc_model,
        'aic_score': best_aic_model,
        'bic_score': best_bic_model,
        'fn_test_using_train_threshold': best_fn_model,
        'fp_test_using_train_threshold': best_fp_mdoel,
        'mean_fp_fn_test_using_train_threshold': best_mean_fp_fn_model,
        'max_fn_and_fp_test_using_train_threshold': best_max_max_fn_and_fp_test_using_train_threshold
    }

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