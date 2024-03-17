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
    
    colors = [Fore.CYAN, Fore.GREEN, Fore.MAGENTA, Fore.YELLOW, Fore.BLUE, Fore.RED]
    permutacoes = []
    for idx, ticker in enumerate(urls.keys()):
        for i in range(2, len(all_regressors())+1):
            permutacoes = [list(p) for p in list(itertools.combinations(all_regressors(), i))]
            for permutation in permutacoes:
                for lags in [30, 60, 90, 120]:
                    
                    chosen_regressors = permutation
                        
                    color = colors[idx % len(colors)] 
                    
                    folder_id = f'./results/{ticker}_{return_type}_lags{lags}_' + "_".join(chosen_regressors)
                    predictor.load_and_prepare_data(ticker)
                    predictor.construct_data_matrix(ticker, chosen_regressors, lags)
                    
                    if not os.path.exists(folder_id):
                        os.makedirs(folder_id)
                        os.makedirs(folder_id + '/train')
                        os.makedirs(folder_id + '/test')
                        
                    optimal_threshold_train, auc_score_train, optimal_threshold_test, auc_score_test = predictor.fit_and_evaluate_model(ticker, folder_id, train_start_date='2023-01-01', train_end_date='2023-01-31',test_end_date='2023-03-01')
                
                    print(color + f'-'*50)
                    print(color + f'Ticker: {ticker}')
                    print(color + f'Regressors: {chosen_regressors}')
                    print(color + f'Return type: {return_type}')
                    print(color + f'Lags: {lags}')
                    print(color + f'Results:')
                    print(color + f'Optimal threshold for train: {optimal_threshold_train:3f}, AUC score for train: {auc_score_train:3f}')
                    print(color + f'Optimal threshold for test: {optimal_threshold_test:3f}, AUC score for test: {auc_score_test:3f}')
                    print(Style.RESET_ALL + f'-'*50)

def URLS():
    return {
        'AAPL': 'https://frd001.s3-us-east-2.amazonaws.com/AAPL_1min_sample_firstratedata.zip',
        'AMZN': 'https://frd001.s3-us-east-2.amazonaws.com/AMZN_1min_sample_firstratedata.zip'
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