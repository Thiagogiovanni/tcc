import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import statsmodels.api as sm
import statsmodels.formula.api as smf
import requests
import os

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from io import BytesIO
from zipfile import ZipFile
import seaborn as sns

# Faz com que o BIC (Bayesian Information Criterion) seja usado para calcular o log-likelihood
from statsmodels.genmod.generalized_linear_model import SET_USE_BIC_LLF
SET_USE_BIC_LLF(True)

class IntradayPricePredictor:
    def __init__(self, urls, return_formulas, return_type):
        self.urls = urls
        self.data_frames = {}
        self.return_formulas = return_formulas  # Armazenando o dicionário de fórmulas de retorno
        self.return_type = return_type  # Tipo de retorno escolhido
        self.models = {}
        self.predictions = {}
        self.train_dates = {}  
        self.test_dates = {} 
        
    def load_and_prepare_data(self, ticker):
        
        url = self.urls[ticker]
        response = requests.get(url)
        zip_file = ZipFile(BytesIO(response.content))
        csv_filename = zip_file.namelist()[0]
        df = pd.read_csv(zip_file.open(csv_filename))
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values('timestamp', inplace=True)  
        
        df.set_index('timestamp', inplace=True)
        
        # Fixando o retorno para o log retorno
        #df['log_return'] = np.log(df['close']).diff()
        
        # Aplicando a forma de cálculo do retorno definida na main
        df[self.return_type] = self.return_formulas[self.return_type](df)
        
        df['date'] = df.index.date
        df['day_change'] = df['date'] != df['date'].shift(1)
        
        # filtered_returns = df[df['day_change'] == False][self.return_type]
        # self.data_frames[ticker] = filtered_returns
        
        df = df[df['day_change'] == False]
        self.data_frames[ticker] = df
        
    def construct_data_matrix(self, ticker, chosen_regressors,  lags_per_regressor, quantile_value = 0.05):
        """
        input: ticker, lags e quais defasagens quero usar
        output: dataframe com as defasagens e o retorno escolhidos
        
        Minha ideia aqui foi calcular todas métricas que podem ser uteis para criar as defasagens, por exemplo "viol", porém deixar a escolha de quais usar para o usuário.
        Não pensei em outro jeito melhor para filtrar os regressores diferentes de deixar aqui todos os possiveis regressores e apenas filtrar na hora de concatenar, porém funciona bem.
        """
        
        df = self.data_frames[ticker]
        extreme = df[self.return_type].quantile(quantile_value)
        viol = (df[self.return_type] < extreme).astype(int)
        
        all_regressors = {}
        for regressor, lags in lags_per_regressor.items():
            if regressor == 'lagged_returns':
                all_regressors[regressor] = pd.DataFrame({f'RET_LAG{lag}': df[self.return_type].shift(lag) for lag in range(1, lags + 1)})
            elif regressor == 'lagged_viol':
                all_regressors[regressor] = pd.DataFrame({f'VIOL_LAG{lag}': viol.shift(lag) for lag in range(1, lags + 1)})
            elif regressor == 'sq_lagged_returns':
                all_regressors[regressor] = pd.DataFrame({f'SQ_RET_LAG{lag}': df[self.return_type].shift(lag) ** 2 for lag in range(1, lags + 1)})
            elif regressor == 'lagged_volume':
                all_regressors[regressor] = pd.DataFrame({f'VOL_LAG{lag}': df['volume'].shift(lag) for lag in range(1, lags + 1)})

        X = pd.concat([all_regressors[name] for name in chosen_regressors if name in all_regressors], axis=1)
        X.index = df.index
        Y = viol.rename('Y').to_frame()
        
        data = pd.concat([Y, X], axis=1)
        self.data_frames[f'{ticker}_matrix'] = data
        
    def calculate_roc_auc(self, folder_id, true_values, predicted_probs, title, plot):
        """
        Calcula, salva e opcionalmente plota a curva ROC e calcula a AUC.
        """
        # fpr = false positive e tpr = true positive
        fpr, tpr, thresholds = roc_curve(true_values, predicted_probs)
        auc_score = roc_auc_score(true_values, predicted_probs)
        optimal_threshold = self.find_optimal_cutoff(thresholds, tpr, fpr)
        
        if not os.path.exists(f'{folder_id}/roc_curve.png'):
            # Salva a curva ROC
            plt.figure(figsize=(10, 6))
            plt.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
            plt.plot([0, 1], [0, 1], 'k--') # Linha de chance
            plt.xlabel('Especificidade (Taxa de Falsos Positivos)')
            plt.ylabel('Sensibilidade (Taxa de Verdadeiros Positivos)')
            plt.title(title)
            plt.legend(loc='lower right')
            
            # Coloca o ponto para corte ótimo
            optimal_idx = np.argmax(tpr - fpr)
            plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', label=f'Ponto de corte ótimo = {optimal_threshold:.3f}')
            plt.legend(loc='lower right')
            
            plt.savefig(f'{folder_id}/roc_curve.png')
            plt.close()
            
            if plot:
                plt.show()
        
        if not os.path.exists(f'{folder_id}/thresholds.png'):
            # Salva gráfico de Sensibilidade e Especificidade vs. Pontos de corte
            plt.figure(figsize=(10, 6))
            plt.plot(thresholds, tpr, label='Sensibilidade', color='blue')
            plt.plot(thresholds, 1-fpr, label='Especificidade', color='green')
            plt.xlim([0.0, 1.0])
            plt.xlabel('Pontos de corte')
            plt.ylabel('Taxa')
            plt.title('Sensibilidade e Especificidade vs. Pontos de corte')
            plt.legend(loc='best')
            plt.axvline(x=optimal_threshold, color='red', linestyle='--', label=f'Ponto de corte ótimo = {optimal_threshold:.3f}')
            plt.legend(loc='best')
            
            plt.savefig(f'{folder_id}/thresholds.png')
            plt.close()    
            
            if plot:
                plt.show()
            
        return auc_score, fpr, tpr, thresholds, optimal_threshold

    def find_optimal_cutoff(self, thresholds, tpr, fpr):
        """
        Encontra o ponto de corte ótimo baseado no método de Youden.
        'the optimal classification threshold is defined as the cut point with the maximum difference between the TPR and FPR'
        """
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        return optimal_threshold

    def create_confusion_matrix(self, folder_id, true_values, predicted_probs, threshold, data_label, plot):
  
        # Convertendo probabilidades em previsões binárias com base no threshold
        predicted_labels = (predicted_probs >= threshold).astype(int)
        cm = confusion_matrix(true_values, predicted_labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        if not os.path.exists(f'{folder_id}/confusion_matrix.png'):
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(cm_normalized, annot=True, fmt=".3f", cmap='Blues', cbar=False,
                        xticklabels=['SEM QUEDA', 'QUEDA'], yticklabels=['NÃO OCORRE', 'OCORRE'])
            plt.ylabel('QUEDA')
            plt.xlabel('PREVISÃO')
            plt.title(F"Matriz de Confusão ({data_label})")
            
            plt.savefig(f'{folder_id}/confusion_matrix.png')
            plt.close()
            if plot:
                plt.show()

    def evaluate_predictions(self, folder_id, true_values, predicted_probs, data_label, plot_roc, plot_cm, use_train_threshold=False, train_threshold=None):
        """
        Avalia as previsões com curva ROC, AUC e matriz de confusão.
        Se use_train_threshold for True, utiliza o threshold de treino (train_threshold)
        para avaliação em vez de encontrar um novo threshold ótimo.
        """
        if use_train_threshold and train_threshold is not None:
            optimal_threshold = train_threshold
            auc_score = roc_auc_score(true_values, predicted_probs)
        else:
            auc_score, fpr, tpr, thresholds, optimal_threshold = self.calculate_roc_auc(folder_id, true_values, predicted_probs, plot=plot_roc, title=f"ROC Curve ({data_label})")
        
        # Casos que foram quedas e prevemos errado
        predicted_labels = (predicted_probs >= (train_threshold if use_train_threshold else optimal_threshold)).astype(int)
        cm = confusion_matrix(true_values, predicted_labels)
        TN, FP, FN, TP = cm.ravel()
        fnr = FN / float(FN + TP) if (FN + TP) > 0 else 0  # Casos que foram quedas e prevemos errado
        fp = FP / float(FP + TN) if (FP + TN) > 0 else 0  # Casos que não foram quedas e prevemos errado
        
        self.create_confusion_matrix(folder_id, true_values, predicted_probs, optimal_threshold, data_label, plot_cm)

        # Se estamos usando o threshold de treino para avaliação, não precisamos recalcular o AUC para a curva ROC específica,
        # mas ainda calculamos para registro e possível comparação.
        if not use_train_threshold:
            auc_score, fpr, tpr, thresholds, optimal_threshold = self.calculate_roc_auc(folder_id, true_values, predicted_probs, title=f"ROC Curve ({data_label})", plot=plot_roc)

        return optimal_threshold, auc_score, fnr, fp

    def plot_roc_train_and_test(self, folder_id, train_true_values, train_predicted_probs, test_true_values, test_predicted_probs):
        """
        Plota as curvas ROC de treino e teste no mesmo gráfico.
        """
        
        train_fpr, train_tpr, train_thresholds = roc_curve(train_true_values, train_predicted_probs)
        test_fpr, test_tpr, test_thresholds = roc_curve(test_true_values, test_predicted_probs)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_fpr, train_tpr, label=f'Treino')
        plt.plot(test_fpr, test_tpr, label=f'Teste')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('Especificidade (Taxa de Falsos Positivos)')
        plt.ylabel('Sensibilidade (Taxa de Verdadeiros Positivos)')
        plt.title('Curva ROC')
        plt.legend(loc='lower right')
        
        plt.savefig(f'{folder_id}/roc_curve_train_and_test.png')
        plt.close()
        
    def plot_adjusted_probabilities_histogram(self, folder_id, test_predicted_probs, optimal_threshold_train):
        """
        Gera histogramas das probabilidades ajustadas para a base de TESTE,
        indicando o limiar crítico da base de TREINO como uma linha vertical.
        
        Parâmetros:
        - test_predicted_probs: Probabilidades preditas para a base de teste.
        - optimal_threshold_train: Limiar ótimo encontrado na base de treino.
        """
        plt.figure(figsize=(10, 6))
        
        # Histograma das probabilidades ajustadas
        plt.hist(test_predicted_probs, bins=50, alpha=0.5, label='Probabilidades Ajustadas (Teste)')
        
        # Linha vertical para o limiar crítico da base de treino
        plt.axvline(x=optimal_threshold_train, color='r', linestyle='--', label=f'Limiar Crítico de Treino: {optimal_threshold_train:.4f}')
        
        plt.xlabel('Probabilidades Ajustadas')
        plt.ylabel('Frequência')
        plt.title('Histograma das Probabilidades Ajustadas (Base de Teste)')
        plt.legend()
        
        plt.savefig(f'{folder_id}/histogram_adjusted_probabilities.png')
        plt.close()

    def plot_probabilities_histograms_by_outcome(self, folder_id, test_data, test_predicted_probs, optimal_threshold_train):
        
        """
        Gera dois histogramas das probabilidades ajustadas para a base de TESTE,
        separados pelo resultado real (queda ocorre vs. queda não ocorre),
        com uma linha vertical indicando o limiar crítico da base de TREINO.
        
        Parâmetros:
        - test_data: DataFrame com os dados de teste incluindo a coluna 'Y' para o resultado real.
        - test_predicted_probs: Probabilidades preditas para a base de teste.
        - optimal_threshold_train: Limiar ótimo encontrado na base de treino.
        """
        
        probs_when_drop_occurs = test_predicted_probs[test_data['Y'] == 1]
        probs_when_drop_does_not_occur = test_predicted_probs[test_data['Y'] == 0]
        
        # Define o espaço do plot
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        
        # Histograma das probabilidades onde queda ocorre
        axs[0].hist(probs_when_drop_occurs, bins=50, alpha=0.5, color='red')
        axs[0].axvline(x=optimal_threshold_train, color='black', linestyle='--')
        axs[0].set_title('QUEDA OCORRE (BASE TESTE)')
        axs[0].set_xlabel('Probabilidades Ajustadas')
        axs[0].set_ylabel('Frequência')
        
        # Histograma das probabilidades onde queda não ocorre
        axs[1].hist(probs_when_drop_does_not_occur, bins=50, alpha=0.5, color='blue')
        axs[1].axvline(x=optimal_threshold_train, color='black', linestyle='--')
        axs[1].set_title('QUEDA NÃO OCORRE (BASE TESTE)')
        axs[1].set_xlabel('Probabilidades Ajustadas')
        axs[1].set_ylabel('Frequência')
        
        # Ajustes finais
        plt.tight_layout()
        
        # Salva o gráfico
        plt.savefig(f'{folder_id}/histograms_by_outcome.png')
        plt.close()
        
    def sum_of_squares_metrics(self, y_true, predicted_probs, num_predictors):
        """
        Calculate r^2, R_ss^2 and R^2_ss_adj for the model.
        """
        # Mean of the observed data
        y_bar = np.mean(y_true)
        
        # Predicted probabilities
        y_pred = predicted_probs
        
        # Total Sum of Squares (SST)
        sst = np.sum((y_true - y_bar)**2)
        
        # Residual Sum of Squares (SSR)
        ssr = np.sum((y_pred - y_bar)**2)
        
        # Sum of Squared Errors (SSE)
        sse = np.sum((y_true - y_pred)**2)
        
        # Pearson's correlation squared (r^2)
        r_squared = (np.corrcoef(y_true, y_pred)[0, 1])**2
        
        # Coefficient of Determination (R^2)
        R_ss_squared = 1 - sse / sst
        
        # Adjusted Coefficient of Determination (R^2 adjusted)
        n = len(y_true)  # Number of observations
        p = num_predictors  # Number of predictors
        R_ss_squared_adj = 1 - (sse / (n - p - 1)) / (sst / (n - 1))
        
        return r_squared, R_ss_squared, R_ss_squared_adj
    
    def plot_predictions_with_errors(self, ticker, folder_id, test_data, predicted_probs, threshold):
        """
        Plota a evolução dos retornos e das probabilidades ajustadas com destaque para os erros de classificação.

        Parâmetros:
            test_data (DataFrame): DataFrame contendo os dados de teste, incluindo 'Y' para resultados reais e 'return' para os valores de retorno.
            predicted_probs (array): Array de probabilidades ajustadas previstas pelo modelo.
            threshold (float): Limiar utilizado para a classificação binária.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        data_return = self.data_frames[ticker]['log_return']
        dates_test = self.test_dates[ticker]
        
        data_return = data_return[(data_return.index > dates_test[0]) & (data_return.index <= dates_test[1])]
        
        # Criando a figura e o eixo
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plotando os retornos como uma linha
        ax.plot(test_data.index, data_return, label='Retornos', color='grey', alpha=0.7)

        # Adicionando as barras verticais para probabilidades ajustadas
        ax.vlines(test_data.index, 0, predicted_probs, color='black', alpha=0.5, label='Probabilidades Ajustadas')

        # Encontrando e destacando as classificações erradas
        for idx, (true, prob) in enumerate(zip(test_data['Y'], predicted_probs)):
            if (true == 1 and prob < threshold) or (true == 0 and prob >= threshold):
                ax.vlines(test_data.index[idx], 0, prob, color='red', alpha=0.7)

        # Configurando o título e os rótulos
        ax.set_title('Evolução dos Retornos e Probabilidades Ajustadas com Erros de Classificação')
        ax.set_xlabel('Data')
        ax.set_ylabel('Probabilidades Ajustadas / Retornos')

        # Adicionando a legenda
        ax.legend()

        # Mostrando o gráfico
        plt.tight_layout()
        plt.savefig(f'{folder_id}/predictions_with_errors.png')

    def fit_and_evaluate_model(self, ticker, folder_id, train_start_date, train_end_date, test_end_date, columns_to_drop = None, refine_model = False):
        
        data = self.data_frames[f'{ticker}_matrix']
        
        # Armazeno para usar depois
        self.train_dates[ticker] = (pd.to_datetime(train_start_date), pd.to_datetime(train_end_date))
        self.test_dates[ticker] = (pd.to_datetime(train_end_date), pd.to_datetime(test_end_date))

        train_data = data[(data.index >= train_start_date) & (data.index <= train_end_date)]
        test_data = data[(data.index > train_end_date) & (data.index <= test_end_date)]
        
        if columns_to_drop is not None:
            train_data = train_data.drop(columns=columns_to_drop)
            test_data = test_data.drop(columns=columns_to_drop)
            
        predictors = train_data.columns.difference(['Y'])
        formula = 'Y ~ ' + ' + '.join(predictors)
        model = smf.glm(formula=formula, data=train_data, family=sm.families.Binomial()).fit()
        
        if refine_model:
            return model
        
        aic_score = model.aic
        bic_score = model.bic
    
        summary_str = model.summary().as_text()
        with open(f'{folder_id}/model_summary.txt', 'w') as text_file:
            text_file.write(summary_str)

        self.models[ticker] = model
        train_predictions = model.predict(train_data.drop(columns=['Y']))
        test_predictions = model.predict(test_data.drop(columns=['Y']))

        # Avaliação com dados de treino
        optimal_threshold_train, auc_score_train, fnr_train, fp_train = self.evaluate_predictions(
            folder_id + '/train',
            train_data['Y'], 
            train_predictions, 
            plot_roc=False, 
            plot_cm=False, 
            data_label="Treino"
        )

        # Avaliação com dados de teste usando threshold de treino
        _, auc, fnr_using_train, fp_using_train = self.evaluate_predictions(
            folder_id + '/test_with_optimal_threshold_train',
            test_data['Y'], 
            test_predictions, 
            plot_roc=False, 
            plot_cm=False, 
            data_label="Teste com Threshold de Treino",
            use_train_threshold=True,  
            train_threshold=optimal_threshold_train  # Passaremos o threshold de treino
        )

        # Avaliação padrão com dados de teste
        optimal_threshold_test, auc_score_test, fnr_test, fp_test = self.evaluate_predictions(
            folder_id + '/test',
            test_data['Y'], 
            test_predictions, 
            plot_roc=False, 
            plot_cm=False, 
            data_label="Teste"
        )
        
        self.plot_roc_train_and_test(folder_id, train_data['Y'], train_predictions, test_data['Y'], test_predictions)
        
        self.plot_adjusted_probabilities_histogram(folder_id, test_predictions, optimal_threshold_train)
        
        if columns_to_drop is not None and refine_model == False:  
            self.plot_probabilities_histograms_by_outcome(folder_id, test_data, test_predictions, optimal_threshold_train)
            
            # Calculando e registrando as métricas de soma dos quadrados
            r_squared, R_ss_squared, R_ss_squared_adj = self.sum_of_squares_metrics(
                train_data['Y'], 
                train_predictions, 
                len(predictors)
            )
            print(f"r^2 (Pearson correlation squared): {r_squared}")
            print(f"R_ss^2 (coefficient of determination): {R_ss_squared}")
            print(f"R^2_ss_adj (adjusted coefficient of determination): {R_ss_squared_adj}")
            
            self.plot_predictions_with_errors(ticker, folder_id, test_data, test_predictions, optimal_threshold_train)

        return optimal_threshold_train, auc_score_train, optimal_threshold_test, auc_score_test, aic_score, bic_score, fnr_train, fnr_using_train, fnr_test, fp_train, fp_using_train, fp_test


    # def fit_and_evaluate_model(self, ticker, folder_id, train_start_date, train_end_date, test_end_date):
    #     data = self.data_frames[f'{ticker}_matrix']
        
    #     # Armazeno para usar depois
    #     self.train_dates[ticker] = (pd.to_datetime(train_start_date), pd.to_datetime(train_end_date))
    #     self.test_dates[ticker] = (pd.to_datetime(train_end_date) + pd.Timedelta(days=1), pd.to_datetime(test_end_date))
    
    #     train_data = data[(data.index >= train_start_date) & (data.index <= train_end_date)]
    #     test_data = data[(data.index > train_end_date) & (data.index <= test_end_date)]
        
    #     predictors = train_data.columns.difference(['Y'])
    #     formula = 'Y ~ ' + ' + '.join(predictors)
    #     model = smf.glm(formula=formula, data=train_data, family=sm.families.Binomial()).fit()
        
    #     summary_str = model.summary().as_text()
    #     with open(f'{folder_id}/model_summary.txt', 'w') as text_file:
    #         text_file.write(summary_str)
    
        
    #     self.models[ticker] = model
    #     self.predictions[ticker] = {
    #         'train': model.predict(train_data.drop(columns=['Y'])),
    #         'test': model.predict(test_data.drop(columns=['Y']))
    #     }

    #     # print("Training Data Evaluation:")
    #     optimal_threshold_train, auc_score_train = self.evaluate_predictions(
    #         folder_id + '/train',
    #         train_data['Y'], 
    #         self.predictions[ticker]['train'], 
    #         plot_roc=False, # Caso queira ver na tela assim que rodar o gráfico basta deixar True
    #         plot_cm=False, # Caso queira ver na tela assim que rodar o gráfico basta deixar True
    #         data_label="Treino"
    #     )

    #     # Avaliação com dados de teste
    #     # print("Test Data Evaluation:")
    #     optimal_threshold_test, auc_score_test = self.evaluate_predictions(
    #         folder_id + '/test',
    #         test_data['Y'], 
    #         self.predictions[ticker]['test'], 
    #         plot_roc=False, # Caso queira ver na tela assim que rodar o gráfico basta deixar True
    #         plot_cm=False, # Caso queira ver na tela assim que rodar o gráfico basta deixar True
    #         data_label="Teste"
    #     )
        
    #     return optimal_threshold_train, auc_score_train, optimal_threshold_test, auc_score_test