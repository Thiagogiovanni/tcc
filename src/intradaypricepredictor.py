import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import statsmodels.api as sm
import statsmodels.formula.api as smf
import requests
import matplotlib.pyplot as plt
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
        
    def construct_data_matrix(self, ticker, chosen_regressors, lags=60):
        """
        input: ticker, lags e quais defasagens quero usar
        output: dataframe com as defasagens e o retorno escolhidos
        
        Minha ideia aqui foi calcular todas métricas que podem ser uteis para criar as defasagens, por exemplo "viol", porém deixar a escolha de quais usar para o usuário.
        Não pensei em outro jeito melhor para filtrar os regressores diferentes de deixar aqui todos os possiveis regressores e apenas filtrar na hora de concatenar, porém funciona bem.
        """
        
        df = self.data_frames[ticker]
        extreme = df[self.return_type].quantile(0.05)
        viol = (df[self.return_type] < extreme).astype(int)
        
        all_regressors = {
            'lagged_returns': pd.DataFrame({f'RET_LAG{lag}': df[self.return_type].shift(lag) for lag in range(1, lags + 1)}),
            'lagged_viol': pd.DataFrame({f'VIOL_LAG{lag}': viol.shift(lag) for lag in range(1, lags + 1)}),
            'sq_lagged_returns': pd.DataFrame({f'SQ_RET_LAG{lag}': df[self.return_type].shift(lag) ** 2 for lag in range(1, lags + 1)}),
            'lagged_volume': pd.DataFrame({f'VOL_LAG{lag}': df['volume'].shift(lag) for lag in range(1, lags + 1)})
        }

        # Apos calcular todos possiveis regressores eu filtro aqui qual quero usar de fato (talvez mudar isso no futuro p/ n precisar calcular tudo, mas n pensei numa forma melhor)
        X = pd.concat([all_regressors[name] for name in chosen_regressors if name in all_regressors], axis=1)

        X.index = df.index 
        Y = viol.rename('Y').to_frame()  
        
        # Devo dropar na aqui?
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
        """
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        return optimal_threshold

    def create_confusion_matrix(self, folder_id, true_values, predicted_probs, threshold, data_label, plot):
  
        # Convertendo probabilidades em previsões binárias com base no threshold
        predicted_labels = (predicted_probs >= threshold).astype(int)
        cm = confusion_matrix(true_values, predicted_labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
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
  

    def evaluate_predictions(self, folder_id, true_values, predicted_probs, data_label, plot_roc, plot_cm):
        """
        Avalia as previsões com curva ROC, AUC e matriz de confusão.
        """
        auc_score, fpr, tpr, thresholds, optimal_threshold = self.calculate_roc_auc(folder_id, true_values, predicted_probs, plot=plot_roc, title=f"ROC Curve ({data_label})")
        self.create_confusion_matrix(folder_id, true_values, predicted_probs, optimal_threshold, data_label, plot_cm)
        
        # print(f"Optimal Threshold: {optimal_threshold}")
        return optimal_threshold, auc_score

    def fit_and_evaluate_model(self, ticker, folder_id, train_start_date, train_end_date, test_end_date):
        data = self.data_frames[f'{ticker}_matrix']
        
        # Armazeno para usar depois
        self.train_dates[ticker] = (pd.to_datetime(train_start_date), pd.to_datetime(train_end_date))
        self.test_dates[ticker] = (pd.to_datetime(train_end_date) + pd.Timedelta(days=1), pd.to_datetime(test_end_date))
    
        train_data = data[(data.index >= train_start_date) & (data.index <= train_end_date)]
        test_data = data[(data.index > train_end_date) & (data.index <= test_end_date)]
        
        predictors = train_data.columns.difference(['Y'])
        formula = 'Y ~ ' + ' + '.join(predictors)
        model = smf.glm(formula=formula, data=train_data, family=sm.families.Binomial()).fit()
        
        summary_str = model.summary().as_text()
        with open(f'{folder_id}/model_summary.txt', 'w') as text_file:
            text_file.write(summary_str)
    
        
        self.models[ticker] = model
        self.predictions[ticker] = {
            'train': model.predict(train_data.drop(columns=['Y'])),
            'test': model.predict(test_data.drop(columns=['Y']))
        }

        # print("Training Data Evaluation:")
        optimal_threshold_train, auc_score_train = self.evaluate_predictions(
            folder_id + '/train',
            train_data['Y'], 
            self.predictions[ticker]['train'], 
            plot_roc=False, # Caso queira ver na tela assim que rodar o gráfico basta deixar True
            plot_cm=False, # Caso queira ver na tela assim que rodar o gráfico basta deixar True
            data_label="Treino"
        )

        # Avaliação com dados de teste
        # print("Test Data Evaluation:")
        optimal_threshold_test, auc_score_test = self.evaluate_predictions(
            folder_id + '/test',
            test_data['Y'], 
            self.predictions[ticker]['test'], 
            plot_roc=False, # Caso queira ver na tela assim que rodar o gráfico basta deixar True
            plot_cm=False, # Caso queira ver na tela assim que rodar o gráfico basta deixar True
            data_label="Teste"
        )
        
        return optimal_threshold_train, auc_score_train, optimal_threshold_test, auc_score_test