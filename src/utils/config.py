import numpy as np

URLS = {
    'AAPL': 'https://frd001.s3-us-east-2.amazonaws.com/AAPL_1min_sample_firstratedata.zip',
}

retornos = {
        'log_return': lambda df: np.log(df['close']).diff(),
        'log_return_squared': lambda df: np.log(df['close']).diff() ** 2,
        # Para testar novos retornos tem que definir aqui o nome e como se calcular
    }


# URLS = {
#     'AAPL': 'https://frd001.s3-us-east-2.amazonaws.com/AAPL_1min_sample_firstratedata.zip',
#     'AMZN': 'https://frd001.s3-us-east-2.amazonaws.com/AMZN_1min_sample_firstratedata.zip'
# }
