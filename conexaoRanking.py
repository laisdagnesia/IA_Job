# Consulta para descobrir todas as Ofertas coletadas pelo Scrapping da Carhunt

from pymongo import MongoClient
import pandas as pd
client = MongoClient('mongodb+srv://gabrielaraujon96:D3umaten0ve@carhunt.sweobl3.mongodb.net/')
db = client['carhunt']

#dados de ofertas:
base_v = pd.DataFrame(db['search_scraping'].find())

# Drop all rows that contain any NaN values
base_v = base_v.dropna(subset=['brand', 'city', 'km', 'model',
       'model_year', 'price', 'state'])


print(base_v)