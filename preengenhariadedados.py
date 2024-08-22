import datetime
from conexaoRanking import base_v
# Clone da base original
coerced_base_v = base_v.copy()

# Preparação dos dados das ofertas para ML


# Colunas que PRECISAM ser numéricas
num_cols = ['km', 'model_year', 'price']

# casting das colunas que devem ser numéricas por natureza
for c in num_cols:
    coerced_base_v[c] = [float(x) for x in coerced_base_v[c]]

# Criando coluna de age (anos desde a fabricação) que diminui a dimensionalidade dos dados
coerced_base_v['age'] = datetime.datetime.now(
).year-coerced_base_v['model_year'].astype(int)

# Dividir a coluna 'model' em duas colunas: 'model' e 'type'
coerced_base_v[['model', 'type']] = coerced_base_v['model'].str.split(
    pat=' ', n=1, expand=True)

# Adicionar uma nova coluna condicional
coerced_base_v['transmission'] = coerced_base_v['type'].apply(
    lambda x: 'Aut.' if 'aut' in x.lower() else 'Manual')

# Reajustar coluna de Type, removendo transmissão
coerced_base_v['type'] = coerced_base_v['type'].apply(lambda x: ' '.join(
    [word for word in x.split() if (('aut' not in word.lower()) and ('manu' not in word.lower()))]))

# # Tipos de carrocerias de carros, para categorizar
# # https://www.icarros.com.br/catalogo/compare.jsp
# tipos_normalizados = ['Hatch (compacto)','Sedan','Picape','SUV (Utilitário esportivo)','Monovolume','Perua/Wagon (SW)','Van','Conversível (coupé)','Hibrido/Elétrico']

# norm_body_types = { 'Conversível': 'Conversível (coupé)',
#   'Coupé': 'Conversível (coupé)',
#   'Conversível (coupé)':'Conversível (coupé)',
#   'Coupê': 'Conversível (coupé)',
#   'Cupê': 'Conversível (coupé)',
#   'Hatch': 'Hatch (compacto)',
#   'Hatchback': 'Hatch (compacto)',
#   'Hatch (compacto)':'Hatch (compacto)',
#   'Compacto': 'Hatch (compacto)',
#   'Minivan': 'Van',
#   'Van/Utilitário': 'Van',
#   'Van':'Van',
#   'Furgão & Van': 'Van',
#   'Monovolume': 'Monovolume',
#   'Perua/SW': 'Perua/Wagon (SW)',
#   'SW & Perua': 'Perua/Wagon (SW)',
#   'Perua/Wagon (SW)':'Perua/Wagon (SW)',
#   'Picape': 'Picape',
#   'Picapes': 'Picape',
#   'Sedan': 'Sedan',
#   'Sedã': 'Sedan',
#   'Utilitário esportivo': 'SUV (Utilitário esportivo)',
#   'SUV (Utilitário esportivo)': 'SUV (Utilitário esportivo)',
#   'SUV': 'SUV (Utilitário esportivo)',
#   'SUV/Utilitário': 'SUV (Utilitário esportivo)',
#   'Hibrido':'Hibrido/Elétrico',
#   'Elétrico':'Hibrido/Elétrico',
#   'Hibrido/Elétrico':'Hibrido/Elétrico',
#   }

# # Fazendo a tradução na base:
# coerced_base_v['body'] = coerced_base_v['body'].map(norm_body_types)

# Setando todos os Body Types como SUv (ATE QUE O GABRIEL ADICIONE A COLUNA BODY)
coerced_base_v["body"] = 'SUV'

print(coerced_base_v)
