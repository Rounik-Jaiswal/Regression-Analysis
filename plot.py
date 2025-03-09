import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn_evaluation import plot
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

df_nb = pd.read_csv('python\probability\laptop_price (1).csv', encoding='latin1') 

#------------------------------------DATA CLEANING START------------------------------------

df_nb['ips'] = df_nb.ScreenResolution.str.contains('ips', case=False)
df_nb['touchscreen'] = df_nb.ScreenResolution.str.contains('touch', case=False)

def extract_resolution(title):
    # Extract resolution in natural language
    natural_language_resolution = re.search(r'(HD\+|Full HD|4K Ultra HD|Ultra HD|Quad HD)', title)
    if natural_language_resolution:
        return natural_language_resolution.group()

    # Extract resolution in numbers
    number_resolution = re.search(r'\d{3,4}x\d{3,4}', title)
    if number_resolution:
        return number_resolution.group()

    # If no resolution is found
    return None

df_nb['resolution_parsed'] = df_nb.ScreenResolution.apply(lambda x: extract_resolution(x))
df_nb['resolution_parsed'] = df_nb['resolution_parsed'].str.replace('Full HD','1920x1080')
df_nb['resolution_parsed'] = df_nb['resolution_parsed'].str.replace('4K Ultra HD','3840x2160')
df_nb['resolution_parsed'] = df_nb['resolution_parsed'].str.replace('Quad HD','2560x1440')
def extract_storage_info(description):
    storage_pattern = r'(\d+(?:\.\d+)?)(GB|TB)\s+(SSD|HDD|Flash Storage|Hybrid)'
    storage_info = re.findall(storage_pattern, description)
    
    if not storage_info:
        return None
    
    memory = []
    technology = []
    
    for size, unit, tech in storage_info:
        size = float(size)
        
        if unit == 'TB':
            size *= 1024
        
        memory.append(size)
        technology.append(tech)
    
    return memory, technology
df_nb.Memory.fillna(-1,inplace=True)
df_nb_memory = df_nb.Memory.apply(lambda x: extract_storage_info(x)[0])
df_nb['first_storage'] = df_nb_memory.apply(lambda x: x[0])
df_nb['second_storage'] = df_nb_memory.apply(lambda x: x[1] if len(x) > 1 else -1)
df_nb_memory_type = df_nb.Memory.apply(lambda x: extract_storage_info(x)[1])
df_nb['first_type'] = df_nb_memory_type.apply(lambda x: x[0])
df_nb['second_storage_type'] = df_nb_memory_type.apply(lambda x: x[1] if len(x) > 1 else -1)
df_nb.Weight = df_nb.Weight.str.replace('kg', '').astype(float)
df_nb.Ram = df_nb.Ram.str.replace('gb', '', case=False)
df_nb.drop(1191, axis=0, inplace=True)
most_common_brands = df_nb.Company.value_counts().head(8).index.to_list()
df_nb = df_nb[df_nb.Company.str.contains('|'.join(most_common_brands))]
def cpu_gpu_brand(string):
    if 'intel' in string.lower():
        return 'intel'
    elif 'amd' in string.lower():
        return 'amd'
    else:
        return 'nvidia'
df_nb['gpu_brand'] = df_nb.Gpu.apply(lambda x: cpu_gpu_brand(x))
df_nb['cpu_brand'] = df_nb.Cpu.apply(lambda x: cpu_gpu_brand(x))
df_nb_clean = df_nb.drop(['laptop_ID', 'Company', 'Product','ScreenResolution', 'Cpu', 'Memory', 'Gpu'], axis=1).reset_index().drop('index', axis=1)
df_nb_clean.second_storage_type = df_nb_clean.second_storage_type.astype('str')
df_nb_categorical = df_nb_clean[['TypeName', 'OpSys', 'resolution_parsed', 'first_type', 'second_storage_type', 'gpu_brand', 'cpu_brand']]
df_nb_categorical.apply(lambda x: LabelEncoder().fit_transform(x), axis=0)
def le_encoding(column):
    try:
        le = LabelEncoder()
        return LabelEncoder().fit_transform(column)
    except:
        pass
    
le_encoding(df_nb_categorical)

df_nb_categorical = df_nb_categorical.apply(lambda x: le_encoding(x))
df_nb_clean[['TypeName', 'OpSys', 'resolution_parsed', 'first_type', 'second_storage_type', 'gpu_brand', 'cpu_brand']] = df_nb_categorical
print(df_nb_clean)

#------------------------------------DATA CLEANING END------------------------------------

df_nb_clean.Ram = df_nb_clean.Ram.astype('int')

df_nb_clean.ips = df_nb_clean.ips.astype(int)
df_nb_clean.touchscreen = df_nb_clean.touchscreen.astype(int)

X = df_nb_clean.drop('Price_in_euros', axis=1)
y = df_nb_clean.Price_in_euros

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

rfr = RandomForestRegressor()

rfr.fit(X_train,y_train)
y_pred = rfr.predict(X_test)

def regression_results(y_true, y_pred):

    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance,4))
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))

print(regression_results(y_test, y_pred))

# plot.residuals(y_test, y_pred)
dict_f_importance = dict(zip(rfr.feature_names_in_,rfr.feature_importances_))
pd.Series(dict_f_importance).sort_values().plot(kind='bar')
param_grid = {
    'n_estimators': [100, 200, 300], 
    'max_depth': [None, 5, 10], 
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4] 
}
 
grid_search = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=5, n_jobs=-1)

grid_search.fit(X, y)

"""print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)"""
y_pred_grid = grid_search.predict(X_test)
regression_results(y_test, y_pred_grid)

grid_search.best_estimator_.feature_importances_

dict_f_importance = dict(zip(grid_search.best_estimator_.feature_names_in_,grid_search.best_estimator_.feature_importances_))

pd.Series(dict_f_importance).sort_values().plot(kind='bar')

feature_importances = grid_search.best_estimator_.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importances)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importances of Best Estimator')
plt.show()