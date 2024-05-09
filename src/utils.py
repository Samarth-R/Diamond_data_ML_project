import os
import sys
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            print("*"*50)
            print("Evaluating", model, "Model")

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            print("Best params ===", gs.best_params_)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


#Dropping dimentionless diamonds
def drop_dimentionless_diamonds(df):
    df = df.drop(df[df["x"]==0].index)
    df = df.drop(df[df["y"]==0].index)
    df = df.drop(df[df["z"]==0].index)
    return df


def get_num_cat_columns(df, target_column):
    numerical_columns = list(df.select_dtypes(exclude="object").columns)
    categorical_columns = list(df.select_dtypes(include="object").columns)
    
    if target_column in numerical_columns:
        numerical_columns.remove(target_column)
    
    if target_column in categorical_columns:
        categorical_columns.remove(target_column)
    
    return numerical_columns, categorical_columns