import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mlflow_uri', default='http://localhost:5001')
args = parser.parse_args()
mlflow.set_tracking_uri(args.mlflow_uri)
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
with mlflow.start_run():
    n_estimators = 50
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_metric('accuracy', acc)
    mlflow.sklearn.log_model(model, 'model')
    print('Logged run with accuracy', acc)
