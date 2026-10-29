from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
X,y=load_breast_cancer(return_X_y=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
model=GridSearchCV(Pipeline([("scaler",StandardScaler()),("svc",SVC(kernel="rbf"))]),{"svc__C":[0.1,1,10,100],"svc__gamma":["scale",0.001,0.01,0.1]},cv=5)
model.fit(X_train,y_train)
print(accuracy_score(y_test,model.predict(X_test)))