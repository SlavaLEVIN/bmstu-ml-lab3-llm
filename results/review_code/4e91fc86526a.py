from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

X,y=load_iris(return_X_y=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
model=GridSearchCV(Pipeline([("scaler",StandardScaler()),("knn",KNeighborsClassifier())]),{"knn__n_neighbors":[1,3,5,7,9]},cv=5,scoring="accuracy")
model.fit(X_train,y_train)
print(model.score(X_test,y_test))