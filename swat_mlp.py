#https://www.springboard.com/blog/beginners-guide-neural-network-in-python-scikit-learn-0-18/
import pandas as pd
print('Start ...')


###### Data Import ############################
DATA = pd.read_csv("dataset.csv")
DATA = DATA.drop(' Timestamp', axis=1)
print(DATA.head())

X = DATA.drop('Class', axis=1)
y = DATA['Class']

###### Data Preprocessing ###############
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)


# Fit only to the training data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

###### Training the model ######################
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
mlp.fit(X_train,y_train)

MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(13, 13, 13), learning_rate='constant',
       learning_rate_init=0.001, max_iter=500, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)


##### Prediction and Evaluation ################
predictions = mlp.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


print('End of run')
