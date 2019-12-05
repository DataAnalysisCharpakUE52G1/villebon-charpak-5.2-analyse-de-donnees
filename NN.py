from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

def NN_MLP_Classifier(X, y, units, layer, activation_function):
    X_train, Xtest, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    hidden_layer =  [units] * layer
    mlp = MLPClassifier(solver='lbfgs',activation = activation_function, random_state=0, hidden_layer_sizes=hidden_layer).fit(X_train, y_train)
    print("Accuracy on training set: {:.3f}".format(mlp.score(X_train, y_train)))
    print("Accuracy on test set: {:.3f}".format(mlp.score(X_test, y_test)))
    return mlp


