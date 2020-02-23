from funk import *


if __name__ == "__main__":

    ImportData()
    X_train_de, X_val_de, y_train_de, y_val_de = GetTrainingAndValidationSet()
    print(X_train_de)
