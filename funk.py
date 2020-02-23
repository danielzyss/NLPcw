from tools import *

def GetTrainingAndValidationSet():
    de_train_src = get_embeddings("en-de/train.ende.src", nlp_en, 'en')
    de_train_mt = get_embeddings("en-de/train.ende.mt", nlp_de, 'de')

    f_train_scores = open("en-de/train.ende.scores", 'r')
    de_train_scores = f_train_scores.readlines()

    de_val_src = get_embeddings("en-de/dev.ende.src", nlp_en, 'en')
    de_val_mt = get_embeddings("en-de/dev.ende.mt", nlp_de, 'de')
    f_val_scores = open("en-de/dev.ende.scores", 'r')
    de_val_scores = f_val_scores.readlines()

    print(f"Training mt: {len(de_train_mt)} Training src: {len(de_train_src)}")
    print()
    print(f"Validation mt: {len(de_val_mt)} Validation src: {len(de_val_src)}")

    # Put the features into a list
    X_train = [np.array(de_train_src), np.array(de_train_mt)]
    X_train_de = np.array(X_train).transpose()

    X_val = [np.array(de_val_src), np.array(de_val_mt)]
    X_val_de = np.array(X_val).transpose()

    # Scores
    train_scores = np.array(de_train_scores).astype(float)
    y_train_de = train_scores

    val_scores = np.array(de_val_scores).astype(float)
    y_val_de = val_scores

    return X_train_de, X_val_de, y_train_de, y_val_de
