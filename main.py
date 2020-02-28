from funk import *

from Seq2SeqAttention import NMTwithAttention

if __name__ == "__main__":

    de_train_src, de_train_mt, de_train_scores, de_val_src, de_val_mt, de_val_scores = ImportData()
    train_src_code, train_mt_code, val_src_code, val_mt_code, max_length, input_size, output_size = PreprocessDataSetForAttention(de_train_src, de_train_mt,  de_val_src, de_val_mt)


    # ATTENTION WEIGHTS

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # NMTatt = NMTwithAttention(max_length, input_size,output_size, hidden_size=256, device=device)
    # # NMTatt.Train(train_src_code,train_mt_code, n_epochs=10)
    #
    # NMTatt.LoadModel()
    # output = NMTatt.InferAttention(train_src_code, train_mt_code)

    attnWeights_val = np.load("tmp/AttnWeights_val.npy")
    attnWeights_train = np.load("tmp/AttnWeights_train.npy")

    CDP_train = CoverageDeviationPenalty(attnWeights_train)
    API_train = AbsentmindednessPenaltyIn(attnWeights_train)
    APO_train = AbsentmindednessPenaltyOut(attnWeights_train)
    APR_train = AbsentmindednessPenaltyRatio(API_train, APO_train)

    CDP_val = CoverageDeviationPenalty(attnWeights_val)
    API_val = AbsentmindednessPenaltyIn(attnWeights_val)
    APO_val = AbsentmindednessPenaltyOut(attnWeights_val)
    APR_val = AbsentmindednessPenaltyRatio(API_val, APO_val)

    # EMBEDDINGS

    # SentenceEmbedding_train = GetSentenceEmbedding(de_train_src, de_train_mt)
    # np.save("tmp/Sentence_embedding_train.npy", SentenceEmbedding_train)
    # SentenceEmbedding_val = GetSentenceEmbedding(de_val_src, de_val_mt)
    # np.save("tmp/Sentence_embedding_val.npy", SentenceEmbedding_train)

    SentenceEmbedding_train = np.load("tmp/Sentence_embedding_train.npy")
    SentenceEmbedding_val = np.load("tmp/Sentence_embedding_val.npy")

    # monoBPE_train = GetMonolingualBPE(de_train_src, de_train_mt)
    # np.save("tmp/monoBPE_train.npy", monoBPE_train)
    # monoBPE_val = GetMonolingualBPE(de_val_src, de_val_mt)
    # np.save("tmp/monoBPE_val.npy", monoBPE_val )

    monoBPE_train = np.load("tmp/monoBPE_train.npy")
    monoBPE_val = np.load("tmp/monoBPE_val.npy")

    # multiBPE_train = GetMonolingualBPE(de_train_src, de_train_mt)
    # np.save("tmp/multiBPE_train.npy", multiBPE_train)
    # multiBPE_val = GetMonolingualBPE(de_val_src, de_val_mt)
    # np.save("tmp/multiBPE_val.npy", multiBPE_val)

    multiBPE_train = np.load("tmp/multiBPE_train.npy")
    multiBPE_val = np.load("tmp/multiBPE_val.npy")


    X_train = np.concatenate((CDP_train.reshape(-1,1), API_train.reshape(-1,1), APO_train.reshape(-1,1), APR_train.reshape(-1,1)), 1)
    X_val = np.concatenate((CDP_val.reshape(-1,1), API_val.reshape(-1,1), APO_val.reshape(-1,1), APR_val.reshape(-1,1)), 1)

    # REGRESSION

    # de_train_scores = np.array(de_train_scores, dtype=np.float64)
    # de_val_scores = np.array(de_val_scores, dtype=np.float64)
    # GP = Regression(X_train, de_train_scores)
    # y_pred = GP.predict(X_val)
    # print(pearson(de_val_scores, y_pred))









