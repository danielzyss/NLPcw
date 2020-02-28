from tools import *
from BPE import BPE
from Seq2SeqAttention import NMTwithAttention


def GetSentenceEmbedding(src, mt):

    src_emb = get_embeddings(src, nlp_en, 'en')
    mt_emb = get_embeddings(mt, nlp_de, 'de')

    X = [np.array(src_emb), np.array(mt_emb)]
    X_de = np.array(X)
    X_de = np.concatenate((X_de[0], X_de[1]), -1)

    return X_de

def GetMonolingualBPE(src_train, src_val, mt_train, mt_val):

    bpemb_src = BPEmb(lang="en")
    bpemb_mt = BPEmb(lang="de")

    src_codes = []
    for sentence in src_train:
        src_codes.append(np.mean(bpemb_src.embed(sentence), axis=0))
    for sentence in src_val:
        src_codes.append(np.mean(bpemb_src.embed(sentence), axis=0))
    src_codes = np.array(src_codes)

    mt_codes = []
    for sentence in mt_train:
        mt_codes.append(np.mean(bpemb_mt.embed(sentence), axis=0))
    for sentence in mt_val:
        mt_codes.append(np.mean(bpemb_mt.embed(sentence), axis=0))
    mt_codes = np.array(mt_codes)

    tnse_src = TSNE(n_components=2).fit_transform(src_codes)
    tnse_mt = TSNE(n_components=2).fit_transform(mt_codes)

    output_train = np.concatenate((tnse_src[:len(src_train)], tnse_mt[:len(mt_train)]), 1)
    output_val = np.concatenate((tnse_src[len(src_train):], tnse_mt[len(mt_train):]), 1)

    return output_train, output_val


def GetMultilingualPBE(src_train, src_val, mt_train, mt_val):

    bpemb = BPEmb(lang="multi")

    src_codes = []
    for sentence in src_train:
        src_codes.append(np.mean(bpemb.embed(sentence), axis=0))
    for sentence in src_val:
        src_codes.append(np.mean(bpemb.embed(sentence), axis=0))
    src_codes = np.array(src_codes)

    mt_codes = []
    for sentence in mt_train:
        mt_codes.append(np.mean(bpemb.embed(sentence), axis=0))
    for sentence in mt_val:
        mt_codes.append(np.mean(bpemb.embed(sentence), axis=0))
    mt_codes = np.array(mt_codes)

    tnse_src = TSNE(n_components=2).fit_transform(src_codes)
    tnse_mt = TSNE(n_components=2).fit_transform(mt_codes)

    output_train = np.concatenate((tnse_src[:len(src_train)], tnse_mt[:len(mt_train)]), 1)
    output_val = np.concatenate((tnse_src[len(src_train):], tnse_mt[len(mt_train):]), 1)

    return output_train, output_val

def PreprocessDataSetForAttention(de_train_src, de_train_mt, de_val_src, de_val_mt):


    de_train_src = CleanCorpus(de_train_src.copy())
    de_train_mt = CleanCorpus(de_train_mt.copy())
    de_val_src = CleanCorpus(de_val_src.copy())
    de_val_mt = CleanCorpus(de_val_mt.copy())

    max_length = MaxSentenceLength(de_train_mt + de_train_src )

    src_vocab, word2index_src, index2word_src = GetVocabulary(de_train_src  )
    mt_vocab, word2index_mt, index2word_mt = GetVocabulary(de_train_mt )

    train_src_code = IndexEncode(de_train_src, word2index_src, max_length)
    train_mt_code = IndexEncode(de_train_mt, word2index_mt, max_length)

    val_src_code = IndexEncode(de_val_src, word2index_src, max_length)
    val_mt_code = IndexEncode(de_val_mt, word2index_mt, max_length)

    input_size = len(src_vocab)
    output_size = len(mt_vocab)


    return train_src_code, train_mt_code, val_src_code, val_mt_code, max_length, input_size, output_size


def Regression(X_train, y_train, type="GP"):

    if type=="GP":
        K = RBF()
        reg = GaussianProcessRegressor(kernel=K, alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,).fit(X_train, y_train)
    elif type=="forrest":
        extra = ExtraTreeRegressor(criterion="mae")
        reg = BaggingRegressor(extra, random_state=0).fit(X_train, y_train)
    elif type=="nn":
        reg = MLPRegressor(hidden_layer_sizes=(100, 200), activation='relu', solver='lbfgs', learning_rate='adaptive', tol=1e-6, verbose=False, max_iter=1000)
        reg.fit(X_train, y_train)
    elif type=="svm":
        reg = SVR(kernel="rbf").fit(X_train, y_train)
    else:
        print("Wrong Regressor Type")
        return 0

    return reg

def GetFeatures(de_train_src, de_train_mt, de_val_src, de_val_mt, from_scratch=False, retrain=False):

    # ATTENTION WEIGHTS

    if from_scratch:
        train_src_code, train_mt_code, val_src_code, val_mt_code, max_length, input_size, output_size = PreprocessDataSetForAttention(
            de_train_src, de_train_mt, de_val_src, de_val_mt)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        NMTatt = NMTwithAttention(max_length, input_size,output_size, hidden_size=256, device=device)
        if retrain:
            NMTatt.Train(train_src_code,train_mt_code, n_epochs=10)

        NMTatt.LoadModel()
        attnWeights_val = NMTatt.InferAttention(val_src_code, val_mt_code)
        attnWeights_train = NMTatt.InferAttention(train_src_code, train_mt_code)

        np.save("tmp/AttnWeights_train.npy", attnWeights_train)
        np.save("tmp/AttnWeights_val.npy", attnWeights_val)
    else:
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

    AttnFeatures_train = np.concatenate((CDP_train.reshape(-1,1),
                              API_train.reshape(-1,1),
                              APO_train.reshape(-1,1),
                              APR_train.reshape(-1,1),),1)
    AttnFeatures_val = np.concatenate((CDP_val.reshape(-1,1),
                            API_val.reshape(-1,1),
                            APO_val.reshape(-1,1),
                            APR_val.reshape(-1,1),),1)

    # EMBEDDINGS

    if from_scratch:
        SentenceEmbedding_train = GetSentenceEmbedding(de_train_src, de_train_mt)
        np.save("tmp/Sentence_embedding_train.npy", SentenceEmbedding_train)
        SentenceEmbedding_val = GetSentenceEmbedding(de_val_src, de_val_mt)
        np.save("tmp/Sentence_embedding_val.npy", SentenceEmbedding_val)
    else:
        SentenceEmbedding_train = np.load("tmp/Sentence_embedding_train.npy")
        SentenceEmbedding_val = np.load("tmp/Sentence_embedding_val.npy")


    if from_scratch:
        monoBPE_train, monoBPE_val = GetMonolingualBPE(de_train_src, de_val_src, de_train_mt, de_val_mt)
        np.save("tmp/monoBPE_train.npy", monoBPE_train)
        np.save("tmp/monoBPE_val.npy", monoBPE_val )
    else:
        monoBPE_train = np.load("tmp/monoBPE_train.npy")
        monoBPE_val = np.load("tmp/monoBPE_val.npy")


    if from_scratch:
        multiBPE_train, multiBPE_val = GetMultilingualPBE(de_train_src, de_val_src, de_train_mt, de_val_mt)
        np.save("tmp/multiBPE_train.npy", multiBPE_train)
        np.save("tmp/multiBPE_val.npy", multiBPE_val)
    else:
        multiBPE_train = np.load("tmp/multiBPE_train.npy")
        multiBPE_val = np.load("tmp/multiBPE_val.npy")

    return AttnFeatures_train, AttnFeatures_val, SentenceEmbedding_train, SentenceEmbedding_val, monoBPE_train, monoBPE_val, multiBPE_train, multiBPE_val