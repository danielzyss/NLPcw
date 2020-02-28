from tools import *
from BPE import BPE


# def PreProcessData():
#
#     with open("en-de/train.ende.src", "r") as f:
#         de_train_src = f.readlines()
#     with open("en-de/train.ende.mt", "r") as f:
#         de_train_mt = f.readlines()
#     with open("en-de/train.ende.scores", 'r') as f:
#         de_train_scores = f.readlines()
#     with open("en-de/dev.ende.src", "r") as f:
#         de_val_src = f.readlines()
#     with open("en-de/dev.ende.mt", "r") as f:
#         de_val_mt = f.readlines()
#     with open("en-de/dev.ende.scores", 'r') as f:
#         de_val_scores = f.readlines()
#
#     BPEencoder = BPE(de_train_src, de_train_src,1500)
#     BPEencoder.LearnBPE()
#     BPEencoder.ApplyBPE()
#     de_train_src_bpe = BPEencoder.ApplyEncoding(de_train_src)
#     print(de_train_src_bpe[0])

def GetSentenceEmbedding(src, mt):

    src_emb = get_embeddings(src, nlp_en, 'en')
    mt_emb = get_embeddings(mt, nlp_de, 'de')

    X = [np.array(src_emb), np.array(mt_emb)]
    X_de = np.array(X)
    X_de = np.concatenate((X_de[0], X_de[1]), -1)

    return X_de


def GetMonolingualBPE(src, mt):
    bpemb_src = BPEmb(lang="en")
    bpemb_mt = BPEmb(lang="de")

    src_codes = []
    for sentence in src:
        src_codes.append(np.mean(bpemb_src.embed(sentence), axis=0))
    src_codes = np.array(src_codes)

    mt_codes = []
    for sentence in mt:
        mt_codes.append(np.mean(bpemb_mt.embed(sentence), axis=0))
    mt_codes = np.array(mt_codes)

    tnse_src = TSNE(n_components=2).fit_transform(src_codes)
    tnse_mt = TSNE(n_components=2).fit_transform(mt_codes)

    output = np.concatenate((tnse_src, tnse_mt), 1)

    return output



def GetMultilingualPBE(src, mt):

    bpemb = BPEmb(lang="multi")

    src_codes = []
    for sentence in src:
        src_codes.append(np.mean(bpemb.embed(sentence), axis=0))
    src_codes = np.array(src_codes)

    mt_codes = []
    for sentence in mt:
        mt_codes.append(np.mean(bpemb.embed(sentence), axis=0))
    mt_codes = np.array(mt_codes)

    tnse_src = TSNE(n_components=2).fit_transform(src_codes)
    tnse_mt = TSNE(n_components=2).fit_transform(mt_codes)

    output = np.concatenate((tnse_src, tnse_mt), 1)

    return output



def PreprocessDataSetForAttention(de_train_src, de_train_mt, de_val_src, de_val_mt):


    de_train_src = CleanCorpus(de_train_src.copy())
    de_train_mt = CleanCorpus(de_train_mt.copy())
    de_val_src = CleanCorpus(de_val_src.copy())
    de_val_mt = CleanCorpus(de_val_mt.copy())

    max_length = MaxSentenceLength(de_train_mt + de_train_src + de_val_mt + de_val_src)

    src_vocab, word2index_src, index2word_src = GetVocabulary(de_train_src + de_val_src )
    mt_vocab, word2index_mt, index2word_mt = GetVocabulary(de_train_mt + de_val_mt )

    train_src_code = IndexEncode(de_train_src, word2index_src, max_length)
    train_mt_code = IndexEncode(de_train_mt, word2index_mt, max_length)

    val_src_code = IndexEncode(de_val_src, word2index_src, max_length)
    val_mt_code = IndexEncode(de_val_mt, word2index_mt, max_length)

    input_size = len(src_vocab)
    output_size = len(mt_vocab)


    return train_src_code, train_mt_code, val_src_code, val_mt_code, max_length, input_size, output_size


def Regression(X_train, y_train):

    K = Matern()
    GP = GaussianProcessRegressor(kernel=K)
    GP.fit(X_train, y_train)

    return GP

