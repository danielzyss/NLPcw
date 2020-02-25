from tools import *
from BPE import BPE

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


def PreprocessDataSetForAttention(de_train_src, de_train_mt, de_val_src, de_val_mt):


    de_train_src = CleanCorpus(de_train_src)
    de_train_mt = CleanCorpus(de_train_mt)
    de_val_src = CleanCorpus(de_val_src)
    de_val_mt = CleanCorpus(de_val_mt)

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


