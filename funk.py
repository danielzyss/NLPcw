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

def CoverageDeviationPenalty(attentionWeights):

    output = []
    for sentence in attentionWeights:
        CDP = -1/len(sentence) * np.sum(np.log(1+(1-np.sum(sentence, 1))**2))
        output.append(CDP)

    return np.array(output)

def AbsentmindednessPenaltyOut(attentionweights):
    output = []
    for sentence in attentionweights:
        APout =  - np.sum(sentence * np.log(sentence))/sentence.shape[1]
        output.append(APout)
    return output

def AbsentmindednessPenaltyIn(attentionweights):
    output = []
    for sentence in attentionweights:
        sentence = np.array(list(zip(*sentence)))
        APout =  - np.sum(sentence * np.log(sentence))/sentence.shape[1]
        output.append(APout)
    return output
