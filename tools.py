from parameters import *


def ImportData():
    with open("en-de/train.ende.src", "r") as ende_src:
        print("Source: ", ende_src.readline())
    with open("en-de/train.ende.mt", "r") as ende_mt:
        print("Translation: ", ende_mt.readline())
    with open("en-de/train.ende.scores", "r") as ende_scores:
        print("Score: ", ende_scores.readline())

    with open("en-de/train.ende.src", "r") as f:
        de_train_src = f.readlines()
    with open("en-de/train.ende.mt", "r") as f:
        de_train_mt = f.readlines()
    with open("en-de/train.ende.scores", 'r') as f:
        de_train_scores = f.readlines()
    with open("en-de/dev.ende.src", "r") as f:
        de_val_src = f.readlines()
    with open("en-de/dev.ende.mt", "r") as f:
        de_val_mt = f.readlines()
    with open("en-de/dev.ende.scores", 'r') as f:
        de_val_scores = f.readlines()

    return de_train_src, de_train_mt, de_train_scores, de_val_src, de_val_mt, de_val_scores

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def CleanCorpus(corpus):

    for i, sentence in enumerate(corpus):
        s = unicodeToAscii(sentence.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        s = s.replace(".", "")
        s+=" <EOS>"
        corpus[i] = s

    return corpus

def GetVocabulary(data):
    vocab = Counter()
    vocab["<SOS>"] = 0
    vocab["<EOS>"] = 1

    for line in data:
        for word in line.split():
            vocab[word] += 1
    word2index = dict(zip(vocab.keys(), range(0, len(vocab.keys()))))
    index2word = dict(zip(range(0, len(vocab.keys())), vocab.keys()))

    return vocab, word2index, index2word

def OneHotEncode(corpus, word2index):

    code = []
    n_vocab = len(word2index.keys())

    missing_words = 0
    total_words = 0
    for s in corpus:
        sentence_code = np.zeros((len(s.split()), n_vocab))
        for i, w in enumerate(s.split()):
            total_words+=1
            if w in word2index.keys():
                sentence_code[i, word2index[w]] =1
            else:
                missing_words+=1
        code.append(sentence_code)

    print("Missing Words in Vocabulary (%) during One Hot Encoding: ", missing_words/total_words)
    return code

def IndexEncode(corpus, word2index, max_length=0):

    code = np.zeros((len(corpus), max_length))
    n_vocab = len(word2index.keys())

    for i, s in enumerate(corpus):
        for j, w in enumerate(s.split()):
            code[i,j] = word2index[w]
        # for k in range(max_length-len(s.split())):
        #     code[i, len(s.split())+k] = 1

    return code

def MaxSentenceLength(corpus):
    return max([len(s.split()) for s in corpus])

