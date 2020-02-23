from parameters import *


def ImportData():
    with open("en-de/train.ende.src", "r") as ende_src:
        print("Source: ", ende_src.readline())
    with open("en-de/train.ende.mt", "r") as ende_mt:
        print("Translation: ", ende_mt.readline())
    with open("en-de/train.ende.scores", "r") as ende_scores:
        print("Score: ", ende_scores.readline())

def get_sentence_emb(line,nlp,lang):
  if lang == 'en':
    text = line.lower()
    l = [token.lemma_ for token in nlp.tokenizer(text)]
    l = ' '.join([word for word in l if word not in stop_words_en])

  elif lang == 'de':
    text = line.lower()
    l = [token.lemma_ for token in nlp.tokenizer(text)]
    l= ' '.join([word for word in l if word not in stop_words_de])

  sen = nlp(l)
  return sen.vector

def get_embeddings(f,nlp,lang):
  file = open(f)
  lines = file.readlines()
  sentences_vectors =[]

  for l in lines:
      vec = get_sentence_emb(l,nlp,lang)
      if vec is not None:
        vec = np.mean(vec)
        sentences_vectors.append(vec)
      else:
        print("didn't work :", l)
        sentences_vectors.append(0)

  return sentences_vectors

def writeScores(method_name,scores):
    fn = "predictions.txt"
    print("")
    with open(fn, 'w') as output_file:
        for idx,x in enumerate(scores):
            #out =  metrics[idx]+":"+str("{0:.2f}".format(x))+"\n"
            #print(out)
            output_file.write(f"{x}\n")
