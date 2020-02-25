from funk import *

from Seq2SeqAttention import NMTwithAttention

if __name__ == "__main__":

    de_train_src, de_train_mt, de_train_scores, de_val_src, de_val_mt, de_val_scores = ImportData()
    train_src_code, train_mt_code, val_src_code, val_mt_code, max_length, input_size, output_size = PreprocessDataSetForAttention(de_train_src, de_train_mt,  de_val_src, de_val_mt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NMTatt = NMTwithAttention(max_length, input_size,output_size, hidden_size=256, device=device)
    NMTatt.Train(train_src_code,train_mt_code, n_epochs=10)
    
    # NMTatt.LoadModel()
    # NMTatt.InferAttention(val_src_code, val_mt_code)


