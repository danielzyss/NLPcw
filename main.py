from funk import *



if __name__ == "__main__":

    de_train_src, de_train_mt, de_train_scores, de_val_src, de_val_mt, de_val_scores = ImportData()
    AttnFeatures_train, AttnFeatures_val, SentenceEmbedding_train, \
    SentenceEmbedding_val, monoBPE_train, monoBPE_val, multiBPE_train, multiBPE_val = GetFeatures(de_train_src, de_train_mt,
                                                                                                  de_val_src, de_val_mt,
                                                                                                  from_scratch=False)


    regression_type = ["GP", "forrest", "nn", "svm"]
    features_type = [("sentence",),
                     ("Attn",),
                     ("monoBPE"),
                     ("multiBPE"),
                     ("sentence", "Attn"),
                     ("sentence", "monoBPE"),
                     ("sentence", "multiBPE"),
                     ("sentence", "Attn", "monoBPE"),
                     ("sentence", "Attn", "multiBPE"),
                     ("sentence", "Attn", "monoBPE", "multiBPE"),
                     ("Attn", "multiBPE"),]


    for rt in regression_type:
        for ft in features_type:

            features_train = []
            features_val = []
            for f in ft:
                if f=="sentence":
                    features_train.append(SentenceEmbedding_train.reshape(-1,4))
                    features_val.append(SentenceEmbedding_val.reshape(-1,4))
                elif f=="Attn":
                    features_train.append(AttnFeatures_train.reshape(-1,4))
                    features_val.append(AttnFeatures_val.reshape(-1,4))
                elif f=="monoBPE":
                    features_train.append(monoBPE_train.reshape(-1,4))
                    features_val.append(monoBPE_val.reshape(-1,4))
                elif f=="multiBPE":
                    features_train.append(multiBPE_train.reshape(-1,4))
                    features_val.append(multiBPE_val.reshape(-1,4))
                else:
                    print("Error in feature type")
                    pass

            X_train = np.concatenate(features_train, 1)
            X_val = np.concatenate(features_val, 1)

            de_train_scores = np.array(de_train_scores, dtype=np.float64)
            de_val_scores = np.array(de_val_scores, dtype=np.float64)

            reg = Regression(X_train, de_train_scores, type=rt)

            y_pred_train = reg.predict(X_train)
            P_train = pearson(de_train_scores, y_pred_train)
            mae_train = mean_absolute_error(de_train_scores, y_pred_train)
            rmse_train = np.sqrt(mean_squared_error(de_train_scores, y_pred_train))


            y_pred_val = reg.predict(X_val)
            P_val = pearson(de_val_scores, y_pred_val)
            mae_val = mean_absolute_error(de_val_scores, y_pred_val)
            rmse_val = np.sqrt(mean_squared_error(de_val_scores, y_pred_val))


            print("-"*50)
            print("Regressor: ", rt)
            print("Features: ", "+".join(ft))
            print("Training Scores  - pearson:", P_train, "MAE:", mae_train, "RMSE:", rmse_train)
            print("Testing Scores   - pearson:", P_val, "MAE:", mae_val, "RMSE:", rmse_val)
            print("-" * 50)





