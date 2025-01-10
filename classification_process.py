import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score,KFold  # type: ignore
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # type: ignore
from sklearn.neighbors import KNeighborsClassifier  # type: ignore
from sklearn.metrics import  accuracy_score,classification_report,roc_auc_score,f1_score,top_k_accuracy_score, top_k_accuracy_score  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
from sklearn.decomposition import PCA # type: ignore
from sklearn.feature_selection import SelectKBest, f_classif  # type: ignore

def knn_euclidean(X_train, X_test, y_train, y_test, k=5):
    print("[INFO] Calculando distância euclidiana...")
    
    selector = SelectKBest(score_func=f_classif, k='all')
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)
    
    n_classes = len(set(y_train))  
    
    best_k = k
    best_accuracy = 0
    best_scores = None
    
    for k_value in range(1, 16): 
        knn = KNeighborsClassifier(n_neighbors=k_value, metric='euclidean', weights='distance', algorithm='auto')
        scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
        accuracy = scores.mean()
        
        auc = None
        f1 = None
        top_k_acc = None
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        y_pred_proba = knn.predict_proba(X_test) if hasattr(knn, "predict_proba") else None
        
        if y_pred_proba is not None:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            if k_value < n_classes:
                top_k_acc = top_k_accuracy_score(y_test, y_pred_proba, k=k_value)
        
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"[INFO] Testando k={k_value}... Acurácia média: {accuracy:.4f} | AUC: {auc:.4f} | F1-Score: {f1:.4f} | Top-{k_value} Accuracy: {top_k_acc if top_k_acc is not None else 'N/A'}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k_value
            best_scores = scores
    
    print(f"[INFO] Melhor valor de k: {best_k}")
    print(f"\nCross-Validation Results for Euclidean KNN:")
    print(f"Mean Accuracy: {best_scores.mean():.4f} (±{best_scores.std():.4f})")
    
    knn = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean', weights='distance', algorithm='auto')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    y_pred_proba = knn.predict_proba(X_test) if hasattr(knn, "predict_proba") else None
    
    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr') if y_pred_proba is not None else None
    f1 = f1_score(y_test, y_pred, average='weighted')
    top_k_acc = top_k_accuracy_score(y_test, y_pred_proba, k=best_k) if y_pred_proba is not None else None
    
    print(f"\n[INFO] KNN Classification Report (Euclidean, K={best_k}):")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    if auc is not None:
        print(f"AUC (Area Under the ROC Curve): {auc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    if top_k_acc is not None:
        print(f"Top-{best_k} Accuracy: {top_k_acc:.4f}")
    
    return {
        "mean_accuracy": best_scores.mean(),
        "std_accuracy": best_scores.std(),
        "auc": auc,
        "f1_score": f1,
        "top_k_accuracy": top_k_acc,
    }

def knn_cosine_cv(X, y, k=5, n_splits=10):
    X = X.to_numpy() if isinstance(X, pd.DataFrame) else np.array(X)
    y = y.to_numpy() if isinstance(y, pd.Series) else np.array(y)
    
    print("[INFO] Calculando a matriz de similaridade cosseno...")
    
    best_k = k
    best_overall_accuracy = 0
    all_k_scores = []
    
    auc_scores = []
    f1_scores = []
    top_k_accuracies = []

    for k_value in range(1, 16):
        fold_scores = []
        fold_auc_scores = []
        fold_f1_scores = []
        fold_top_k_accuracies = []

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for train_idx, test_idx in kf.split(X):
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]
            
            X_train_cosine = cosine_similarity(X_train_fold)
            X_test_cosine = cosine_similarity(X_test_fold, X_train_fold)
            
            X_train_cosine = np.clip(X_train_cosine, 0, 1)
            X_test_cosine = np.clip(X_test_cosine, 0, 1)
            
            X_train_cosine = np.log1p(X_train_cosine)
            X_test_cosine = np.log1p(X_test_cosine)
            
            min_max_scaler = MinMaxScaler(feature_range=(0, 1))
            X_train_cosine = min_max_scaler.fit_transform(X_train_cosine)
            X_test_cosine = min_max_scaler.transform(X_test_cosine)
            
            knn = KNeighborsClassifier(n_neighbors=k_value, metric='precomputed', weights='distance')
            knn.fit(X_train_cosine, y_train_fold)
            y_pred = knn.predict(X_test_cosine)
            y_pred_prob = knn.predict_proba(X_test_cosine)
            
            fold_scores.append(accuracy_score(y_test_fold, y_pred))
            
            auc = roc_auc_score(y_test_fold, y_pred_prob, multi_class='ovr', average='macro') if len(np.unique(y_test_fold)) > 1 else 0
            fold_auc_scores.append(auc)
            
            f1 = f1_score(y_test_fold, y_pred, average='macro')
            fold_f1_scores.append(f1)
            
            top_k_accuracy = np.mean([1 if y_test_fold[i] in y_pred_prob[i].argsort()[-k_value:][::-1] else 0 for i in range(len(y_test_fold))])
            fold_top_k_accuracies.append(top_k_accuracy)

        mean_accuracy = np.mean(fold_scores)
        mean_auc = np.mean(fold_auc_scores)
        mean_f1 = np.mean(fold_f1_scores)
        mean_top_k_accuracy = np.mean(fold_top_k_accuracies)
        
        all_k_scores.append(mean_accuracy)
        
        print(f"[INFO] Testando k={k_value}... Acurácia média: {mean_accuracy:.4f} | AUC: {mean_auc:.4f} | F1-Score: {mean_f1:.4f} | Top-{k_value} Accuracy: {mean_top_k_accuracy:.4f}")
        
        if mean_accuracy > best_overall_accuracy:
            best_overall_accuracy = mean_accuracy
            best_k = k_value
    
    print(f"[INFO] Melhor valor de k: {best_k}")
    
    final_predictions = []
    final_true_labels = []
    final_fold_scores = []
    
    final_auc_scores = []
    final_f1_scores = []
    final_top_k_accuracies = []
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(X):
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        
        X_train_cosine = cosine_similarity(X_train_fold)
        X_test_cosine = cosine_similarity(X_test_fold, X_train_fold)
        X_train_cosine = np.clip(X_train_cosine, 0, 1)
        X_test_cosine = np.clip(X_test_cosine, 0, 1)
        X_train_cosine = np.log1p(X_train_cosine)
        X_test_cosine = np.log1p(X_test_cosine)
        min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        X_train_cosine = min_max_scaler.fit_transform(X_train_cosine)
        X_test_cosine = min_max_scaler.transform(X_test_cosine)
        
        knn = KNeighborsClassifier(n_neighbors=best_k, metric='precomputed', weights='distance')
        knn.fit(X_train_cosine, y_train_fold)
        y_pred = knn.predict(X_test_cosine)
        y_pred_prob = knn.predict_proba(X_test_cosine)
        
        final_predictions.extend(y_pred)
        final_true_labels.extend(y_test_fold)
        final_fold_scores.append(accuracy_score(y_test_fold, y_pred))
        
        auc = roc_auc_score(y_test_fold, y_pred_prob, multi_class='ovr', average='macro') if len(np.unique(y_test_fold)) > 1 else 0
        final_auc_scores.append(auc)
        
        f1 = f1_score(y_test_fold, y_pred, average='macro')
        final_f1_scores.append(f1)
        
        top_k_accuracy = np.mean([1 if y_test_fold[i] in y_pred_prob[i].argsort()[-best_k:][::-1] else 0 for i in range(len(y_test_fold))])
        final_top_k_accuracies.append(top_k_accuracy)
    
    print(f"\n[INFO] KNN Classification Report (Cosine, K={best_k}):")
    print(classification_report(final_true_labels, final_predictions))
    print(f"Accuracy: {accuracy_score(final_true_labels, final_predictions):.4f}")
    print(f"AUC: {np.mean(final_auc_scores):.4f}")
    print(f"F1-Score: {np.mean(final_f1_scores):.4f}")
    print(f"Top-{best_k} Accuracy: {np.mean(final_top_k_accuracies):.4f}")
    
    return np.mean(final_fold_scores), np.std(final_fold_scores)

def knn_classification(df: pd.DataFrame, metric: str = 'euclidean', k: int = 5):
    
    if 'embedding' not in df.columns or 'syndrome_id' not in df.columns:
        print("[ERROR] Required columns ('embedding', 'syndrome_id') not found in the DataFrame.")
        return

    embeddings = np.array(df['embedding'].tolist())
    labels = df['syndrome_id']

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X_test: {X_test.shape}")

    if metric == 'euclidean':
        results = knn_euclidean(X_train, X_test, y_train, y_test, k)
        mean_accuracy = results["mean_accuracy"]
        std_accuracy = results["std_accuracy"]
        print(f"\nCross-Validation Results for Euclidean KNN:")
        print(f"Mean Accuracy: {mean_accuracy:.4f} (±{std_accuracy:.4f})")
    elif metric == 'cosine':
        mean_accuracy, std_accuracy = knn_cosine_cv(X_train, y_train, k)
        print(f"\nCross-Validation Results for Cosine KNN:")
        print(f"Mean Accuracy: {mean_accuracy:.4f} (±{std_accuracy:.4f})")
    else:
        print(f"[ERROR] Métrica desconhecida: {metric}")


if __name__ == "__main__":
    from data_processing import DataProcessor  
    pickle_path = "/media/paulo-jaka/Extras/DesafiosML/mini_gm_public_v0.1.p"
    processor = DataProcessor(pickle_path)
    df = processor.load_and_flatten_data()

    if df is not None:
        print("\n[INFO] Running KNN with Euclidean Distance...")
        knn_classification(df, metric='euclidean', k=5)

        print("\n[INFO] Running KNN with Cosine Distance...")
        knn_classification(df, metric='cosine', k=5)
