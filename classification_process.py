import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score,KFold  # type: ignore
from sklearn.preprocessing import  MinMaxScaler , LabelBinarizer # type: ignore
from sklearn.neighbors import KNeighborsClassifier  # type: ignore
from sklearn.metrics import  accuracy_score,classification_report,roc_auc_score,f1_score,top_k_accuracy_score, top_k_accuracy_score, roc_curve, auc  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
from sklearn.feature_selection import SelectKBest, f_classif  # type: ignore
import matplotlib.pyplot as plt
import os

def knn_euclidean(X_train, X_test, y_train, y_test):
    print("[INFO] Calculando distância euclidiana...")
    
    selector = SelectKBest(score_func=f_classif, k='all')
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)
    
    n_classes = len(set(y_train))  
    
    best_accuracy = 0
    best_k = None
    best_scores = None
    
    metrics_per_k = []  # Lista para armazenar as métricas para cada k
    aucs = []  # Para armazenar as AUCs de cada k
    f1_scores = []  # Para armazenar os F1-Scores de cada k
    top_k_accuracies = []  # Para armazenar o Top-K Accuracy de cada k
    
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
        
        # Atualizando as métricas
        aucs.append(auc if auc is not None else 0)
        f1_scores.append(f1)
        top_k_accuracies.append(top_k_acc if top_k_acc is not None else 0)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k_value
            best_scores = scores
        
        # Armazenando as métricas para cada k
        metrics_per_k.append({
            'k': k_value,
            'accuracy': accuracy,
            'auc': auc,
            'f1_score': f1,
            'top_k_accuracy': top_k_acc,
            'y_pred_proba': y_pred_proba  # Adicionando as probabilidades previstas
        })
    
    print(f"[INFO] Melhor valor de k: {best_k}")
    print(f"\nCross-Validation Results for Euclidean KNN:")
    print(f"Mean Accuracy: {best_scores.mean():.4f} (±{best_scores.std():.4f})")
    
    # Média das AUCs, F1-Scores e Top-K Accuracies
    print(f"Mean AUC: {np.mean(aucs):.4f}")
    print(f"Mean F1-Score: {np.mean(f1_scores):.4f}")
    print(f"Mean Top-K Accuracy: {np.mean(top_k_accuracies):.4f}")
    
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
        "best_k": best_k,
        "best_accuracy": best_accuracy,        # Melhor acurácia
        "best_scores": best_scores,            # Resultados de cross-validation
        "final_auc": auc,                      # AUC do melhor modelo
        "final_f1_score": f1,                  # F1-Score do melhor modelo
        "final_top_k_accuracy": top_k_acc,     # Top-K Accuracy do melhor modelo
        "metrics_per_k": metrics_per_k,        # Métricas para cada k
        "final_y_pred_proba": y_pred_proba     # Adicionando as probabilidades de previsão
    }


def preprocess_cosine_similarity(X_train, X_test):
    X_train_cosine = cosine_similarity(X_train)
    X_test_cosine = cosine_similarity(X_test, X_train)
    
    X_train_cosine = np.clip(X_train_cosine, 0, 1)
    X_test_cosine = np.clip(X_test_cosine, 0, 1)
    
    X_train_cosine = np.log1p(X_train_cosine)
    X_test_cosine = np.log1p(X_test_cosine)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_cosine = scaler.fit_transform(X_train_cosine)
    X_test_cosine = scaler.transform(X_test_cosine)
    
    return X_train_cosine, X_test_cosine

def calculate_metrics(y_true, y_pred, y_pred_prob, k_value):
    """Função para calcular as métricas"""
    auc = roc_auc_score(y_true, y_pred_prob, multi_class='ovr', average='macro')
    f1 = f1_score(y_true, y_pred, average='weighted')
    top_k_accuracy = np.mean([1 if y_true[i] in y_pred[i][:k_value] else 0 for i in range(len(y_true))])
    return auc, f1, top_k_accuracy

def knn_cosine_cv(X, y, n_splits=10):
    X = X.to_numpy() if isinstance(X, pd.DataFrame) else np.array(X)
    y = y.to_numpy() if isinstance(y, pd.Series) else np.array(y)
        
    print("[INFO] Normalizando os dados com MinMaxScaler...")
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    
    best_k = None
    best_overall_accuracy = 0
    best_scores = None 
    metrics_per_k = []

    for k_value in range(1, 16):
        fold_scores = []
        fold_auc_scores = []
        fold_f1_scores = []
        fold_top_k_accuracies = []
        fold_y_pred_prob = []

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for train_idx, test_idx in kf.split(X_normalized):
            X_train_fold, X_test_fold = X_normalized[train_idx], X_normalized[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]
            
            knn = KNeighborsClassifier(n_neighbors=k_value, metric='cosine', weights='distance')
            knn.fit(X_train_fold, y_train_fold)
            y_pred = knn.predict(X_test_fold)
            y_pred_prob = knn.predict_proba(X_test_fold)
            
            auc, f1, top_k_accuracy = calculate_metrics(y_test_fold, y_pred, y_pred_prob, k_value)
            
            fold_scores.append(accuracy_score(y_test_fold, y_pred))
            fold_auc_scores.append(auc)
            fold_f1_scores.append(f1)
            fold_top_k_accuracies.append(top_k_accuracy)
            fold_y_pred_prob.append(y_pred_prob)

        mean_accuracy = np.mean(fold_scores)
        mean_auc = np.mean(fold_auc_scores)
        mean_f1 = np.mean(fold_f1_scores)
        mean_top_k_accuracy = np.mean(fold_top_k_accuracies)
        
        metrics_per_k.append({
            'k': k_value,
            'accuracy': mean_accuracy,  # Renomeado para manter consistência
            'auc': mean_auc,
            'f1_score': mean_f1,  # Renomeado para manter consistência
            'top_k_accuracy': mean_top_k_accuracy,
            'y_pred_proba': np.concatenate(fold_y_pred_prob, axis=0)  # Renomeado para manter consistência
        })
        
        print(f"[INFO] Testando k={k_value}... Acurácia média: {mean_accuracy:.4f} | AUC: {mean_auc:.4f} | F1-Score: {mean_f1:.4f} | Top-{k_value} Accuracy: {mean_top_k_accuracy:.4f}")
        
        if mean_accuracy > best_overall_accuracy:
            best_overall_accuracy = mean_accuracy
            best_k = k_value
            best_scores = np.array(fold_scores)  # Adicionado para manter consistência
    
    print(f"[INFO] Melhor valor de k: {best_k}")
    print(f"\nCross-Validation Results for Cosine KNN:")
    print(f"Mean Accuracy: {best_scores.mean():.4f} (±{best_scores.std():.4f})")
    
    # Avaliação final com o melhor k
    final_predictions = []
    final_true_labels = []
    final_auc_scores = []
    final_f1_scores = []
    final_top_k_accuracies = []
    final_y_pred_prob = []

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(X_normalized):
        X_train_fold, X_test_fold = X_normalized[train_idx], X_normalized[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        
        knn = KNeighborsClassifier(n_neighbors=best_k, metric='cosine', weights='distance')
        knn.fit(X_train_fold, y_train_fold)
        y_pred = knn.predict(X_test_fold)
        y_pred_prob = knn.predict_proba(X_test_fold)
        
        auc, f1, top_k_accuracy = calculate_metrics(y_test_fold, y_pred, y_pred_prob, best_k)
        
        final_predictions.extend(y_pred)
        final_true_labels.extend(y_test_fold)
        final_auc_scores.append(auc)
        final_f1_scores.append(f1)
        final_top_k_accuracies.append(top_k_accuracy)
        final_y_pred_prob.append(y_pred_prob)

    print(f"\n[INFO] KNN Classification Report (Cosine, K={best_k}):")
    print(classification_report(final_true_labels, final_predictions))
    print(f"Accuracy: {accuracy_score(final_true_labels, final_predictions):.4f}")
    print(f"AUC: {np.mean(final_auc_scores):.4f}")
    print(f"F1-Score: {np.mean(final_f1_scores):.4f}")
    print(f"Top-{best_k} Accuracy: {np.mean(final_top_k_accuracies):.4f}")
    
    return {
        "best_k": best_k,
        "best_accuracy": best_overall_accuracy,
        "best_scores": best_scores,  # Adicionado para manter consistência
        "final_auc": np.mean(final_auc_scores),
        "final_f1_score": np.mean(final_f1_scores),
        "final_top_k_accuracy": np.mean(final_top_k_accuracies),
        "metrics_per_k": metrics_per_k,
        "final_y_pred_proba": np.concatenate(final_y_pred_prob, axis=0)
    }




def plot_roc_curve(metrics_per_k, y_test, output_dir="./", syndrome_mapping=None):
    """
    Plota a curva ROC para o melhor modelo.
    
    Parameters:
    -----------
    metrics_per_k : list of dict
        Lista com métricas para cada k, incluindo as probabilidades preditas
    y_test : array-like
        Labels verdadeiras
    output_dir : str
        Diretório onde salvar o plot
    syndrome_mapping : dict
        Dicionário que mapeia `syndrome_id` para seus respectivos nomes. 
        Exemplo: {0: 'Sindrome A', 1: 'Sindrome B'}
    """
    plt.figure(figsize=(10, 8))
    
    # Encontrar o melhor k (aquele com maior AUC)
    best_k_idx = np.argmax([m['auc'] for m in metrics_per_k])
    best_k_metrics = metrics_per_k[best_k_idx]
    
    # Pegar as probabilidades preditas do melhor modelo
    y_pred_proba = best_k_metrics['y_pred_proba']
    
    # Converter y_test para formato one-hot
    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)
    
    # Se tivermos apenas duas classes, ajustar o formato
    if y_test_bin.shape[1] == 1:
        y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])
    
    # Garantir que temos um mapeamento válido
    if syndrome_mapping is None:
        syndrome_mapping = {i: f'Class {i}' for i in range(y_test_bin.shape[1])}
    
    # Plotar curva ROC para cada classe
    for i in range(y_test_bin.shape[1]):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        class_name = syndrome_mapping.get(i, f'Class {i}')  # Usar o nome mapeado
        plt.plot(fpr, tpr, lw=2,
                label=f'ROC curve ({class_name}) (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # Salvar o plot
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()



def knn_classification(df: pd.DataFrame, metric: str = 'euclidean'):
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
        # Chamando a função knn_euclidean para a métrica Euclidiana
        results = knn_euclidean(X_train, X_test, y_train, y_test)
        
        # Acessando as métricas do dicionário retornado
        best_k = results["best_k"]
        best_accuracy = results["best_accuracy"]
        best_scores = results["best_scores"]
        final_auc = results["final_auc"]
        final_f1_score = results["final_f1_score"]
        final_top_k_accuracy = results["final_top_k_accuracy"]
        metrics_per_k = results["metrics_per_k"]  # Acessando as métricas por k
        
        # Imprimindo as métricas
        print(f"\nCross-Validation Results for Euclidean KNN:")
        print(f"Best k: {best_k}")
        print(f"Best Mean Accuracy: {best_accuracy:.4f}")
        print(f"Cross-validation scores (mean ± std): {best_scores.mean():.4f} ± {best_scores.std():.4f}")
        print(f"AUC: {final_auc:.4f}")
        print(f"F1-Score: {final_f1_score:.4f}")
        print(f"Top-{best_k} Accuracy: {final_top_k_accuracy:.4f}")
        
        # Plotando a curva ROC e salvando a imagem
        plot_roc_curve(metrics_per_k, y_test, output_dir="./")  # Salvar no diretório atual
        
    elif metric == 'cosine':
        # Chamando a função knn_cosine_cv
        results = knn_cosine_cv(X_train, y_train)

        # Acessando as métricas do dicionário retornado
        best_k = results["best_k"]
        best_accuracy = results["best_accuracy"]
        best_scores = results["best_scores"]
        final_auc = results["final_auc"]
        final_f1_score = results["final_f1_score"]
        final_top_k_accuracy = results["final_top_k_accuracy"]
        metrics_per_k = results["metrics_per_k"]

        # Imprimindo as métricas
        print(f"\nCross-Validation Results for Cosine KNN:")
        print(f"Best k: {best_k}")
        print(f"Best Mean Accuracy: {best_accuracy:.4f}")
        print(f"Cross-validation scores (mean ± std): {best_scores.mean():.4f} ± {best_scores.std():.4f}")
        print(f"AUC: {final_auc:.4f}")
        print(f"F1-Score: {final_f1_score:.4f}")
        print(f"Top-{best_k} Accuracy: {final_top_k_accuracy:.4f}")

        # Garantir que temos as predições finais apenas
        metrics_for_plot = [{
            'k': best_k,
            'auc': final_auc,
            'y_pred_proba': results['final_y_pred_proba']
        }]

        # Plotando a curva ROC apenas com as predições finais
        plot_roc_curve(metrics_for_plot, y_test, output_dir="./")

    else:
        print(f"[ERROR] Métrica desconhecida: {metric}")





if __name__ == "__main__":
    from data_processing import DataProcessor  
    pickle_path = "/media/paulo-jaka/Extras/DesafiosML/mini_gm_public_v0.1.p"
    processor = DataProcessor(pickle_path)
    df = processor.load_and_flatten_data()

    if df is not None:
        print("\n[INFO] Running KNN with Euclidean Distance...")
        knn_classification(df, metric='euclidean')

        print("\n[INFO] Running KNN with Cosine Distance...")
        knn_classification(df, metric='cosine')
