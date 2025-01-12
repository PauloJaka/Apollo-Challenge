import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score,KFold  # type: ignore
from sklearn.preprocessing import  MinMaxScaler , LabelBinarizer, label_binarize # type: ignore
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
    
    metrics_per_k = []  
    aucs = [] 
    f1_scores = [] 
    top_k_accuracies = [] 
    
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
        
        # Att
        aucs.append(auc if auc is not None else 0)
        f1_scores.append(f1)
        top_k_accuracies.append(top_k_acc if top_k_acc is not None else 0)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k_value
            best_scores = scores
        
        # Saving the K metrics
        metrics_per_k.append({
            'k': k_value,
            'accuracy': accuracy,
            'auc': auc,
            'f1_score': f1,
            'top_k_accuracy': top_k_acc,
            'y_pred_proba': y_pred_proba  
        })

    print(f"[INFO] Melhor valor de k: {best_k}")
    print(f"\nCross-Validation Results for Euclidean KNN:")
    print(f"Mean Accuracy: {best_scores.mean():.4f} (±{best_scores.std():.4f})")
    
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
        "best_accuracy": best_accuracy,       
        "best_scores": best_scores,            
        "final_auc": auc,                     
        "final_f1_score": f1,                
        "final_top_k_accuracy": top_k_acc,    
        "metrics_per_k": metrics_per_k,       
        "final_y_pred_proba": y_pred_proba   
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
            'accuracy': mean_accuracy, 
            'auc': mean_auc,
            'f1_score': mean_f1, 
            'top_k_accuracy': mean_top_k_accuracy,
            'y_pred_proba': np.concatenate(fold_y_pred_prob, axis=0) 
        })
        
        print(f"[INFO] Testando k={k_value}... Acurácia média: {mean_accuracy:.4f} | AUC: {mean_auc:.4f} | F1-Score: {mean_f1:.4f} | Top-{k_value} Accuracy: {mean_top_k_accuracy:.4f}")
        
        if mean_accuracy > best_overall_accuracy:
            best_overall_accuracy = mean_accuracy
            best_k = k_value
            best_scores = np.array(fold_scores)  # Better cosisntesy
    
    print(f"[INFO] Melhor valor de k: {best_k}")
    print(f"\nCross-Validation Results for Cosine KNN:")
    print(f"Mean Accuracy: {best_scores.mean():.4f} (±{best_scores.std():.4f})")
    
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
        "best_scores": best_scores, 
        "final_auc": np.mean(final_auc_scores), 
        "final_f1_score": np.mean(final_f1_scores),
        "final_top_k_accuracy": np.mean(final_top_k_accuracies),
        "metrics_per_k": metrics_per_k,
        "final_y_pred_proba": np.concatenate(final_y_pred_prob, axis=0)
    }

def align_arrays(y_test_bin, y_pred_proba):
    min_samples = min(y_test_bin.shape[0], y_pred_proba.shape[0])
    y_test_bin_aligned = y_test_bin[:min_samples]
    y_pred_proba_aligned = y_pred_proba[:min_samples]
    
    return y_test_bin_aligned, y_pred_proba_aligned

def plot_roc_curve(metrics_per_k, y_test, output_dir="./", syndrome_mapping=None, file_name='ROC_curve'):

    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)
    if y_test_bin.shape[1] == 1:
        y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])
    
    if syndrome_mapping is None:
        syndrome_mapping = {i: f'Class {i}' for i in range(y_test_bin.shape[1])}
    
    n_classes = y_test_bin.shape[1]
    colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
    
    for metric_data in metrics_per_k:
        metric_name = metric_data['metric']
        y_pred_proba = metric_data['y_pred_proba']
        
        plt.figure(figsize=(10, 8))
        
        def align_arrays(arr1, arr2):
            min_length = min(len(arr1), len(arr2))
            return arr1[:min_length], arr2[:min_length]
        
        y_test_bin_aligned, y_pred_proba = align_arrays(y_test_bin, y_pred_proba)
        
        for i in range(y_test_bin_aligned.shape[1]):
            fpr, tpr, _ = roc_curve(y_test_bin_aligned[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            syndrome_id = lb.classes_[i]
            
            plt.plot(
                fpr, tpr, 
                lw=2,
                color=colors[i],
                label=f'syndrome {syndrome_id} (AUC = {roc_auc:.2f})'
            )
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {metric_name}')
        plt.legend(loc="lower right")
        
        plt.savefig(os.path.join(output_dir, f'roc_curve_{metric_name}_{file_name}.png'))
        plt.close()
    
    plt.figure(figsize=(12, 8))
    
    linestyles = {'Euclidean': '-', 'Cosine': '--'}
    
    for metric_data in metrics_per_k:
        metric_name = metric_data['metric']
        y_pred_proba = metric_data['y_pred_proba']
        
        y_test_bin_aligned, y_pred_proba = align_arrays(y_test_bin, y_pred_proba)
        
        for i in range(y_test_bin_aligned.shape[1]):
            fpr, tpr, _ = roc_curve(y_test_bin_aligned[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            syndrome_id = lb.classes_[i]
            
            plt.plot(
                fpr, tpr, 
                lw=2,
                color=colors[i],
                linestyle=linestyles[metric_name],
                label=f'{metric_name} - syndrome {syndrome_id} (AUC = {roc_auc:.2f})'
            )
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison - Euclidean vs Cosine')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f'roc_curve_comparison_{file_name}.png'))
    plt.close()

def summarize_metrics(metrics_per_k, output_dir="./", rank_by="accuracy", file_name="Resume_data"):
    
    metrics_df = pd.DataFrame(metrics_per_k)
    
    if 'syndrome_id' not in metrics_df.columns:
        print("[WARNING] 'syndrome_id' não encontrado nas métricas. A coluna será preenchida com NaN.")
        metrics_df['syndrome_id'] = None 
    
    metrics_df = metrics_df.sort_values(by=rank_by, ascending=False).reset_index(drop=True)
    
    metrics_df["rank"] = metrics_df.index + 1  
    
    if output_dir:
        output_path = f"{output_dir}/{file_name}.csv"
        metrics_df.to_csv(output_path, index=False)
        print(f"[INFO] Métricas salvas em: {output_path}")
    
    print("\nResumo das Métricas:")
    print(metrics_df.head())
    
    return metrics_df

def knn_classification(df: pd.DataFrame):
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

    def process_results(results, metric_name):
        best_k = results["best_k"]
        best_accuracy = results["best_accuracy"]
        best_scores = results["best_scores"]
        final_auc = results["final_auc"]
        final_f1_score = results["final_f1_score"]
        final_top_k_accuracy = results["final_top_k_accuracy"]
        metrics_per_k = results["metrics_per_k"]
        
        print(f"\nCross-Validation Results for {metric_name} KNN:")
        print(f"Best k: {best_k}")
        print(f"Best Mean Accuracy: {best_accuracy:.4f}")
        print(f"Cross-validation scores (mean ± std): {best_scores.mean():.4f} ± {best_scores.std():.4f}")
        print(f"AUC: {final_auc:.4f}")
        print(f"F1-Score: {final_f1_score:.4f}")
        print(f"Top-{best_k} Accuracy: {final_top_k_accuracy:.4f}")
        
        metrics_df = pd.DataFrame(metrics_per_k)
        metrics_df['metric'] = metric_name
        metrics_df['y_pred_proba'] = [results['final_y_pred_proba']] * len(metrics_df)
        
        for i, row in metrics_df.iterrows():
            metrics_df.at[i, 'syndrome_id'] = y_test.values[i]
        
        return metrics_df, {
            'metric': metric_name,
            'best_k': best_k,
            'auc': final_auc,
            'y_pred_proba': results['final_y_pred_proba']
        }

    values_for_plot = []
    
    results_euclidean = knn_euclidean(X_train, X_test, y_train, y_test)
    metrics_df_euclidean, plot_data_euclidean = process_results(results_euclidean, "Euclidean")
    values_for_plot.append(plot_data_euclidean)
    
    results_cosine = knn_cosine_cv(X_train, y_train)
    metrics_df_cosine, plot_data_cosine = process_results(results_cosine, "Cosine")
    values_for_plot.append(plot_data_cosine)
    
    all_metrics_df = pd.concat([metrics_df_euclidean, metrics_df_cosine], ignore_index=True)
    
    euclidean_filename = "euclidean_resume"
    cosine_filename = "cosine_resume"
    
    summarize_metrics(
        metrics_df_euclidean, 
        output_dir="./", 
        rank_by="accuracy", 
        file_name=euclidean_filename
    )
    
    summarize_metrics(
        metrics_df_cosine, 
        output_dir="./", 
        rank_by="accuracy", 
        file_name=cosine_filename
    )
    
    plot_roc_curve(values_for_plot, y_test, output_dir="./", file_name="combined")
    
    return all_metrics_df


def plot_comparison_roc(metrics_per_k, y_test, output_dir="./", file_name='ROC_comparison_curve'):
    plt.figure(figsize=(10, 8))
    
    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)
    
    if y_test_bin.shape[1] == 1:
        y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])
    
    def align_arrays(arr1, arr2):
        min_length = min(len(arr1), len(arr2))
        return arr1[:min_length], arr2[:min_length]

    for model_data in metrics_per_k:
        y_pred_proba = model_data['y_pred_proba']
        y_test_bin, y_pred_proba = align_arrays(y_test_bin, y_pred_proba)
        
        for i in range(y_test_bin.shape[1]):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            syndrome_id = lb.classes_[i]
            metric_name = model_data['metric']
            plt.plot(fpr, tpr, lw=2,
                     label=f'{metric_name} (Class {syndrome_id}) (AUC = {roc_auc:.2f})')  

    plt.plot([0, 1], [0, 1], 'k--', lw=2)  
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    plt.savefig(os.path.join(output_dir, f'{file_name}.png'))
    plt.close()


if __name__ == "__main__":
    from data_processing import DataProcessor  
    pickle_path = "/media/paulo-jaka/Extras/DesafiosML/mini_gm_public_v0.1.p"
    processor = DataProcessor(pickle_path)
    df = processor.load_and_flatten_data()

    if df is not None:
        knn_classification(df)