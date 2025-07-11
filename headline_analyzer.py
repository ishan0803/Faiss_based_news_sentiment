import numpy as np
import pandas as pd
import faiss
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
import xgboost as xgb

# Load FinBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModel.from_pretrained("yiyanghkust/finbert-tone")
model.eval()

def get_finbert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

def predict_delta_from_input_text(input_text: str, df: pd.DataFrame) -> float:
    # Step 1: Create embeddings for all titles
    embeddings = np.array([get_finbert_embedding(title) for title in df["Title"]])
    faiss.normalize_L2(embeddings)
    pca = PCA(n_components=3)
    pca_train = pca.fit_transform(embeddings)
    norm_train = np.linalg.norm(embeddings, axis=1)

    # Step 2: Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Step 3: Extract deltas
    percentage_diff = np.array(df["Percentage_Diff"].astype(float))

    # Step 4: Build training dataset from FAISS + deltas
    features = []
    targets = []

    for i in range(len(embeddings)):
        query_vec = embeddings[i].reshape(1, -1)
        D, I = index.search(query_vec, 6)  # Top 6, first is itself

        distances = D[0][1:]
        indices = I[0][1:]
        deltas = percentage_diff[indices]

        f1, f2 = distances[0], distances[1]
        d1, d2 = deltas[0], deltas[1]
        weighted_avg = np.sum(distances * deltas) / 5

        features.append([f1, f2, d1, d2, weighted_avg, pca_train[i][0], pca_train[i][1], pca_train[i][2], norm_train[i]])
        targets.append(percentage_diff[i]) 

    train_df = pd.DataFrame(features, columns=["f1", "f2", "d1", "d2", "weighted_avg", "pca_1", "pca_2", "pca_3", "norm"])
    train_df["target"] = targets

    # Step 5: Train XGBoost
    model_xgb = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
    model_xgb.fit(train_df[["f1", "f2", "d1", "d2", "weighted_avg", "pca_1", "pca_2", "pca_3", "norm"]], train_df["target"])

    # Step 6: Predict for input_text
    input_vec = get_finbert_embedding(input_text).reshape(1, -1)
    pca_test = pca.transform(input_vec)[0]
    norm_test = float(np.linalg.norm(input_vec))
    faiss.normalize_L2(input_vec)

    D, I = index.search(input_vec, 5)
    distances = D[0]
    indices = I[0]
    deltas = percentage_diff[indices]

    f1, f2 = distances[0], distances[1]
    d1, d2 = deltas[0], deltas[1]
    weighted_avg = np.sum(distances * deltas) / 5

    input_features = np.array([f1, f2, d1, d2, weighted_avg, pca_test[0], pca_test[1], pca_test[2], norm_test])
    input_features = input_features.reshape(1, -1).astype(np.float32)
    predicted_delta = model_xgb.predict(input_features)[0]

    return predicted_delta

# Debug/Test main function
def main():
    csv_path = "INFY_NS_predictions.csv"  # Replace with your CSV path
    df = pd.read_csv(csv_path)
    headline = "Infosys to acquire a new company in the tech sector."
    predicted = predict_delta_from_input_text(headline, df)
    print(f"Predicted Percentage Difference for input: {predicted:.2f}%")

if __name__ == "__main__":
    main()