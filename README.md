
# Recommender Systems with Event-Based Implicit Feedback

This project explores recommender system design using **user-item interactions** (events such as views, add-to-cart, and transactions). The pipeline moves from **data preprocessing** and **matrix factorization models** to **evaluation and diversification strategies**.

---
##  Resources

- **Dataset (Google Drive)**  
  Access the interaction logs and item metadata here:  
  [📂 Download Dataset](https://drive.google.com/drive/folders/1zyKYpKSyGHJqmS_kL9aL4tx-6iIPEJTM?usp=sharing)

- **Run Experiments on Google Colab**  
  Click to open the interactive notebook:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17_ji6t3TEQDWIugKC7Epb1ojdjnyqAzv?usp=sharing)


---

##  Key Concepts

* **Implicit Feedback**
  Unlike explicit ratings (e.g., 1–5 stars), we use user behavior as signals of preference.

  * `view → 1`
  * `add-to-cart → 2`
  * `transaction → 3`
    These are mapped into **numeric feedback levels** for training models.

* **Matrix Factorization (ALS)**

  * Implicit library’s Alternating Least Squares (ALS) for recommendation.
  * Handles sparse matrices efficiently.
  * Learns latent representations of users and items.

* **Random Forest Classifier**
  *Predicts transaction likelihood from user-item features. SMOTE ensures     balanced training.

* **Anomaly Detection (Isolation Forest)**
  *Detects abnormal browsing/purchasing behavior clusters.

**Evaluation**

Ranking Metrics (Recommenders): Precision@K, Recall@K, NDCG.

* **Classification Metrics (Random Forest)**
*  Precision, Recall, F1, ROC-AUC.

Anomaly Scoring: Silhouette score for cluster separability, % of anomalies detected.

**Results Visualization**

* Heatmaps for classification metrics.

* Line plots for item availability trends.

Cluster summaries for anomaly groups.
---

## Project Workflow

1. **Data Preprocessing**

   * Load and merge user-event interactions.
   * Map categorical events → numeric feedback (`view`, `addcart`, `transaction`).
   * Create **sparse user-item matrices**.

2. **Model Training**

   * **Content-Based Filtering (CBF):** Leveraged item metadata (category, availability) for personalized recommendations.  
   * **Matrix Factorization (Implicit):** Used latent factors to model user–item interactions.  
   * **Random Forest Classifier:** Trained on item and user features to predict transaction likelihood.  
   * **Anomaly Detection (Isolation Forest):** Identified unusual visitor behavior patterns.  


3. **Evaluation**

   *  **Ranking metrics:** Precision@K, Recall@K, NDCG@K for recommendation models.
   *  **Classification metrics:** Accuracy, F1-score, and ROC-AUC for Random Forest Classifier.
   *   **Clustering metric:** Silhouette Score for anomaly detection.
     
4. **Scalability**

   * Trained on full dataset (~2.7M records) with sampling strategies for efficient evaluation.  
   *  Applied normalization, one-hot encoding, and SMOTE balancing to handle sparsity and class imbalance.
     
5. **Insights & Applications**

    * Recommendations improve personalization but require richer features for higher accuracy.  
    * Random Forest achieved near-perfect classification after SMOTE balancing.  
    * Anomaly detection revealed ~14K visitors with unusual interaction behavior.  


## Future Work

* Balance event categories (views vs. addcart vs. transaction).
* Compare with neural recommenders (LightFM, deep models).
* Implement user/item feature integration (content + collaborative).
* Explore contextual recommendations (time, device, session).

---

## Tech Stack

* **Python**
* **Pandas, Numpy, Scipy** – preprocessing & sparse matrices
* **Implicit** – ALS recommender

---

## Example Output

```
RMSE: 0.1341
MAE:  0.0190
Model evaluation complete.

Precision@10: 0.00039
Recall@10:    0.9997
```

---

## Learning Takeaways

* Recommender systems **must balance accuracy and diversity**.
* Large datasets need **sampling strategies** for experimentation.
* Evaluation goes beyond RMSE → **ranking-based metrics matter more**.

---

This repo is a practical journey through **building scalable recommender systems** from event logs to recommendation evaluation.


