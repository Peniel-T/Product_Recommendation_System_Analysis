
# Recommender Systems with Event-Based Implicit Feedback

This project explores recommender system design using **user-item interactions** (events such as views, add-to-cart, and transactions). The pipeline moves from **data preprocessing** and **matrix factorization models** to **evaluation and diversification strategies**.

---

##  Key Concepts

* **Implicit Feedback**
  Unlike explicit ratings (e.g., 1–5 stars), we use user behavior as signals of preference.

  * `view → 1`
  * `add-to-cart → 2`
  * `transaction → 3`
    These are mapped into **numeric feedback levels** for training models.

* **Matrix Factorization (ALS)**

  * We initially used the **Implicit library’s Alternating Least Squares (ALS)** for recommendation.
  * Handles sparse matrices efficiently.
  * Learns latent representations of users and items.

* **Surprise Library for Explicit Models**

  * Converted implicit interactions into **pseudo-ratings** for use in **SVD/SVD++ from Surprise**.
  * Evaluated using **RMSE** and **MAE** to measure reconstruction accuracy.

* **Evaluation Metrics for Recommendations**

  * Beyond accuracy, we evaluate **Precision\@K** and **Recall\@K**.
  * Example result (unbalanced case):

    * Precision\@10 ≈ 0.00039
    * Recall\@10 ≈ 0.9997
  * Highlights challenges when one category dominates.

* **Data Sampling for Low Memory**

  * To handle large interaction datasets, we implemented **badge/random sampling** of `sample_merged`.
  * This allows model training and evaluation under **limited RAM**.

* **Diversification with MMR (Maximal Marginal Relevance)**

  * Implemented a **Simple MMR Diversification** approach using sampled interactions.
  * Reduces redundancy in recommendations without dense embeddings.

---

## Project Workflow

1. **Data Preprocessing**

   * Load and merge user-event interactions.
   * Map categorical events → numeric feedback (`view`, `addcart`, `transaction`).
   * Create **sparse user-item matrices**.

2. **Model Training**

   * **ALS (Implicit)** on sparse matrices.
   * **SVD (Surprise)** using pseudo-explicit ratings.

3. **Evaluation**

   * **Error metrics**: RMSE, MAE.
   * **Ranking metrics**: Precision\@K, Recall\@K.

4. **Sampling**

   * Low-memory training using **sampled subsets** of `sample_merged`.

5. **Diversification**

   * Simple MMR implementation on top recommendations.

---

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
* **Surprise** – SVD, model evaluation
* **Scikit-learn** – metrics, MMR diversification

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
* Implicit feedback requires **careful weighting** of interactions.
* Large datasets need **sampling strategies** for experimentation.
* Evaluation goes beyond RMSE → **ranking-based metrics matter more**.

---

This repo is a practical journey through **building scalable recommender systems** from event logs to recommendation evaluation.

