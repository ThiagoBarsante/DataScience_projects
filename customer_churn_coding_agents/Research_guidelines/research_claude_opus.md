I'll give you a complete, opinionated overview — not just definitions, but when each metric lies, which one to choose depending on the type of problem, and how this changes between churn (moderate imbalance) and fraud (extreme imbalance).

First the foundation, because every binary classification metric is born from the confusion matrix: the confusion matrix is the origin of almost everything. Fix a decision threshold and you get four numbers — of these four, precision reads "vertically" (the column of what you predicted as positive) and recall reads "horizontally" (the row of what was actually positive).

Now the metrics in three layers — because the most common confusion is mixing metrics that depend on the threshold with those that evaluate the model's *ranking*.

## Layer 1 — metrics based on a fixed threshold

These require you to have already cut the probability into positive/negative (the default 0.5 is almost never right for imbalanced data).

**Accuracy** = (TP+TN)/total. This is the most misleading metric under imbalance — the *accuracy paradox*. In fraud with 0.2% positives, a model that predicts "everything legitimate" gets 99.8% accuracy and catches zero fraud. Only use accuracy when classes are reasonably balanced AND error costs are symmetric. In churn/fraud, practically never.

**Precision** (TP/(TP+FP)): of everything you flagged, how many were actually true. Governs operational cost — every FP is a wasted investigation or a legitimate customer blocked.

**Recall / Sensitivity / TPR** (TP/(TP+FN)): of the actual positives, how many you caught. Governs the loss from omission — every FN is a fraud that slipped through or a customer who churned without receiving an offer.

Precision and recall trade off against each other as the threshold moves: lower the cutoff and recall rises at the expense of precision, and vice versa. There's no "good model" without specifying which point on that curve you're operating at.

**F1** = harmonic mean 2·(P·R)/(P+R). The harmonic mean punishes imbalance — a high F1 requires both P and R to be high *simultaneously*. The problem: F1 assumes precision and recall carry equal weight, which is almost never true in your domain. Use **F-beta** = (1+β²)·(P·R)/(β²·P+R): β>1 (e.g., F2) prioritizes recall — appropriate in fraud/churn where missing the positive hurts more; β<1 (e.g., F0.5) prioritizes precision — when the false alarm is expensive.

**Balanced Accuracy** = (Recall + Specificity)/2. Average of per-class hit rates, immune to imbalance. Good as a sanity check, but still ignores the relative *cost* of errors.

**MCC (Matthews Correlation Coefficient)** = (TP·TN − FP·FN)/√((TP+FP)(TP+FN)(TN+FP)(TN+FN)). In my view this is the best *single* threshold-based metric for imbalanced data: it only stays high when all four cells of the matrix look good. It ranges from −1 to +1, and unlike F1, it takes TN into account. F1 can look decent while MCC reveals that the model is actually bad.

**Cohen's Kappa** = (p_o − p_e)/(1 − p_e): chance-corrected agreement. Useful, but MCC tends to be more informative and stable in this scenario.

## Layer 2 — threshold-independent metrics (evaluate the ranking)

These measure how well the model *ranks* cases by probability, without fixing a cutoff. They're the right ones for comparing and tuning models, because they separate model quality from the operational choice of threshold.

**ROC-AUC**: area under the TPR × FPR curve. Clean interpretation — the probability that a random positive gets a higher score than a random negative; baseline of 0.5. **The trap under imbalance**: FPR has TN in the denominator (FP/(FP+TN)). With millions of negatives, you can generate an avalanche of FPs that barely moves the FPR, so ROC-AUC ends up beautifully optimistic while precision is on the floor. In fraud, a ROC-AUC of 0.98 can coexist with a precision of 5%.

**PR-AUC / Average Precision**: area under Precision × Recall. This is the central point of your question — **in imbalanced data, the PR curve is the correct ranking metric**. Precision reacts directly to FPs without the dampening effect of TN, so PR-AUC "feels" what ROC hides. Crucial detail: the PR-AUC baseline isn't 0.5, it's the *prevalence* of the positive class (0.002 in fraud with 0.2%). A PR-AUC of 0.4 in that case is excellent; always interpret it against the baseline, never in absolute terms.

Rule of thumb: **ROC-AUC for comparing models when classes are close to balanced; PR-AUC when the positive class is rare and it's the class you actually care about.** Report both, but optimize for the latter in fraud.

**Log loss** = −(1/N)Σ[y·log(p)+(1−y)·log(1−p)] and **Brier score** = (1/N)Σ(p−y)²: measure the *quality of the probability*, not the ranking or the decision. These matter when the probability feeds into an expected-value calculation (e.g., P(churn)×CLV vs. the cost of the offer). A model can rank well (high AUC) and still be poorly calibrated — which is why you need **calibration** (Platt scaling or isotonic regression) and should check the *reliability diagram*. Warning: resampling and class weights *distort* the probabilities; if you use SMOTE/undersampling, recalibrate afterward.

## Layer 3 — business metrics (the ones that actually decide)

This is where serious consulting is separated from a Kaggle exercise. The metrics above are proxies; the real objective has a dollar sign attached.

- **Expected cost**: define the cost matrix (C_FP, C_FN) and choose the threshold that minimizes E[cost] = C_FP·FP + C_FN·FN. If you can quantify the costs, optimize this directly — it's the optimal decision.
- **Lift / cumulative gains**: in churn with a limited retention budget, rank by score and target the top deciles. Lift at decile k = (response rate in the top-k)/(overall rate). Communicates campaign ROI to business stakeholders better than any AUC.
- **Precision@k / Recall@k**: in fraud, the team investigates only N alerts per day. What matters is the precision in the top-N and how much recall you achieve within the alert budget — not the whole curve.
- **Value-weighted recall ($-weighted)**: catching a $50k fraud is worth more than catching ten $100 ones. Weight recall by transaction value; the same applies to CLV-weighted churn lift.

## Strategy for imbalanced data

The order I follow, from cheapest to most costly:

1. **Choose the metric first.** Before any model, define the optimization target based on business cost. This guides everything else. Never accuracy.
2. **Class weights** (`class_weight='balanced'` in sklearn, `scale_pos_weight` in XGBoost/LightGBM). This is the first resort: cheap, doesn't invent data, doesn't require repipelining. Often solves a good chunk of the problem on its own.
3. **Threshold tuning.** Probably the highest-return and most neglected step. Train the model, then sweep the threshold on the validation set to hit your objective (maximize F-beta, minimize expected cost, or fix a target like "precision ≥ 20% while maximizing recall"). The cutoff is a business decision, not a default.
4. **Resampling — with care.** Undersampling the majority class (fast, throws away data), oversampling the minority class (overfitting risk), or synthetic methods (SMOTE, Borderline-SMOTE, ADASYN; SMOTE-Tomek/SMOTE-ENN combos clean up borders). Traps that sink projects:
   - **Resample only within the training fold**, never before the CV split — otherwise information leaks and the metric inflates falsely.
   - **Always evaluate on the real distribution** (a test set that hasn't been resampled). The metric has to reflect production prevalence.
   - SMOTE is fragile with categorical features, high dimensionality, and mixed tabular data — it can generate unrealistic samples. In extreme fraud it often underperforms class weights + a well-chosen threshold.
   - Resampling breaks calibration; recalibrate if the probability matters.
5. **Cost-sensitive learning**: embed the cost matrix directly into the loss function when the algorithm allows it. More elegant than resampling.
6. **Anomaly framing** for extreme rarity (Isolation Forest, autoencoder) — sometimes useful for new/unlabeled fraud, generally combined with a supervised model.

## Validation and tuning strategy

- **Separate two decisions:** (a) model + hyperparameters and (b) threshold. Tune the model on a *ranking* metric (PR-AUC/average precision in fraud; ROC-AUC if nearly balanced), because it doesn't depend on the cutoff. Only afterward fix the threshold for the business objective. Tuning directly on F1@0.5 mixes the two together and leads to poor choices.
- **Stratified CV** to preserve class proportions in each fold. If there are multiple rows per customer, use a *group* split (GroupKFold) to avoid leaking the same customer between training and validation.
- **Out-of-time validation** is mandatory in fraud and highly recommended in churn: validate on a period *after* the training period. Fraud undergoes concept drift (patterns evolve), and random CV masks this by allowing the model to "see the future." The temporal test is what resembles production.
- **Nested CV** (or train/val/test with an out-of-time test) so hyperparameter selection doesn't contaminate the final evaluation set.
- **Avoid leakage in transformations**: fit scalers, encoders, feature selection, and resampling *inside* each fold (use `Pipeline`/`imblearn.Pipeline`), never on the whole dataset before the split.

## Applied: telecom churn vs. financial fraud

These are two quite different regimes, and treating them the same is a common mistake:

**Telecom churn** — moderate imbalance (typically ~15–30% churn). FN = lost customer (entire CLV); FP = wasted retention offer (discount/incentive — not free, but orders of magnitude smaller). Since it's not rare, ROC-AUC is still usable for ranking, but I'd report PR-AUC alongside it. Calibration matters a lot because the decision is economic: act if P(churn)×CLV > cost of the offer. Lift/gains for budget-constrained campaigns. And the step that separates a beginner from a senior practitioner: the problem *isn't* predicting churn, it's **uplift modeling** — modeling the treatment effect to target the "persuadables," not wasting the offer on people who'd stay anyway (which would erode margin) nor on people who'll leave regardless of what you do. The sophisticated metric here is Qini/uplift, not F1.

**Financial fraud** — extreme imbalance (0.1–1% or less), strongly asymmetric cost, and a hard operational constraint (finite investigation capacity) plus accelerated concept drift. Here: ROC-AUC is misleading, **PR-AUC/average precision is the ranking metric**; operations are measured by **precision@top-N** and **recall at a fixed FPR/alert budget**; the threshold comes from the fraud team's capacity and SLA, not abstract optimization. Weight by transaction value (expensive fraud counts more) and validate out-of-time with continuous monitoring and retraining, because the fraud pattern shifts underneath the model. FP isn't just an investigation cost: blocking a legitimate transaction erodes trust and customer experience, so that cost enters the calculation too.

The throughline: **the metric isn't a technical choice, it's the translation of business cost.** Define C_FP and C_FN first, let them dictate the optimization metric and the threshold, validate on the real distribution and in real time, and report to stakeholders in terms of lift/cost avoided, not AUC.

If you'd like, I can put together a reference notebook with the full `imblearn` pipeline (stratified, leakage-free, expected-cost threshold tuning, and calibration) for both churn and fraud — let me know if you'd prefer plain scikit-learn, XGBoost/LightGBM, or something already running distributed on Spark/Databricks given your stack.
