# Research guidelines — classification metrics under imbalance

Background research behind the metric choices in the churn project. One brief was sent to **two different research agents**, independently, so the reasoning could be cross-checked instead of trusted.

## The brief

Verbatim, as sent — [`_prompt_for_GuideLines_classification metrics—such as recall.txt`](./_prompt_for_GuideLines_classification%20metrics%E2%80%94such%20as%20recall.txt):

> Explain classification metrics—such as **recall**, precision, ROC-AUC, and accuracy—in the context of **imbalanced data**, and discuss the best strategies for **evaluating and tuning** machine learning models for use cases like **telecom churn** and **financial fraud detection**.

## The two answers

| Report | Agent | How it is organised |
|---|---|---|
| [`deep-research-report_gptweb.md`](./deep-research-report_gptweb.md) | ChatGPT (deep research) | Executive summary → classic metrics → limits and trade-offs under imbalance → when to use each metric → evaluation & tuning strategies → churn vs. fraud recommendations → comparative table |
| [`research_claude_opus.md`](./research_claude_opus.md) | Claude Opus (deep research) | Layered, from the confusion matrix up: fixed-threshold metrics → threshold-independent metrics (ranking) → business metrics → imbalance strategy → validation & tuning → churn vs. fraud applied |

Same brief, two structures: one surveys the metrics and then applies them, the other builds from the confusion matrix toward the business decision. Reading both is the point — where they agree independently is where the guidance is safe to act on.

## Where they agree

- **Accuracy is the wrong headline under imbalance.** At a 26.5% churn rate, "nobody churns" scores 73.5% and saves no one.
- **Recall is never reported alone.** It only means something against a stated cost — quote it at a fixed alert budget (precision@top-N, or recall at a set false-positive rate). Flag everyone and recall hits 100%.
- **The decision threshold comes from capacity, not optimisation.** How many customers the retention team can call sets the cut-off; 0.5 is a library default.
- **F1 assumes precision and recall matter equally** — rarely true. Use **F-beta** (F2) when a missed customer costs more than a wasted offer.
- **Under heavier imbalance, PR-AUC and calibration beat ROC-AUC**, and resampling (SMOTE) is over-prescribed.
- **Churn and fraud are different problems.** Churn is moderate imbalance and an economic call — act when *P(churn) × customer lifetime value* > cost of the offer. Fraud is extreme imbalance with a hard capacity limit, so PR-AUC and precision@K govern.


