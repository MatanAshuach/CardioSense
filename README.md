### *CardioSense*

### *ECG Arrhythmia Detection via Multi-Branch Transformer Fusion*

<img width="576" height="448" alt="CardioSense_Picture" src="https://github.com/user-attachments/assets/acadfed6-448f-40b7-a029-85e9560ecbec" />

---

### Project Overview

CardioSense is a deep learning project focused on detecting arrhythmias from electrocardiogram (ECG) signals. Using the **ECG5000 dataset**, we transform the original 5-class arrhythmia classification into a **binary classification** problem: **normal vs. anomalous heartbeat**.

The project experiments with **three model architectures**:
1. **Baseline Transformer** – a standard sequence-to-sequence model for ECG classification.  
2. **Dual-Branch Model** – introduces two parallel branches for multi-channel ECG signal processing.  
3. **Tri-Branch Model** – extends the dual-branch concept with an additional fusion branch for richer feature representation.  

Our **goal** is not only to compare these architectures under fair experimental settings, but also to **design improved architectures beyond the "simple" baseline Transformer**. We take into consideration how ECG data behaves and the characteristics it holds:
- **Temporal nature** – ECG signals are time-series with dependencies across short and long intervals.  
- **Biological structure** – heartbeats have repeating waveforms (P-QRS-T complex), which may be better captured by specialized branches.  
- **Class imbalance and anomalies** – arrhythmias are rare compared to normal beats, making sampling strategies critical.  

By integrating these insights, we aim to demonstrate that adapting model architectures to the **temporal and biological properties of ECG data** can improve classification performance and generalization compared to a plain baseline Transformer.

---
### Background
Cardiovascular diseases remain the leading cause of death globally, with arrhythmias representing a critical subset that can lead to sudden cardiac death if undetected. Traditional ECG monitoring relies heavily on manual interpretation by cardiologists, creating bottlenecks in clinical workflows and potential for human error in life-critical diagnoses. CardioSense addresses this challenge by exploring how vanilla transformer architectures can be enhanced for biological signal processing, specifically targeting the unique temporal-biological characteristics inherent in ECG signals. Unlike text or speech, ECG waveforms exhibit multi-scale temporal dependencies (both short-term P-QRS-T complexes and long-term rhythmic variations), hierarchical biological structure with clear anatomical correlates, and non-stationary dynamics that require adaptive feature extraction.
Our multi-branch approach was inspired by recent SOTA developments.

We designed three complementary branches: a Transformer branch for global context modeling across heartbeat sequences, a TCN branch with dilated convolutions for local morphological feature extraction and a BiLSTM branch for sequential pattern recognition between consecutive heartbeats (motivated by recent CNN-BiLSTM hybrid models). Our key innovation lies in the learnable softmax-gated fusion mechanism that dynamically weights each branch's contribution, combined with WeightedRandomSampler to address clinical class imbalance, creating a biologically-informed architecture that tailors transformer capabilities to cardiac signal analysis.

---
## Repository Structure

- CardioSense.ipynb - Main Jupyter notebook with the full pipeline

- README.md - Documentation (this file)
---

## Dataset Setup

To run the project, you will need to manually download the ECG5000 dataset and upload it to your environment, or you can use the dataset already included in the repository.

1. **Download the ECG5000 dataset** from [Time Series Classification Archive - ECG5000 Dataset](https://www.timeseriesclassification.com/description.php?Dataset=ECG5000).
   
2. Upload the dataset to your Google Drive in the directory: `/content/drive/MyDrive/Deep_Learning/datasets` (this is the default path in the code).

3. If you choose to use a different directory in your Google Drive, make sure to update the directory path in the code to reflect the change.

The rest of the code will automatically use the ECG5000 dataset once it's placed in the correct directory.

--- 

# Model Architectures

### 1. Baseline Transformer

- **Input processing:** A Conv1d stem projects input ECG signals into `d_model` dimensions.  
- **Core layers:** Positional encoding + stacked Transformer encoder layers (`num_layers`, `nhead`).  
- **Pooling:** Attention pooling compresses the sequence into a single vector.  
- **Classifier:** Linear → RELU → Linear for final logits. (changed to GELU in ensemble models)
- **Rationale:** Captures long-range dependencies in ECG sequences, establishing a clean baseline.  

---

### 2. Dual-Branch Model (Transformer + TCN)

- **Branches:**  
  - **Transformer Branch:** As above, models long-range dependencies.  
  - **TCN Branch:** Temporal Convolutional Network with dilated convolutions, residual blocks, and attention pooling. Captures local waveform shapes (e.g., QRS complex).  

- **Fusion:**  
  - Outputs from both branches are normalized.  
  - A **softmax-gated fusion module** learns feature-wise weights for each branch.  
  - Final representation is a convex combination of Transformer and TCN outputs.  

- **Classifier:** Shared linear head on fused features.  
- **Rationale:** ECG has both **local morphology** (better captured by TCN) and **global context** (better captured by Transformer). The gate lets the model learn per-feature weighting between them.  

---

### 3. Tri-Branch Model (Transformer + TCN + BiLSTM)

- **Branches:**  
  - **Transformer Branch:** Same as before.  
  - **TCN Branch:** Same as before.  
  - **BiLSTM Branch:** A bidirectional LSTM captures sequential heartbeat dependencies. Output is projected to `d_model` and pooled with attention.  

- **Fusion:**  
  - All three branches are normalized.  
  - A **softmax-gated tri-branch fusion** learns per-feature weights for each of the three feature vectors.  
  - Produces a fused representation balancing convolutional, attentional, and recurrent perspectives.  

- **Classifier:** Same as dual-branch head.  
- **Rationale:** Adds **sequential recurrence** to complement convolution and attention, yielding a richer feature set that mirrors ECG’s temporal-biological structure.  
---

## Hyperparameters

The following hyperparameters are configurable in the notebook:

| Hyperparameter | Description                                       |
| -------------- | ------------------------------------------------- |
| `d_model`      | Embedding dimension of the Transformer model      |
| `nhead`        | Number of attention heads in multi-head attention |
| `num_layers`   | Number of stacked Transformer encoder layers      |
| `epochs`       | Number of training epochs                         |
| `base_lr`      | Initial learning rate for the optimizer           |
| `batch_size`   | Number of samples per gradient update             |


## Dataset and Data Processing

 ECG5000 Dataset

- Collected from the **BIDMC Congestive Heart Failure Database**.  
- Contains 5000 heartbeat samples categorized into 5 classes:  
  - Normal (N)  
  - R-on-T Premature Ventricular Contraction (Ron-T PVC)  
  - Premature Ventricular Contraction (PVC)  
  - Supraventricular Premature or Ectopic Beat (SP/EB)  
  - Unclassified Beat (UB)  
- For this project: transformed into **binary classification** (normal vs. anomaly).

---

### Data Splitting

- Used **StratifiedShuffleSplit** to ensure class balance.  
- Split ratio: **60% training / 20% validation / 20% testing**.  
- Guarantees fair comparison between models.

---

### Handling Class Imbalance

- **WeightedRandomSampler** assigns higher weights to minority class samples.  
- Ensures the model sees a balanced distribution during training, preventing bias toward majority class.

---
## Conclusion

The results from the different models confirm that integrating multiple architectures, especially those that capture both local and global features, significantly improves ECG arrhythmia detection performance. Here’s a breakdown of the key models and why the results make sense in the context of their design:

### Baseline Transformer 
The **Baseline Transformer** performed well, achieving an accuracy of **99.10%**. This result is expected given that the Transformer model is excellent at capturing long-range temporal dependencies in time-series data, which is important for detecting arrhythmias. However, while the Transformer can learn global patterns over time, it may not capture the local intricacies of ECG signals (like the P-QRS-T waveform), which are vital for accurate arrhythmia detection. This limitation likely explains why the **accuracy is slightly lower** compared to the more advanced architectures.

### Dual-Branch Model 
The **Dual-Branch Model**, which combines the Transformer with a **Temporal Convolutional Network (TCN)**, outperformed the baseline Transformer with an accuracy of **99.80%**. The **TCN branch** helps capture **local temporal features** like the QRS complex, which is crucial in ECG signals. The TCN uses dilated convolutions to efficiently process the data and capture local dependencies, which is why we see an improvement over the baseline model. The fusion of the **Transformer's global context modeling** and **TCN's local pattern recognition** enabled the model to achieve **better performance**, especially in terms of capturing arrhythmia signals that depend on both long-term and short-term temporal patterns.

### Tri-Branch Model
The **Tri-Branch Model**, which integrates **Transformer**, **TCN**, and **BiLSTM**, achieved the **highest performance** with an accuracy of **99.60%,** though it's slightly lower than the Dual-Branch’s performance. This model benefits from the BiLSTM branch, which captures **sequential dependencies** from both past and future states in ECG signals. The BiLSTM enables the model to **better understand the sequential nature of heartbeats**, which is crucial for arrhythmia detection. Furthermore, the **softmax-gated fusion mechanism** that combines the three branches ensures that the model can dynamically choose the best representation from each of the branches, making it more adaptable and robust for various types of ECG signal patterns.

The reason for the **Tri-Branch Model's highest ranking** is due to our **method of selecting the best model based on multiple evaluation criteria**, prioritizing **PR-AUC (threshold-free)**, then **F1** (thresholded at tau), **ROC-AUC**, and finally **Balanced Accuracy**. By using **PR-AUC** as the primary metric, we focus on the model's ability to correctly identify minority class instances, which is critical in imbalanced classification tasks like arrhythmia detection. The **Tri-Branch Model** consistently outperformed the other models on **PR-AUC**, which is why it ranks the highest, especially in terms of **robustness to new, unseen data**.

### Why These Results Make Sense:
- **Dual-Branch's success** is attributed to the **combination of TCN and Transformer**. The TCN is effective in capturing local features, while the Transformer adds the ability to model long-range dependencies, making it well-suited for ECG signals.
- The **Tri-Branch model** incorporates **BiLSTM**, which improves performance by capturing **sequential and temporal relationships** that are not as easily learned by Transformers or TCNs alone. The slight difference in performance with the Dual-Branch model suggests that the BiLSTM’s impact might be more beneficial when combined with more complex datasets or longer sequences. The **fusion mechanism** in the Tri-Branch model allows the model to dynamically prioritize which branch (TCN, Transformer, or BiLSTM) to focus on, enhancing the overall performance and adaptability of the model.
- The **baseline Transformer**, while strong, does not account for the **local temporal intricacies** present in ECG signals, which limits its ability to capture critical features like the QRS complex, leading to slightly lower accuracy compared to the other models.

### Final Thoughts:
Overall, the results show that **multi-branch models** (Dual-Branch and Tri-Branch) offer significant improvements over the simple Transformer by effectively combining multiple strategies to capture different aspects of ECG data. The **Tri-Branch Model**, in particular, demonstrates how combining **local feature extraction (TCN)**, **global context modeling (Transformer)**, and **sequential pattern learning (BiLSTM)** can lead to a more robust and accurate model for arrhythmia detection. Furthermore, the **Tri-Branch Model's performance** is the best due to our **multi-metric ranking policy**, which emphasizes **PR-AUC** as the most important criterion for model selection, ensuring robust performance across different thresholds and minimizing the impact of class imbalance. These results validate the effectiveness of this hybrid approach in improving ECG classification tasks.


---

## Academic Inspirations

Our project draws inspiration from several key academic articles in the field of ECG classification using deep learning methods. Below are some relevant papers:

**Bi S, Lu R, Xu Q, Zhang P.** Accurate Arrhythmia Classification with Multi-Branch, Multi-Head Attention Temporal Convolutional Networks. Sensors (Basel). 2024 Dec 19;24(24):8124. doi: 10.3390/s24248124. PMID: 39771858; PMCID: PMC11679161.  
This paper proposes a method very similar to our dual-branch approach, using multi-branch architectures with temporal convolutional networks and attention mechanisms for ECG arrhythmia classification.

**Ingolfsson, T. M., Wang, X., Hersche, M., Burrello, A., Cavigelli, L., & Benini, L.** "ECG-TCN: Wearable Cardiac Arrhythmia Detection with a Temporal Convolutional Network," 2021 IEEE 3rd International Conference on Artificial Intelligence Circuits and Systems (AICAS), Washington DC, DC, USA, 2021, pp. 1-4, doi: 10.1109/AICAS51828.2021.9458520.  
This paper directly informs our TCN branch implementation for temporal processing of ECG signals, providing insights into real-time arrhythmia detection on wearable devices.

**Kim, D., Lee, K. R., Lim, D. S., Lee, K. H., Lee, J. S., Kim, D.-Y., & Sohn, C.-B.** "A novel hybrid CNN-transformer model for arrhythmia detection without R-peak identification using stockwell transform," Scientific Reports, 2025.  
This article is closely related to our Transformer-based approach, combining convolutional elements with attention mechanisms to enhance ECG arrhythmia detection.



