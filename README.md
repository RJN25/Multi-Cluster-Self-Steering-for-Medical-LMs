# Multi-Cluster Self-Steering with Confidence Calibration
*(Summary of Implementation for RunPod Execution)*

---

##  Core Concept

This experiment extends steering-vector methods by **clustering latent steering directions** and **learning to self-adjust hidden activations**.  
The key idea is that each input (question) yields a steering vector that reflects how hidden states shift from incorrect → correct reasoning.  
We then:

1. **Cluster** these steering vectors into groups (semantic reasoning modes).
2. Compute **mean steering vectors** per cluster → this becomes our **dictionary**.
3. Train a **projection matrix (P)** that maps new hidden activations to a steering direction resembling the right cluster.
4. During inference, we measure **cosine similarity** between projected activations and the dictionary means.
5. Weight these dictionary vectors to compute a *steering adjustment* that nudges activations toward confidence-aligned regions.

---

## Procedure Breakdown

| Step | Module | Description |
|------|---------|-------------|
| 1 | `data_medqa.py` | Loads MedQA-USMLE dataset and splits into train/val/test. |
| 2 | `models.py` | Wraps a HuggingFace LLM (Qwen2.5-3B-Instruct default). Extracts hidden states and log probabilities for answer options. |
| 3 | `steering.py` | Computes per-sample steering vectors (correct – mean(incorrect)). Clusters them via K-Means. |
| 4 | `projection.py` | Trains a linear projection matrix **P** so that **P(h)** aligns with the true steering vector. |
| 5 | `main_train.py` | Builds the steering dictionary, trains **P**, and saves both. |
| 6 | `main_eval.py` | Runs inference with **self-steering + confidence calibration**, evaluates on AUROC, Brier, and ECE. |
| 7 | `metrics.py` | Implements AUROC, Brier, and Expected Calibration Error (ECE). |
| 8 | `calibrator.py` | Logistic regression that maps latent similarity features → probability of correctness (the confidence). |

---

##  Confidence Computation (Critical Section)

At inference time, we derive confidence **q** from a small feature vector:

\[
\phi = [
  \max w, \,
  \text{margin of top2}(w), \,
  \cos(P(h), s), \,
  ||s||, \,
  \Delta \text{logprob}
]
\]

Then fit a **logistic regression calibrator**:

\[
q = \sigma(\beta^\top \phi)
\]

where:
- \( w \): cosine-softmax weights over cluster means,
- \( s \): weighted steering adjustment vector,
- \( P(h) \): projected hidden representation,
- \( \Delta \text{logprob} \): margin between top and runner-up options.

This **q** ∈ [0, 1] serves as the final calibrated confidence — directly compatible with AUROC, Brier, and ECE.

---

##  Evaluation Metrics

| Metric | Meaning |
|---------|----------|
| **AUROC** | How well the model’s confidence ranks correct vs. incorrect answers. |
| **Brier Score** | Mean squared error between predicted probability and true label. |
| **ECE** | Expected Calibration Error: bucketized difference between predicted and true accuracy. |

---

##  Environment & RunPod Compatibility

**Yes — fully RunPod compatible**.

### Recommended Pod:
| Setting | Value |
|----------|--------|
| Runtime | PyTorch 2.2+ (Python 3.10) |
| GPU | A100 40GB or better (1–4 GPUs supported) |
| Memory | 32 GB+ RAM |
| Storage | ≥ 20 GB for HF models + dataset cache |

### Create Environment
```bash
conda create -n steering python=3.10 -y
conda activate steering

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate scikit-learn tqdm numpy scipy

# Optional (for visual logs)
pip install seaborn matplotlib

```
##  Baseline Results

### MedQA Test Dataset:

**Qwen2.5-3B Base Model (No Training)**

AUROC: 0.5766651340314239
ECE: 0.3436040770136646
Brier: 0.3646788759271213

Accuracy: 0.4807541241162608
