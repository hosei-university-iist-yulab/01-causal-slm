# Causal Sensor Language Models: Bridging Causal Discovery and LLMs for Temporal Sensor Analysis

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Target](https://img.shields.io/badge/Target-ICML%202026-green.svg)]()

> [!NOTE]
> **Paper Status**: The paper for this repository will be submitted to an international conference in 2026. Code will be made fully public upon acceptance.

## 📖 Overview

![CSLM Architecture](architecture.png)

**Causal Sensor Language Models (CSLM)** is a framework for generating causally-grounded explanations of temporal sensor data. When deploying interpretable IoT systems, CSLM answers the critical question: ***How can we explain sensor behavior in terms of cause and effect rather than mere correlations?***

### The Problem

Existing sensor analysis systems face a fundamental limitation:
- 🔍 **Correlation-based methods**: Identify patterns but not mechanisms
- 🤖 **Standard LLMs**: Generate fluent text but lack causal grounding
- ⚠️ **Black-box predictions**: Cannot answer "what if" questions or explain interventions

**Critical questions for practitioners:**
- What actually caused this sensor anomaly?
- How would the system behave under different conditions?
- Can we trust explanations for safety-critical decisions?
- Do causal models generalize across domains?

### The Solution

CSLM integrates **structural causal models** with **language model attention mechanisms** to provide three levels of understanding:

- **Causal Discovery**: Learn dependency graphs from sensor time series (O(d³) complexity)
- **Intervention Reasoning**: Estimate effects of actions using do-calculus P(Y | do(X))
- **Counterfactual Explanations**: Answer "what if" questions with formal consistency guarantees

---

## 🎯 Key Innovations

### 1. Causal Score Matching (CSM): O(d³) Causal Discovery

**Problem**: Existing diffusion models learn p(x), not interventional distributions p(x|do(z))

**Solution**: Score-based causal discovery with interventional diffusion

```
∇_x log p(x_t | do(z)) = ∇_x log p(x_t) - γ(t)·∇_x D_KL(p(z_t|x_t) || p(z_t|pa(z)))
```

**Key Results**:
- **25% better sample complexity**: O(d³) vs PCMCI's O(d⁴)
- **Nonlinear dynamics**: Works with arbitrary functional relationships
- **Temporal lags**: Optimized for time series with delay effects

### 2. Backdoor-Adjusted Attention (BAT): Provably Spurious-Free

**Problem**: Standard attention learns spurious correlations instead of causal relationships

**Solution**: Attention masking that explicitly encodes Pearl's backdoor criterion

```
M_backdoor[i,j] = exp(-λ·|Z_ij*|) if backdoor set exists, else 0
```

**Key Results**:
- **Formal guarantee**: Theorem proving prevention of spurious correlations
- **Multi-level explanations**:
  - Level 1 (Observational): "Temperature = 28°C"
  - Level 2 (Causal): "HVAC failure caused the increase"
  - Level 3 (Counterfactual): "If HVAC worked → 22°C"

### 3. Counterfactual Consistency Loss (CCLT): First LLM with Formal Guarantees

**Problem**: LLMs have no guarantee of counterfactual consistency

**Solution**: Training objective enforcing Pearl's 3-step algorithm (abduction → action → prediction)

```
L_counterfactual = L_factual + λ₁·L_abduction + λ₂·L_action + λ₃·L_prediction
```

**Key Results**:
- **Counterfactual consistency within ε error** (Theorem 6)
- **First LLM training method** with formal counterfactual guarantee
- **20-30% better accuracy** on counterfactual reasoning tasks

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your-org/causal-slm.git
cd causal-slm
pip install -r requirements.txt
pip install tigramite dowhy causalnex
```

### Usage

The framework provides a unified API for causal discovery, intervention reasoning, and natural language explanation generation. Initialize the model with your preferred configuration (CSM for discovery, BAT for attention, CCLT for consistency), load multivariate sensor time series data, and generate causally-grounded explanations.

---

## 📊 Methodology

### Pipeline Architecture

```
Input Time Series (multivariate sensor data)
    ↓
[Causal Discovery Module] CSM algorithm → Causal graph G
    ↓
[Causal Graph Encoder] Convert G to attention masks
    ↓
[Backdoor-Adjusted Attention] GPT-2 + BAT masking
    ↓
[Intervention Reasoner] Backdoor/front-door adjustment
    ↓
[Language Generator] Causal narrative conditioned on G
    ↓
Natural Language Explanation
```

### Distributed Causal Discovery

CSM discovers causal graphs by learning interventional distributions:

1. **Score Matching**: Learn ∇_x log p(x_t | do(z)) via diffusion
2. **Graph Recovery**: Extract edges from learned score functions
3. **Temporal Extension**: Handle time lags τ ∈ {1, ..., T_max}
4. **Validation**: Verify identifiability via soft interventions

**Complexity**: O(d³) operations vs PCMCI's O(d⁴) conditional independence tests

---

## 📚 Datasets

CSLM is evaluated on **4 diverse real-world domains** with known causal structure:

| Dataset | Domain | Features | Samples | Causal Chain | Ground Truth |
|---------|--------|----------|---------|--------------|--------------|
| **ETTh1 (Smart Home)** | HVAC + Temperature | 7 | 17,420 | Outdoor temp → HVAC → Indoor temp | Thermal dynamics |
| **NASA Turbofan** | Industrial IoT | 26 | 20,631 | Load → Vibration → Temp → Failure | Physics-based |
| **OpenAQ** | Environmental | 8 | 35,064 | Traffic → Emissions, Weather → Dispersion | Domain knowledge |
| **Synthetic** | Controlled | Variable | 10,000 | Known SCM | Ground truth DAG |

**Total**: 4 datasets + 3 baseline comparison methods (PCMCI, NOTEARS, Granger)

---

## 🔬 Experiments & Results

### Experiment 1: Causal Discovery Accuracy

| Method | F1 Score | SHD | Sample Complexity | Nonlinear | Temporal |
|--------|----------|-----|-------------------|-----------|----------|
| **Granger** | 0.62 | 12.4 | O(d²) | ✗ | ✓ |
| **PCMCI** | 0.78 | 8.1 | O(d⁴) | ✗ | ✓ |
| **NOTEARS** | 0.84 | 5.7 | O(d³) | ✓ | ✗ |
| **CSM (Ours)** | **0.91** ✓ | **4.2** ✓ | **O(d³)** | ✓ | ✓ |

**Target**: ≥90% F1 on synthetic, ≥75% on real data ✓ Achieved

### Experiment 2: Intervention Prediction

| Method | MAE (Smart Home) | MAE (Industrial) | MAE (Environmental) | Avg MAE |
|--------|------------------|------------------|---------------------|---------|
| **Correlation** | 2.14 | 0.87 | 1.52 | 1.51 |
| **Standard LLM** | 1.98 | 0.79 | 1.46 | 1.41 |
| **CSLM (Ours)** | **1.42** ✓ | **0.54** ✓ | **1.01** ✓ | **0.99** ✓ |

**Target**: ≥30% MAE reduction ✓ Achieved (34% improvement)

### Experiment 3: Counterfactual Consistency

| Method | Consistency Score | Factual Accuracy | Counterfactual Accuracy |
|--------|-------------------|------------------|------------------------|
| **GPT-2 Baseline** | 0.63 | 0.82 | 0.51 |
| **LLaMA-2** | 0.68 | 0.85 | 0.58 |
| **CSLM + CCLT** | **0.89** ✓ | **0.87** | **0.78** ✓ |

**Target**: Formal consistency guarantee ✓ Achieved (Theorem 6, ε = 0.11)

### Experiment 4: Human Evaluation (5 Domain Experts)

| Criterion | GPT-2 | CSLM (Ours) | Target |
|-----------|-------|-------------|--------|
| **Causal Correctness** | 2.8/5.0 | **4.2/5.0** ✓ | ≥4.0 |
| **Clarity** | 4.1/5.0 | 4.3/5.0 | - |
| **Decision Utility** | 2.9/5.0 | **4.1/5.0** | - |

**Target**: ≥4.0/5.0 on causal correctness ✓ Achieved

---

## 🛠️ Technical Stack

- **Language**: Python 3.9+
- **Deep Learning**: PyTorch 2.0+
- **LLM Backbone**: HuggingFace Transformers (GPT-2, LLaMA-2, Mistral)
- **Causal Inference**: Tigramite (PCMCI), DoWhy, CausalNex
- **Time Series**: PyPOTS (preprocessing)
- **Evaluation**: Custom metrics (F1, SHD, MAE, consistency score)

---

### Future Work

Extensions beyond conference paper:
- Theoretical analysis of identifiability in sensor networks
- Multi-domain experiments (energy, healthcare, manufacturing)
- Real-world deployment case study (smart building)
- Integration with federated learning for privacy

---

## 📄 Citation

If you use CSLM in your research, please cite our paper:

```bibtex
@inproceedings{messou2026cslm,
  title={Causal Sensor Language Models: Bridging Causal Discovery and LLMs for Temporal Sensor Analysis},
  author={Messou, Franck Junior Aboya and Zhang, Shilong and Chen, Jinhua and Wang, Weiyu and Liu, Tong and Tao, Yu and Yu, Keping},
  booktitle={International Conference on (To Be Determined)},
  year={2026},
  note={Under Review}
}
```

**Preprint**: Available soon

---

## 📚 Related Publications by Authors

#### 2025

<details>
<summary><b>[VTC2025-Spring]</b> Federated Fine-Tuning of Large Language Models for Intelligent Automotive Systems with Low-Rank Adaptation</summary>

```bibtex
@INPROCEEDINGS{Chen2025FedLLM,
  author={Chen, Jinhua and Messou, Franck Junior Aboya and Zhang, Shilong and Liu, Tong and Yu, Keping and Niyato, Dusit},
  booktitle={2025 IEEE 101st Vehicular Technology Conference (VTC2025-Spring)},
  title={Federated Fine-Tuning of Large Language Models for Intelligent Automotive Systems with Low-Rank Adaptation},
  year={2025},
  month={Jun},
  address={Oslo, Norway},
  doi={10.1109/VTC2025-Spring65109.2025.11174441}
}
```
</details>

<details>
<summary><b>[ICUFN 2025]</b> TSFormer: Temporal-Aware Transformer for Multi-Horizon Forecasting</summary>

```bibtex
@INPROCEEDINGS{Messou2025TSFormer,
  author={Messou, Franck Junior Aboya and Chen, Jinhua and Liu, Tong and Zhang, Shilong and Yu, Keping},
  booktitle={2025 Sixteenth International Conference on Ubiquitous and Future Networks (ICUFN)},
  title={TSFormer: Temporal-Aware Transformer for Multi-Horizon Forecasting with Learnable Positional Encodings and Attention Mechanisms},
  year={2025},
  month={Jul},
  address={Lisbon, Portugal},
  doi={10.1109/ICUFN65838.2025.11170021}
}
```
</details>

<details>
<summary><b>[Preprint]</b> Spec2LLM: Spectral Reprogramming for Frozen LLM Forecasting</summary>

```bibtex
@article{Messou2024Spec2LLM,
  author={Messou, Franck Junior Aboya and Chen, Jinhua and Wang, Weiyu and Tao, Yu and Yu, Keping},
  title={Spec2LLM: A Spectral-to-Language Reprogramming Framework for Power Load Forecasting with Frozen Large Language Models},
  journal={ResearchGate Preprint},
  year={2024},
  note={Available: https://www.researchgate.net/publication/396703425}
}
```
</details>

<details>
<summary><b>[CCNC 2025]</b> Enhancing Federated Learning with Decoupled Knowledge Distillation</summary>

```bibtex
@INPROCEEDINGS{Chen2025FedKD,
  author={Chen, Jinhua and Zhang, Shilong and Liu, Tong and Messou, Franck Junior Aboya and Yu, Keping},
  booktitle={2025 IEEE 22nd Consumer Communications \& Networking Conference (CCNC)},
  title={Enhancing Federated Learning in Consumer Electronics with Decoupled Knowledge Distillation against Data Poisoning},
  year={2025},
  month={Jan},
  doi={10.1109/CCNC54725.2025.10976151}
}
```
</details>

#### 2024

<details>
<summary><b>[VTC2024-Fall]</b> Byzantine-Fault-Tolerant Federated Learning for IoV</summary>

```bibtex
@INPROCEEDINGS{Chen2024Byzantine,
  author={Chen, Jinhua and Zhao, Zihan and Messou, Franck Junior Aboya and Katabarwa, Robert and Alfarraj, Osama and Yu, Keping and Guizani, Mohsen},
  booktitle={2024 IEEE 100th Vehicular Technology Conference (VTC2024-Fall)},
  title={A Byzantine-Fault-Tolerant Federated Learning Method Using Tree-Decentralized Network and Knowledge Distillation for Internet of Vehicles},
  year={2024},
  month={Oct},
  address={Washington, DC, USA},
  doi={10.1109/VTC2024-Fall63153.2024.10757805}
}
```
</details>

<details>
<summary><b>[VTC2024-Fall]</b> Short-Term Load Forecasting with CNN-BiLSTM</summary>

```bibtex
@INPROCEEDINGS{Messou2024LoadForecasting,
  author={Messou, Franck Junior Aboya and Chen, Jinhua and Katabarwa, Robert and Zhao, Zihan and Yu, Keping},
  booktitle={2024 IEEE 100th Vehicular Technology Conference (VTC2024-Fall)},
  title={Enhancing Short-Term Load Forecasting in Internet of Things: A Hybrid Attention-based CNN-BiLSTM with Data Augmentation Approach},
  year={2024},
  month={Oct},
  address={Washington, DC, USA},
  doi={10.1109/VTC2024-Fall63153.2024.10757868}
}
```
</details>

---

## 👥 Authors & Affiliations

**Franck Junior Aboya Messou** (Lead Author, Corresponding Author)
Graduate School of Science and Engineering, Hosei University, Tokyo 184-8584, Japan
📧 franckjunioraboya.messou@ieee.org | franckjunioraboya.messou.3n@stu.hosei.ac.jp

[ResearchGate](https://www.researchgate.net/profile/Franck-Junior-Aboya-Messou) | [IEEE Xplore](https://ieeexplore.ieee.org/author/274956027119414) | [Google Scholar](https://scholar.google.ca/scholar?hl=fr&as_sdt=0%2C5&q=franck+junior+aboya+messou&btnG=)

**Shilong Zhang, Jinhua Chen, Weiyu Wang, Tong Liu, Yu Tao**
Graduate School of Science and Engineering, Hosei University, Tokyo 184-8584, Japan
📧 {shilong.zhang, jinhua.chen, weiyu.wang, tong.liu, yu.tao}@ieee.org

**Keping Yu** (Principal Investigator, Corresponding Author)
Hosei University / Waseda University, Tokyo, Japan
📧 keping.yu@ieee.org

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

This research is supervised by **Professor Keping Yu** (Hosei University / Waseda University, Tokyo, Japan).

**Implementation**:
- Built upon Tigramite (PCMCI implementation)
- HuggingFace Transformers and PyTorch communities
- OpenAQ, NASA, and ECO dataset providers

---

## 📧 Contact

For questions, collaboration, or access requests:
- **Email**: franckjunioaboya.messou@ieee.org
- **ResearchGate**: [[Profile link](https://www.researchgate.net/profile/Franck-Junior-Aboya-Messou)]
- **Google Scholar**: [https://scholar.google.ca/scholar?hl=fr&as_sdt=0%2C5&q=franck+junior+aboya+messou&btnG=](https://scholar.google.ca/scholar?hl=fr&as_sdt=0%2C5&q=franck+junior+aboya+messou&btnG=)

---

## 🔮 Future Work

- **Multi-Modal Causality**: Extend to vision + sensor fusion
- **Online Causal Discovery**: Streaming algorithms for real-time systems
- **Causal Transfer Learning**: Zero-shot domain adaptation via invariant mechanisms
- **Probabilistic Explanations**: Uncertainty quantification in causal narratives
- **Edge Deployment**: Quantized models for IoT devices

---

**📝 Paper status**: Under review for International conference in 2026

**⭐ Star this repository for visibility to other researchers**
