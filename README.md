<div align="center">

# 🎓 Knowledge Distillation Experiments

**Teaching small "student" networks to imitate larger "teacher" networks on tabular and image data.**

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)

</div>

---

## 💡 What is knowledge distillation?

A large, accurate **teacher** model is trained first. A smaller **student** model is then trained to match the teacher's *softened* output distribution (logits divided by a temperature `T`) instead of, or in addition to, the hard labels. The soft targets carry information about how classes relate to each other, so the student often ends up more accurate than it would be when trained alone.

```
            ┌──────────────┐
  input ───►│   Teacher    │──► softmax(z_t / T) ─┐
     │      └──────────────┘                       │  distillation loss
     │      ┌──────────────┐                       ▼
     └─────►│   Student    │──► softmax(z_s / T) ──► KL / MSE  +  CE(labels)
            └──────────────┘
```

## 🧪 Experiments

| Experiment | Data | Teacher | Student | Files |
|---|---|---|---|---|
| **Tabular KD** | [UCI Adult Income](https://archive.ics.uci.edu/dataset/2/adult) (>50K prediction) | `Baseline` MLP | `RecursiveMLPBlock` (residual MLP, `[100, 100, 100]`) | `knowleddist2.py`, `residual_mlp.py`, `confusion_matrix.py` |
| **MNIST KD** | MNIST | MLP 784→1200→1200→10 | MLP 784→800→800→10 | `mnist/teacher_mnist.py`, `mnist/student_mnist.py`, `mnist/distill_mnist.py` |
| **CIFAR-10 KD** | CIFAR-10 | ResNet | Smaller network | `train.py`, `model/`, `search_hyperparams.py`, `synthesize_results.py` |

### Tabular pipeline (Adult Income)

1. Load `adult.data` / `adult.test`, impute missing values, **one-hot encode** categorical columns and **min-max scale** numeric ones.
2. Train the `Baseline` teacher.
3. Distill into `RecursiveMLPBlock`, a residual MLP whose blocks nest recursively, with temperature `T = 10`.
4. Compare accuracy, the classification report and the confusion matrix for both models.

## 🚀 Getting started

```bash
git clone https://github.com/yusuuferkan/Knowledge_Distillaton.git
cd Knowledge_Distillaton
pip install torch torchvision tensorflow scikit-learn pandas numpy tqdm
```

**MNIST:** train the teacher, then distill into the student

```bash
python mnist/teacher_mnist.py
python mnist/distill_mnist.py --epochs 10 --lr 0.01
```

**Adult Income:** download `adult.data` and `adult.test` from UCI into the repo root, then run

```bash
python knowleddist2.py
python confusion_matrix.py
```

> The MNIST and CIFAR-10 scripts were written for the old PyTorch API pinned in `requirements.txt` (e.g. `loss.data[0]`). On a recent PyTorch, replace `loss.data[0]` with `loss.item()`. The CIFAR-10 entry point also expects `model/net.py`, the other architectures and `evaluate.py` from the upstream scaffold linked below; only the ResNet and data loader are included here.

## 📁 Structure

```
├── knowleddist2.py         # tabular teacher → student distillation
├── residual_mlp.py         # Baseline & RecursiveMLPBlock models
├── confusion_matrix.py     # evaluation with confusion matrices
├── mnist/                  # MNIST teacher, student, distillation
├── train.py                # CIFAR-10 train / KD entry point
├── model/                  # data loader & ResNet
├── search_hyperparams.py   # grid search over KD hyperparameters
├── synthesize_results.py   # aggregate metrics into a table
└── utils.py
```

## 📚 References

- Hinton, Vinyals, Dean. [*Distilling the Knowledge in a Neural Network*](https://arxiv.org/abs/1503.02531), 2015.
- The CIFAR-10 training scaffold follows [haitongli/knowledge-distillation-pytorch](https://github.com/haitongli/knowledge-distillation-pytorch).

---

<div align="center">

Made by **[Yusuf Erkan](https://www.linkedin.com/in/yusuferkan1/)** · ⭐ the repo if it helped you

</div>
