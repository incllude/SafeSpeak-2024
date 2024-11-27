# SafeSpeak-2024

## Task

Focus on advancing audio spoofing detection technologies for secure voice authentication resolving arising challenges in the field of voice anti-spoofing and presentation attack detection. Competition challenges to develop lightweight, high-performance models capable of detecting audio spoofing attacks. With a strong focus on computational efficiency and real-world applicability, participants will be evaluated based on ASVspoof metrics, ensuring their models meet industry standards for robustness and accuracy

**Evaluation Metric — Equal Error Rate (EER)**

## Keypoints:
- For training and local validation were selected [ASVspoof 2019](https://www.asvspoof.org/index2019.html) and [ASVspoof 2021](https://www.asvspoof.org/index2021.html) public datasets
- For training used **full train** and **75% of eval** from **ASVspoof 2019** and about **11k** of negative samples from **eval** part from **ASVspoof 2021** (negative samples are based on **RawNet2** scores)
- [AASIST2](https://arxiv.org/abs/2309.08279v2) was used as model
- Weights were initialized from [XLS-R](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec/xlsr) and [AASIST](https://github.com/asvspoof-challenge/asvspoof5/tree/main/Baseline-AASIST) pretrained weights
- **Additive Angular Margin Softmax (AM-Softmax)** was used as loss with constant scale of 8 and margin of 0.3
