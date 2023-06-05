# attADs_NWP

## Introduction
This paper proposes a novel method for conducting adversarial attacks against optical aerial detectors by leveraging natural weather perturbations. Compared to existing methods, our scheme produces more natural and stealthy adversarial perturbations. We formulate the generation of adversarial weather perturbations in black-box as an optimization problem and effectively solve it using the Differential Evolution (DE) algorithm under the constraints of $L_{\infty}$ and $L_2$, respectively. Through extensive experiments, we verify the effectiveness of our method and investigate the transferability of the generated adversarial examples across different models. Additionally, we assess the robustness of our method against typical defense mechanisms, demonstrating its strong resilience. The proposed method offers a new perspective for developing effective adversarial attack methods against optical aerial detectors and helps to understand the vulnerability and interpretability of DNNs.

## Rerquiremens
Our code was built on [Pytorch](https://pytorch.org/).

## Usage
Run
`python TFNS.py.`

You need to replace the weightfile in `TFNS.py` and `TDE_main_hybrid.py` with your weight files. 

## DOTA-W
- This is a subset of DOTA with adversarail *weather* perturbations (DOTA-W) by our proposed method.
- DOTA-W is provided for reproducing our results.
- DOTA-W serves as a potential benchmark to evaluate the robustness of Optical Aerial Detectors (OAD). Also, it can be act as a  promising and effective data augmentation method for model training, ultimately improving the robustness of DNNs.
- DOTA-W is built upon the subset of [DOTA v1.0](https://captain-whu.github.io/DOTA/dataset.html) validation set with 1k samples.

## References
- [DOTA](https://captain-whu.github.io/DOTA/code.html)
- [Download](https://pan.baidu.com/s/1_RLu3vT6ri3yzfrLixZQMA), password: DAWP.
