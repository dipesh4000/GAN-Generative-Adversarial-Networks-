# 📚 Module 6: Generative Adversarial Networks — Study Notes

**Course:** Generative AI and Deep Learning  
**Institute:** Wadhwani School of AI, IIT Madras  
**Instructors:** Prof. Balaji Srinivasan & Prof. Ganapathy Krishnamurthi  
**Sources:** Lecture Slides (65 slides) + GAN from Scratch Notebook (FashionMNIST)

---

## 🗂️ Note Files

| # | File | Topics Covered |
|---|------|----------------|
| 01 | [Introduction to Generative Models](./01_introduction_to_generative_models.md) | Discriminative vs Generative, latent space, model taxonomy (VAE/GAN/Flow/Diffusion) |
| 02 | [GAN Theory & Mathematics](./02_gan_theory_and_mathematics.md) | Minimax objective, optimal discriminator, JSD, Nash equilibrium, training algorithm |
| 03 | [GAN Architectures](./03_gan_architectures.md) | Vanilla GAN, DCGAN, StyleGAN, activation functions, latent space |
| 04 | [Training Dynamics](./04_training_dynamics.md) | Alternating optimization, loss functions, D/G training steps, loss curves |
| 05 | [Challenges & Solutions](./05_gan_challenges_and_solutions.md) | Mode collapse, vanishing gradients, WGAN, evaluation metrics (FID, IS) |
| 06 | [Complete Code Implementation](./06_complete_code_implementation.md) | Full PyTorch GAN from scratch on FashionMNIST, annotated code |

---

## 🔑 Key Equations at a Glance

**Minimax Objective:**
$$\min_G \max_D V(D,G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

**Optimal Discriminator:**
$$D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$$

**Nash Equilibrium:** $D^*(x) = \frac{1}{2}$ when $p_g = p_{\text{data}}$

**BCE Loss:**
$$\mathcal{L}_{\text{BCE}}(y, \hat{y}) = -[y \cdot \log(\hat{y}) + (1-y) \cdot \log(1-\hat{y})]$$

**JSD Connection:**
$$V(D^*, G) = -\log 4 + 2 \cdot \text{JSD}(p_{\text{data}} \| p_g)$$

---

## 🏗️ Notebook Architecture (FashionMNIST GAN)

| Component | Config |
|-----------|--------|
| Dataset | FashionMNIST (60,000 images, 28×28) |
| Latent size | 64 |
| Hidden size | 256 |
| Batch size | 100 |
| Epochs | 100 |
| Optimizer | Adam, lr=0.0002 |
| Loss | BCELoss |
| Final D loss | 0.8781 |
| Final G loss | 1.5290 |
| Final D(x) | 0.75 (ideal: 0.5) |
| Final D(G(z)) | 0.30 (ideal: 0.5) |

---

## 📖 Key References

| Paper | Authors | Year | Contribution |
|-------|---------|------|--------------|
| [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) | Goodfellow et al. | 2014 | Original GAN paper |
| [DCGAN](https://arxiv.org/abs/1511.06434) | Radford et al. | 2015 | Stable convolutional GAN |
| [StyleGAN](https://arxiv.org/abs/1812.04948) | Karras et al. (Nvidia) | 2019 | Style-based generator |
| [WGAN](https://arxiv.org/abs/1701.07875) | Arjovsky et al. | 2017 | Wasserstein distance, fixes mode collapse |
| [Improved GAN Training](https://arxiv.org/abs/1606.03498) | Salimans et al. | 2016 | Minibatch disc., label smoothing |
| [Progressive GAN](https://arxiv.org/abs/1710.10196) | Karras et al. | 2018 | Progressive growing |
| [FID](https://arxiv.org/abs/1706.08500) | Heusel et al. | 2017 | Fréchet Inception Distance metric |

---

> [!TIP]
> Start with **01** for conceptual grounding, then **02** for math depth, **03** for architecture details, **04** for practical training, **05** for debugging, and **06** for a complete working implementation.
