# CS-300 Project Report: Image Inpainting using Conditional Generative Adversarial Networks (cGANs)

**Title**: Image Inpainting Using Generative Adversarial Networks  
**Author**: Ayush Dutt Mishra (Roll No: 2301058)  
**Advisor**: Dr. Kaustuv Nag, Assistant Professor  
**Department**: Department of Computer Science and Engineering, Indian Institute of Information Technology Guwahati (IIIT Guwahati)  
**Degree / Context**: B.Tech Semester VI Project Report, April 2026  
**Framework**: PyTorch  
**Dataset**: CelebA Facial Dataset (20,000 images, 256×256 resolution)  
**Official PDF Report**: [`Ayush_2301058_report (2).pdf`](Ayush_2301058_report%20%282%29.pdf)  

---

## Abstract

Image Inpainting is a fundamental computer vision task that aims to reconstruct missing, corrupted, or occluded regions in images while maintaining semantic consistency and visually plausible textures. In this work, I study GAN-based inpainting and implement a Pix2Pix-inspired conditional adversarial framework for face completion on the CelebA dataset. To overcome common GAN instability issues and boundary artifacts, I introduce **Spectral Normalization (`spectral_norm`)** in the PatchGAN Discriminator, a **VGG-16 Perceptual Loss** alongside L1 pixel reconstruction loss, and a **Soft-Composite Cosine Alpha Feathering** post-processing stage. The final model was trained across 4 progressive phases up to **116 epochs**, achieving a peak Peak Signal-to-Noise Ratio (**PSNR**) of **29.01 dB** and Structural Similarity Index (**SSIM**) of **0.998**.

---

## 1. Introduction & Problem Statement

Given an incomplete face image $I \in \mathbb{R}^{H \times W \times 3}$ and a binary mask $M \in \{0, 1\}^{H \times W}$ indicating missing pixel locations ($M_{ij} = 1$ for missing), the goal of image inpainting is to produce a completed image $\hat{I}$ such that:
1. The unmasked region $\hat{I}_{ij} = I_{ij}$ for all valid pixels ($M_{ij} = 0$).
2. The filled region $\hat{I}_{ij}$ is perceptually realistic, structurally smooth, and semantically consistent for all missing pixels ($M_{ij} = 1$).

The corrupted observation available to the model is $I_m = I \odot (1 - M) + c \odot M$, where $\odot$ denotes element-wise multiplication and $c$ is a constant fill value. Traditional algorithmic approaches (such as diffusion or exemplar patch synthesis) fail when the missing area is large and demands high-level semantic understanding of facial structure (eyes, nose, lips).

---

## 2. System Architecture & Model Components

![cGAN Inpainting System Architecture](assets/cgan_inpainting_architecture.jpg)  
*Figure 1: Complete System Architecture showing UNet Generator, Spectral-Normalized PatchGAN Discriminator, VGG-16 Perceptual Loss, and Soft-Composite Post-Processor.*

### 2.1 UNet Generator Component (54.4M Parameters)
The Generator employs an **8-stage Encoder-Decoder architecture with Skip Connections**:
- **Encoder**: 8 convolutional downsampling stages (`e1`..`e7` + `bn`) with stride 2 and LeakyReLU activation ($\alpha=0.2$). Downsamples spatial dimensions from $256 \times 256$ down to $1 \times 1$.
- **Bottleneck**: Captures global semantic context (facial geometry, eye spacing, nose bridge).
- **Decoder**: 8 transposed convolutional upsampling stages (`d1`..`d7` + `out`) with skip connections concatenating encoder feature maps directly to corresponding decoder stages. Skip connections bypass the bottleneck, allowing fine spatial details (edges, skin texture, color boundaries) to transfer cleanly.

### 2.2 Spectral-Normalized PatchGAN Discriminator Component
The Discriminator evaluates local $70 \times 70$ patches:
- Concatenates the condition image $I_m$ and target/generated image $\hat{I}$ along the channel dimension ($6$ channels).
- **Spectral Normalization (`spectral_norm`)**: Wraps every convolutional layer with `torch.nn.utils.spectral_norm` to enforce Lipschitz continuity ($\sigma(W) \le 1$), stabilizing adversarial training.

### 2.3 Loss Function Formulation
The generator is optimized using a weighted compound loss:

$$\mathcal{L}_{\text{Generator}} = \mathcal{L}_{\text{adv}}(D(I_m, \hat{I}), 1) + 100 \cdot \| I - \hat{I} \|_1 + 10 \cdot \mathcal{L}_{\text{perc}}(\hat{I}, I)$$

where $\mathcal{L}_{\text{perc}}$ measures feature activation distances in layer `features[16]` of a pre-trained VGG-16 network.

---

## 3. Four-Phase Progressive Training & Dynamics

| Phase | Description | Epochs | Batch Size | LR (G) | LR (D) | Best Metric |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **P1 Baseline** | 50-epoch baseline (Adversarial + L1) | 50 | 16 | $2 \times 10^{-4}$ | $5 \times 10^{-5}$ | PSNR 24.11 dB, SSIM 0.993 |
| **P2 Enhanced** | Checkpoint continuation + Perceptual Loss | ~30 | 8 | $5 \times 10^{-5}$ | $1 \times 10^{-5}$ | PSNR ~25.5 - 26.0 dB |
| **P3 Refine** | Eye/Mouth fine detail refinement | ~30 | 8 | $5 \times 10^{-5}$ | $1 \times 10^{-5}$ | Sharper mask boundaries |
| **P4 Final** | Final consolidation up to Epoch 116 | ~30 | 8 | $1 \times 10^{-5}$ | $5 \times 10^{-6}$ | **`PSNR 29.01 dB, SSIM 0.998`** |

![Training Loss Curves and Metrics](assets/training_loss_curves.png)  
*Figure 2: (a) Baseline 50-Epoch Training Dynamics showing Generator/Discriminator Loss and PSNR/SSIM growth; (b) Cross-phase PSNR progression.*

---

## 4. Quantitative & Qualitative Results

### 4.1 Quantitative Comparison with Prior Methods

| Method | Mask Type | Perceptual Loss | Domain | PSNR (approx.) |
| :--- | :--- | :--- | :--- | :--- |
| Context Encoders (Pathak et al., 2016) | Rectangular | No | General | ~20.0 dB |
| Pix2Pix (Isola et al., 2017) | Rectangular | No | General | ~23.0 dB |
| Partial Conv. (Liu et al., 2018) | Irregular | Yes | General | ~27.0 dB |
| DeepFill (Yu et al., 2018) | Free-form | Yes | General | ~28.0 dB |
| **Proposed (Phase-4 Final)** | **Rectangular** | **Yes** | **Face** | **`29.01 dB`** |

### 4.2 Qualitative Image Showcase

![Inpainting Test Results](assets/qualitative_reconstructions.png)  
*Figure 3: Qualitative evaluation grid across test faces showing Masked Input (top), Inpainted Generator Output (middle), and Ground Truth target (bottom).*

![Soft Compositing Feathering](assets/soft_composite_feathering.png)  
*Figure 4: Soft compositing cosine alpha feathering comparison showing seamless edge boundary transitions.*

---

## 5. Conclusions & Future Work

In this project, a complete PyTorch cGAN inpainting framework was designed, implemented, and evaluated for face completion on the CelebA dataset. Multi-phase checkpoint continuation coupled with Spectral Normalization and VGG Perceptual Loss achieved a **+12.81 dB PSNR gain** over initial epoch 1 baseline, reaching a peak **29.01 dB PSNR** and **0.998 SSIM**.

---

## References
1. Elharrouss, O., et al. (2020). *Image inpainting: A review*. Neural Processing Letters.
2. Bertalmio, M., et al. (2000). *Image inpainting*. ACM SIGGRAPH.
3. Goodfellow, I., et al. (2014). *Generative adversarial nets*. NIPS.
4. Isola, P., et al. (2017). *Image-to-image translation with conditional adversarial networks*. CVPR.
5. Liu, Z., et al. (2015). *Deep learning face attributes in the wild*. ICCV.
6. Wang, Z., et al. (2004). *Image quality assessment: From error visibility to structural similarity*. IEEE TIP.
7. Johnson, J., et al. (2016). *Perceptual losses for real-time style transfer and super-resolution*. ECCV.
8. Mirza, M., & Osindero, S. (2014). *Conditional generative adversarial nets*. arXiv preprint.
9. Liu, G., et al. (2018). *Image inpainting for irregular holes using partial convolutions*. ECCV.
10. Pathak, D., et al. (2016). *Context encoders: Feature learning by inpainting*. CVPR.
11. Ronneberger, O., et al. (2015). *U-net: Convolutional networks for biomedical image segmentation*. MICCAI.
12. Yu, J., et al. (2019). *Free-form image inpainting with gated convolution*. ICCV.
13. Yu, J., et al. (2018). *Generative image inpainting with contextual attention*. CVPR.
