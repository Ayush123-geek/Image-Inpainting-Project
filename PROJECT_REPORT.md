# 📑 CS-300 Project Report: Image Inpainting using Conditional Generative Adversarial Networks (cGANs)

**Course**: CS-300  
**Project Topic**: Image Inpainting using Conditional GANs (pix2pix)  
**Author**: Ayush Mishra  
**Framework**: PyTorch  
**Dataset**: CelebA Facial Dataset (20,000 images, 256×256 resolution)  

---

## Abstract

Image Inpainting is a fundamental computer vision task that aims to reconstruct missing, corrupted, or occluded regions in images while maintaining semantic consistency and visually plausible textures. In this project, we implement and evaluate a Conditional Generative Adversarial Network (cGAN) based on the pix2pix architecture. To overcome common GAN instability issues and boundary artifacts, we introduce **Spectral Normalization (`spectral_norm`)** in the PatchGAN Discriminator, a **VGG-16 Perceptual Loss** alongside L1 pixel reconstruction loss, and a **Soft-Composite Cosine Alpha Feathering** post-processing stage. The final model was trained for **116 epochs**, achieving a peak Peak Signal-to-Noise Ratio (**PSNR**) of **29.01 dB** and Structural Similarity Index (**SSIM**) of **0.998**.

---

## 1. Introduction & Problem Statement

Given an incomplete image \(I_{\text{masked}} = I \odot (1 - M)\) corrupted by a binary mask \(M\), the goal of image inpainting is to generate a reconstructed image \(\hat{I}\) such that:
1. The unmasked region \(\hat{I} \odot (1 - M)\) matches the ground-truth target image \(I\) exactly.
2. The filled region \(\hat{I} \odot M\) is semantically consistent, structurally smooth, and visually indistinguishable from natural image context.

Traditional algorithmic approaches (such as Telea or Navier-Stokes PDE-based fluid dynamics) rely solely on local patch statistics and fail to synthesize complex semantic features like facial structure, eyes, nose, or lips. Generative Adversarial Networks, specifically Conditional GANs, learn a deep prior over image distributions to generate missing semantic features.

---

## 2. Literature Review & Theoretical Foundation

Our implementation builds upon three foundational computer vision & deep learning papers:

1. **Generative Adversarial Networks (Goodfellow et al., 2014)**:
   Introduced the minimax game between a Generator \(G\) and a Discriminator \(D\).
   \[
   \min_G \max_D V(D, G) = \mathbb{E}_{y \sim p_{\text{data}}(y)}[\log D(y)] + \mathbb{E}_{z \sim p_z(z)}[\log (1 - D(G(z)))]
   \]

2. **Conditional GANs (Mirza & Osindero, 2014)**:
   Extends GANs by conditioning both Generator and Discriminator on observed auxiliary information \(x\) (the masked input image):
   \[
   \min_G \max_D V(D, G) = \mathbb{E}_{x, y}[\log D(x, y)] + \mathbb{E}_{x, z}[\log (1 - D(x, G(x, z)))]
   \]

3. **pix2pix Architecture (Isola et al., 2016)**:
   Demonstrated that combining conditional adversarial loss with an \(L_1\) pixel-distance loss produces sharp, realistic image-to-image translations without blurring:
   \[
   \mathcal{L}_{L1}(G) = \mathbb{E}_{x, y, z}[\| y - G(x, z) \|_1]
   \]

---

## 3. Method & Network Architecture

### 3.1 UNet Generator
The Generator employs an **8-stage Encoder-Decoder architecture with Skip Connections**:
- **Encoder**: 8 convolutional layers with stride 2 and LeakyReLU activation (\(\alpha=0.2\)). Downsamples spatial dimensions from \(256 \times 256\) down to \(1 \times 1\).
- **Decoder**: 8 transposed convolutional layers upsampling back to \(256 \times 256\).
- **Skip Connections**: Concatenates feature maps from encoder layer \(i\) directly into decoder layer \(N - i\) to prevent information loss during downsampling bottlenecking.
- **Total Parameters**: **54,414,531** (~54.4M parameters).

### 3.2 Spectral-Normalized PatchGAN Discriminator
The Discriminator evaluates local \(70 \times 70\) patches rather than the entire image:
- Concatenates the condition image \(x\) and target/generated image \(y\) along the channel dimension (6 input channels).
- **Spectral Normalization**: Wraps every convolutional layer with `torch.nn.utils.spectral_norm`. Spectral normalization bounds the matrix operator norm of convolutional weights:
  \[
  \sigma(W) = \max_{h: h \neq 0} \frac{\|Wh\|_2}{\|h\|_2} = 1
  \]
  This prevents exploding gradients in the discriminator and ensures smooth discriminator loss trajectories throughout 116 training epochs.

### 3.3 Loss Function Formulation
The generator is optimized using a weighted compound loss:

\[
\mathcal{L}_{\text{Generator}} = \mathcal{L}_{\text{BCE}}(D(x, G(x)), 1) + 100 \cdot \mathcal{L}_{L1}(G(x), y) + 10 \cdot \mathcal{L}_{\text{VGG16}}(G(x), y)
\]

where \(\mathcal{L}_{\text{VGG16}}\) extracts feature representations from layer `features[16]` of a pre-trained VGG-16 network, enforcing high-level structural and contextual similarity.

### 3.4 Soft Compositing & Feathering
Raw GAN outputs often display subtle color shifts at mask boundaries. To remedy this, we implement a **Cosine Feather Mask**:
\[
\alpha(u, v) = \frac{1}{2} \left[1 + \cos\left(\pi \cdot \frac{d(u, v)}{r}\right)\right]
\]
The final composite output smoothly blends generated patch pixels with original surrounding pixels over a 15-pixel transition margin.

---

## 4. Experimental Setup & Training Details

| Hyperparameter | Value |
| :--- | :--- |
| **Dataset** | CelebA Dataset (5,000 to 20,000 sub-sampled facial images) |
| **Resolution** | \(256 \times 256 \times 3\) |
| **Mask Type** | Center Mask (\(64 \times 64\)) & Random Rectangular Masking |
| **Batch Size** | 8 to 16 |
| **Learning Rate (G)** | \(1.5 \times 10^{-4}\) (refined down to \(1.0 \times 10^{-5}\)) |
| **Learning Rate (D)** | \(3.0 \times 10^{-5}\) (refined down to \(5.0 \times 10^{-6}\)) |
| **Optimizer** | Adam (\(\beta_1 = 0.5, \beta_2 = 0.999\)) |
| **Total Epochs** | **116 Epochs** |
| **Hardware** | NVIDIA CUDA-enabled GPU (Google Colab / Kaggle) |

---

## 5. Quantitative & Qualitative Results

### 5.1 Quantitative Performance Metrics

| Epoch Milestone | Generator Loss (\(\mathcal{L}_G\)) | Discriminator Loss (\(\mathcal{L}_D\)) | Perceptual Loss | PSNR (dB) | SSIM |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Epoch 10** | 3.421 | 0.691 | 0.142 | 22.45 dB | 0.965 |
| **Epoch 50** | 2.518 | 0.684 | 0.098 | 25.12 dB | 0.984 |
| **Epoch 80** | 2.114 | 0.680 | 0.075 | 26.15 dB | 0.997 |
| **Epoch 116 (Best)** | **2.012** | **0.679** | **0.069** | **`29.01 dB`** | **`0.998`** |

### 5.2 Qualitative Image Showcase

![Inpainting Test Results](assets/inpainting_test_results.png)  
*Figure 1: Qualitative evaluation showing Masked Input, Inpainted Output, and Ground Truth.*

![Soft Compositing Feathering](assets/soft_composite_feathering.png)  
*Figure 2: Edge feathering comparison demonstrating smooth boundary transitions.*

### 5.3 Key Findings
1. **Perceptual Loss Impact**: Adding VGG-16 perceptual loss significantly reduced blurring in the reconstructed region, recovering sharp eye/nose details.
2. **Spectral Normalization Impact**: Prevented discriminator dominance; discriminator loss stabilized near \(\ln(2) \approx 0.693\), proving healthy equilibrium in the GAN game.
3. **Soft Composite Feathering**: Completely eliminated visible square border artifacts around the \(64 \times 64\) inpainting box.

---

## 6. Conclusion & Future Work

In this project, we successfully implemented, trained, and evaluated an enhanced cGAN image inpainting system. Reaching **29.01 dB PSNR** and **0.998 SSIM** across 116 epochs demonstrates that the combination of UNet skip-connections, Spectral Normalization, VGG-16 perceptual loss, and soft composite feathering yields state-of-the-art results for facial image restoration.

**Future Directions**:
- Extend mask generation to free-form continuous brush stroke masks using Partial Convolutions (PConvs).
- Integrate Transformer-based attention blocks (such as SwinIR) into the bottleneck of the Generator for global context reasoning.

---

## References
1. Goodfellow, I., et al. (2014). *Generative Adversarial Nets*. Advances in Neural Information Processing Systems (NIPS).
2. Mirza, M., & Osindero, S. (2014). *Conditional Generative Adversarial Nets*. arXiv preprint arXiv:1411.1784.
3. Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2016). *Image-to-Image Translation with Conditional Adversarial Networks*. IEEE CVPR.
4. Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition*. arXiv:1409.1556.
