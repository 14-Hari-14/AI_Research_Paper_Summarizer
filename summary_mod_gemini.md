# LIGHTWEIGHT UNDERWATER IMAGE ENHANCEMENT VIA IMPULSE RESPONSE OF LOW-PASS FILTER BASED ATTENTION NETWORK
*Summary generated on October 29, 2024*

### 1. Introduction / Abstract

Underwater images are often blurry, color-distorted, and have low contrast. This is because water absorbs and scatters light, especially red light, which gives images a characteristic blue-green tint and a foggy appearance. This poor image quality is a major problem for ocean engineering, marine biology research, and underwater robotics.

This paper proposes a new, improved Artificial Intelligence (AI) model to enhance these underwater images. The model is designed to be "lightweight," meaning it is small, fast, and doesn't require a lot of computational power. It improves upon an existing model (called Shallow-UWnet) by adding two key features:
1.  A special filtering technique that helps the model better distinguish between image details and random noise.
2.  An "attention module" that allows the model to focus on the most important parts of the image.

The main finding is that this new model produces enhanced images that are comparable in quality to those from much larger, more complex state-of-the-art methods. Its key advantage is its efficiency—it has fewer parameters and processes images very quickly. This makes it ideal for real-time applications on resource-constrained devices like underwater drones or robots.

### 2. Methodology

The researchers built their model by modifying an existing lightweight model called Shallow-UWnet. Here are the key steps they took:

1.  **Start with a Base Model:** They used Shallow-UWnet, a type of **Convolutional Neural Network (CNN)** (an AI model designed for processing images by applying successive filters), as their foundation.
2.  **Introduce a Smarter Skip Connection:** The original model used "skip connections" (shortcuts that pass information from early layers to later layers to prevent details from being lost). The researchers modified this by feeding not only the raw underwater image but also an **impulse response of a low-pass filter (LPF)** into the network. An LPF is a filter that removes high-frequency signals, which often correspond to noise, while keeping low-frequency signals, which correspond to the main image structure. The "impulse response" is essentially a fingerprint of this filter, which gives the AI a clue about how to separate noise from the actual image content.
3.  **Add an Attention Module:** They integrated a simple, parameter-free attention module called **SimAM**. This module works by analyzing the features within the image and assigning an "importance" score to different neurons (the basic processing units of the network). This allows the model to automatically focus its processing power on the most visually significant areas of the image, improving clarity and detail without adding computational bulk.
4.  **Training and Testing:** The model was trained on thousands of pairs of "bad" underwater images and their corresponding "good," clear versions. They then tested its performance against several other leading methods using standard underwater image datasets (EUVP-Dark, UFO-120, and UIEB).

### 3. Theory / Mathematics

The paper uses several mathematical concepts to model the problem and evaluate the solution.

- **The Jaffe-McGlamery Underwater Imaging Model:** This model describes the physics of how an underwater image is formed.

$$ U_T = U_d + U_{fs} + U_{bs} $$

**What it means:** The final image captured by the camera ($U_T$) is the sum of three components: the direct light reflected from the object ($U_d$), the light that is scattered on its way to the object, causing blur ($U_{fs}$), and the light from the surrounding water that is scattered back toward the camera, creating a foggy effect ($U_{bs}$). Understanding this model is key to reversing these effects.

- **SimAM Energy Function:** The SimAM attention module is based on a neuroscience theory that important neurons have lower "energy" because they are more distinct from their neighbors. The minimum energy is calculated as:

$$ \epsilon_T = \frac{4(\hat{\rho}^2 + \alpha)}{(T - \hat{\eta})^2 + 2\hat{\rho}^2 + 2\alpha} $$

**What it means:** This equation calculates an energy value ($\epsilon_T$) for a target neuron ($T$). It uses the variance ($\hat{\rho}^2$) and mean ($\hat{\eta}$) of all neurons in a channel. By finding the neurons with the lowest energy, the model can identify the most important information in the image. The model then refines the image by giving more weight to these low-energy, high-importance features.

- **Underwater Image Quality Measure (UIQM):** Since it's often impossible to have a "perfect" reference image in the real world, this metric is used to score the quality of an enhanced image based on how a human would perceive it.

$$ UIQM = c_1 \times UICM + c_2 \times UISM + c_3 \times UIConM $$

**What it means:** The UIQM score is a weighted sum of three factors: color balance (UICM), sharpness (UISM), and contrast (UIConM). A higher UIQM score generally means the image is more visually appealing, with vibrant colors, sharp details, and good contrast.

### 4. Key Diagrams or Visual Elements

![Figure 2](output_images/figure_2.png)
- **Figure 2:** This diagram shows the architecture of the AI models. It places the original Shallow-UWnet model next to the new, proposed model. The visual makes it clear what the researchers added: the "Impulse Response of LPF" is now fed into the network alongside the raw image, and a "SimAM" block has been inserted into each **convolution** block (a fundamental layer in a CNN that applies a filter to an image). This highlights the two core innovations of their method.

![Figure 4](output_images/figure_4.png)
- **Figure 4:** This figure provides a powerful visual comparison of the results. For three different datasets, it shows the original blurry input image, the results from two other popular methods (WaterNet, FUnIE-GAN), the result from the original Shallow-UWnet, the result from the new proposed method, and the ideal "ground truth" image. **Significance:** This figure visually proves the paper's claims. The images produced by the proposed method are visibly clearer, have better color correction, and are less noisy than the results from the other methods, often looking very close to the ground truth.

- **Table 1 & 2:** These tables present the quantitative results.
    - **Table 1** compares the performance of the proposed model against eight other methods using numerical scores (PSNR, SSIM, and UIQM). It shows that the new model achieves scores that are comparable to, and in some cases better than, the competition, especially its predecessor, Shallow-UWnet.
    - **Table 2** focuses on efficiency. It lists the number of trainable parameters (a measure of model size) and the time it takes to process a single image. **Significance:** This table is crucial because it shows that the proposed model is one of the smallest (only ~216,000 parameters) and fastest, reinforcing its suitability for real-time applications.

### 5. Conclusion

The researchers successfully developed a lightweight and efficient AI model that significantly enhances the quality of underwater images. By incorporating an impulse response from a low-pass filter and a simple attention module (SimAM), their model effectively reduces noise, corrects color distortion, and improves image clarity. The results show that it performs comparably to much larger and more computationally expensive models while being significantly smaller and faster.

**Why It Matters:** This research provides a practical solution for improving image quality in real-time on devices with limited processing power, such as autonomous underwater robots used for ocean exploration, marine biology research, and infrastructure inspection.