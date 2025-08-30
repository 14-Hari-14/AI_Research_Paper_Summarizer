# Low Pass Filter

_Summary Date: August 30, 2025_

### Introduction / Abstract

The core problem addressed is the limitation of conventional attention modules in neural networks, which often require additional subnetworks, are restricted to channel or spatial dimensions, and use complex operations that increase computational complexity.

The proposed solution is a method that incorporates a **SimAM** (a non-parametric, energy-based attention mechanism that generates 3D weights based on neuroscience theory) module into each ConvBlock of a network to enhance feature extraction. This method also utilizes a skip connection to fuse the raw underwater image with the impulse response of a Low-Pass Filter (LPF). This approach differs from a similar network, Shallow-UWnet, by using 58 features in the last convolutional layer of each block, compared to Shallow-UWnet's 61.

The main finding mentioned is that incorporating the **SimAM** module provides a "powerful feature extraction ability" and is presented as a way to overcome the complexity and limitations of other attention mechanisms.

A tradeoff is noted: using **SimAM** with limited image datasets can negatively impact the model's generalization ability.

### Methodology

The methodology begins by creating an impulse response for a Low-Pass Filter (LPF). First, the size of the LPF's impulse response is specified based on the input image size. Then, the corresponding frequency response is applied, with the context providing equations for three filter types (DLPF, GLPF, and BLPF) using a cut-off angular frequency of π/2 and a filter order of 1. The impulse response is finalized using the **inverse Fourier transform** (a technique that converts the frequency response into a spatial domain image).

After creating the impulse response, a skip connection fuses the raw underwater image with the LPF's impulse response inside each ConvBlock of the network.

To improve feature extraction, the **SimAM** (a non-parametric, energy-based attention mechanism that generates 3D weights) is incorporated into each ConvBlock. This module is based on the neuroscience theory that the most significant neurons are those with the lowest energy, making them the most separable from neighboring neurons.

Finally, the proposed method differs from a model called Shallow-UWnet in that the last Conv layer of each ConvBlock uses 58 features.

### Theory / Mathematics

The core theoretical and mathematical principles discussed in the context relate to Low-Pass Filters (LPFs) and the SimAM attention module.

### Low-Pass Filters (LPFs)

The context describes three types of LPFs used to generate an impulse response from a frequency response. The impulse response is then converted into a spatial domain image via the inverse Fourier transform. The cut-off angular frequency, $$ \omega_c $$, is set to $$ \pi/2 $$, and the filter order, $$ k $$, is set to 1.

1.  **DLPF (Ideal Low-Pass Filter):** This filter sharply cuts off frequencies above the cut-off frequency. Its frequency response $$ H_D(\omega_1, \omega_2) $$ is defined as:

    $$
    H_D(\omega_1, \omega_2) =
    \begin{cases}
    1, & \sqrt{(\omega_1^2 + \omega_2^2)} \le \omega_c \\
    0, & \sqrt{(\omega_1^2 + \omega_2^2)} > \omega_c
    \end{cases}
    $$

    This formula states that the filter allows all frequencies within the radius of the cut-off frequency $$ \omega_c $$ to pass through with a gain of 1, while completely blocking all frequencies outside this radius.

2.  **GLPF (Gaussian Low-Pass Filter):** This filter provides a smoother transition between the passband and stopband compared to the DLPF. Its frequency response $$ H_G(\omega_1, \omega_2) $$ is given by:

    $$
    H_G(\omega_1, \omega_2) = e^{-(\omega_1^2 + \omega_2^2) / 2\omega_c^2}
    $$

    This formula uses a Gaussian function to attenuate frequencies. Frequencies closer to zero have a response near 1, and the response decreases smoothly as the distance from the origin increases.

3.  **BLPF (Butterworth Low-Pass Filter):** This filter also offers a smooth transition and is defined by its order, $$ k $$. Its frequency response $$ H_B(\omega_1, \omega_2) $$ is:
    $$
    H_B(\omega_1, \omega_2) = \frac{1}{1 + \left[ \frac{\sqrt{(\omega_1^2 + \omega_2^2)}}{\omega_c} \right]^{2k}}
    $$
    In this formula, the filter's response gradually decreases from 1 to 0. The rate of this transition is controlled by the filter order $$ k $$, which is specified to be 1.

### SimAM Attention Module

SimAM is described as a non-parametric, energy-based attention mechanism. Its theoretical foundation comes from neuroscience, which posits that the most significant neurons are those with lower energy because they are the most separable from neighboring neurons. The minimum energy of a target neuron $$ T $$ is calculated with the following formula:

$$
\epsilon_T = \frac{4}{(\rho^2 + \alpha)} (T - \eta)^2 + 2\rho^2 + 2\alpha
$$

In this equation:

- $$ \epsilon_T $$ represents the lower energy of the target neuron $$ T $$.
- $$ T $$ is the target neuron.
- $$ \eta $$ is the mean of the neurons.

### Key Diagrams or Visual Elements

**Figure 1:** Based on the provided context, Figure 1 is a schematic diagram that illustrates the process of underwater imaging. It depicts how light in an underwater environment encounters suspended particles within the water medium before reaching a camera. This interaction alters the direction of the light, leading to image blurring, low contrast, and a fog-like effect. The diagram represents the Jaffe-McGlamery model, which breaks down the final underwater image captured by the camera (UT) into three components: direct attenuation (Ud), which is light reflected by an object without scattering, and the forward (Ufs) and backward (Ubs) scattering components. (see: figure_1.png)

**Figure 2:** Based on the provided context, Figure 2 illustrates a comparison between two neural network architectures: (a) the conventional Shallow-UWnet and (b) the proposed method.

**Figure 2(a): Conventional Shallow-UWnet**
This diagram depicts the architecture of the standard Shallow-UWnet. The process begins with a raw underwater image as input. This input is processed by an initial convolutional (Conv) layer that uses 64 feature maps and a kernel size of 3x3, followed by a ReLU activation function. The architecture then features three consecutive convolutional blocks (ConvBlocks). Each ConvBlock consists of a Conv layer, a ReLU activation function, and a drop-out regularization technique. A skip connection links the raw input image to these ConvBlocks. The text specifies that in this conventional model, the final Conv layer within each ConvBlock utilizes 61 features. After passing through the ConvBlocks, the data goes through a final Conv layer to produce the enhanced underwater image as the output.

**Figure 2(b): Proposed Method**
This part of the figure shows the architecture of the proposed method, which is a modification of the Shallow-UWnet. The key difference highlighted in the text is the incorporation of a SimAM (Simultaneous Attention Module) into each of the ConvBlocks. This addition is intended to provide a more powerful feature extraction capability. Another specified difference is the number of features in the last Conv layer of each ConvBlock; the proposed method uses 58 features, in contrast to the 61 used by the conventional Shallow-UWnet.

(see: figure_2.png)

**Figure 3:** Based on the provided context, Figure 3 is titled "Power spectrum sparsity of image". The text does not offer any further visual description or explanation of what the figure depicts. The caption is placed in the middle of a sentence that is defining variables for an equation related to the SimAM attention module, where `yi` denotes other neurons within a single channel, `i` is the spatial dimension, and `N` is the total number of neurons (see: figure_3.png).

**Figure 4:** Based on the provided context, Figure 4 is a visual comparison of different underwater image enhancement methods applied to three datasets: EUVP-Dark, UFO-120, and UIEB, which are presented from top to bottom.

The figure displays the results of several methods side-by-side for comparison. The methods shown are:

- Raw Input Image
- WaterNet
- FUnIE-GAN
- Shallow-UWnet
- The Proposed method (SLPF)
- Ground Truth (the reference image)

Each resulting image in the comparison includes its corresponding Peak Signal-to-Noise Ratio (PSNR) value.

The context provides specific observations regarding the results on the EUVP-Dark Dataset as seen in the figure:

- **WaterNet:** Produces images with artificial colors and noise artifacts in both blue and green-hued images when compared to the proposed method.
- **FUnIE-GAN:** Shows incorrect color corrections, which is particularly evident in an image of a fish's fin.
- **Shallow-UWnet and the proposed method:** Both are described as having a noticeable effect on the images.

(see: figure_4.png)

### Conclusion

The researchers successfully developed a lightweight and eﬀicient AI model that
significantly enhances the quality of underwater images. By incorporating an
impulse response from a low-pass filter and a simple attention module (SimAM),
their model effectively reduces noise, corrects color distortion, and improves
image clarity. The results show that it performs comparably to much larger and
more computationally expensive models while being significantly smaller and
faster
