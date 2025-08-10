Underwater Image Restoration Through a Prior Guided Hybrid Sense Approach and Extensive Benchmark Analysis  
*Summarized on August 5, 2025*

1. **Introduction / Abstract**  
Underwater images often suffer from color distortions (mainly blue or green hues) and blurriness due to how water absorbs and scatters light, especially because red light is absorbed fastest. These issues reduce image clarity and hinder tasks like marine life recognition. The paper addresses this by proposing a new underwater image restoration framework called **GuidedHybSensUIR**. This framework combines multi-scale processing to restore fine details and correct color casts, guided by a novel **Color Balance Prior** (an assumption that the average color in a natural scene should be neutral gray). The method outperforms 37 state-of-the-art techniques on multiple real-world datasets, providing a new benchmark for underwater image restoration.

2. **Methodology**  
The researchers designed a **U-shaped neural network** architecture combining two main components:  
- **Detail Restorer**: Uses **CNNs (Convolutional Neural Networks, a type of deep learning model good at capturing local image features)** to recover fine, blurry details at multiple scales. It employs quaternion convolutions (which treat RGB color channels as parts of a quaternion to better capture color interdependencies) and two specialized blocks: Residual Context Block (RCB) and Nonlinear Activation-Free Block (NAFB).  
- **Feature Contextualizer**: Uses **Transformers (a deep learning model that captures long-range dependencies)** to model global color relationships and contextual information. It applies inter-channel attention (focusing on relationships between color channels rather than image patches) and is guided strongly by the Color Balance Prior.  
Additionally, **Scale Harmonizer** modules in the decoder fuse multi-scale features smoothly. The Color Balance Prior is integrated both in the Feature Contextualizer and the final decoding stage to guide color correction.

3. **Theory / Mathematics**  
The Color Balance Prior is based on the **Gray World Assumption**, which states that in a well-lit scene, the average intensities of the red, green, and blue channels should be equal, representing a neutral gray. Mathematically, the observed pixel intensity in channel $i$ at position $(x,y)$ is:  
$$ f_i(x,y) = G(x,y) R_i(x,y) I_i(x,y) $$  
where $G$ is geometry, $R_i$ is reflectance (true color), and $I_i$ is illumination. Assuming constant geometry and reflectance, the average intensity $a_i$ over the image relates to illumination as:  
$$ a_i \approx I_i \times \text{constant} $$  
In air, $I_R = I_G = I_B$, so:  
$$ a_R \approx a_G \approx a_B $$  
The prior is defined as the average of the three channels:  
$$ \text{Prior}_i(x,y) = \frac{R(x,y) + G(x,y) + B(x,y)}{3} $$  
This prior guides the network to restore balanced colors by compensating for underwater color attenuation.

The network’s attention mechanism uses inter-channel cross-attention defined as:  
$$ \text{InterCAttn}(Q,K,V) = \text{Softmax}\left(\frac{\hat{Q} \cdot \hat{K}^\top}{\tau}\right) \cdot \hat{V} $$  
where $Q$, $K$, and $V$ are query, key, and value tensors derived from image and prior features, and $\tau$ is a learnable temperature parameter controlling attention sharpness.

4. **Key Diagrams or Visual Elements**  
![Figure 1](output_images/figure_1.png)
- **Figure 1:** Shows 3D scatter plots of color distributions for an input underwater image, the color balance prior, and the restored output. The prior lies centrally in the restored image’s color distribution, indicating its effectiveness in guiding color correction.  
![Figure 2](output_images/figure_2.png)
- **Figure 2:** Illustrates the overall U-shaped GuidedHybSensUIR architecture, highlighting the Detail Restorer modules (for local detail recovery), the Feature Contextualizer (for global color correction), and Scale Harmonizers (for multi-scale feature fusion).  
- **Figure 3 & 4:** Detail the Nonlinear Activation-Free Block (NAFB) and Residual Context Block (RCB), showing how they process features to restore details.  
- **Figure 5 & 6:** Depict the Feature Contextualizer’s Multi-Attention Quaternion (MAQ) blocks, combining three types of Transformers (Adjust Color Transformer, Keep Feature Transformer, and Self-Attention Transformer) to capture different attention aspects between image and prior features.  
![Figure 7](output_images/figure_5.png)
- **Figure 7:** Shows the architecture of the Adjust Color Transformer, explaining how it uses the color balance prior to adjust image features via inter-channel attention.  
![Figure 8](output_images/figure_6.png)
- **Figure 8:** Describes the Scale Harmonizer module, which calibrates and fuses features from different scales during decoding.

5. **Conclusion**  
The proposed GuidedHybSensUIR framework effectively restores underwater images by combining local detail restoration (via CNN-based Detail Restorer) and global color correction (via Transformer-based Feature Contextualizer) guided by a novel Color Balance Prior. Extensive experiments on a newly compiled benchmark dataset demonstrate that this method outperforms 37 existing state-of-the-art underwater image restoration techniques across multiple metrics and datasets. The approach balances computational efficiency with superior restoration quality, improving color accuracy and detail clarity. This research advances underwater imaging, which is crucial for marine exploration, environmental monitoring, and underwater robotics, by providing clearer, more natural images that enhance subsequent visual analysis tasks.