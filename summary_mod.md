# Summary of Research Paper

# Summary of "Underwater Image Enhancement Using FPGA-Based Gaussian Filters with Approximation Techniques" (July 31, 2025)

---

### 1. **Introduction / Abstract**

The paper addresses the challenge of **underwater image degradation** caused by natural phenomena like absorption and scattering, which introduce haze and noise, reducing visual quality. It proposes a novel method to enhance underwater images by improving the efficiency of **Gaussian filters** used for noise reduction. The key innovation is implementing a **pipeline structure** for the Gaussian filter on FPGA hardware and employing **approximate adders** to optimize performance. Simulation results show a **speed increase over 150%** and **power consumption reduction exceeding 34%**, though at the cost of increased spatial resource usage. The design is particularly suited for **error-resilient applications** such as image and video processing where speed and power savings outweigh slight quality loss[1].

---

### 2. **Methodology**

- The approach uses a **pipeline Gaussian filter (PGF)** architecture on FPGA, which divides the filtering process into stages to enable parallel processing and faster throughput.
- The design incorporates **approximate adders**—simplified arithmetic units that trade some accuracy for reduced power and delay—to accelerate the Gaussian filter.
- The system uses **line buffers and window buffers** to store image pixels efficiently during convolution, minimizing memory access.
- Ten different types of approximate full adders (APFAs) are evaluated, each with varying degrees of approximation affecting sum and carry outputs.
- The approximation is applied selectively to the least significant bits (LSBs) of 16-bit adders, balancing output quality and hardware efficiency.
- The method is validated by simulating Gaussian noise addition to real underwater images and comparing output quality metrics such as PSNR, SSIM, and error distances[1].

---

### 3. **Theory / Mathematics**

- The **Gaussian filter** is based on the 2D Gaussian function:

  \[
  f(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}
  \]

  where \(\sigma\) controls the spread of the filter.
  
- The filter acts as a **low-pass smoothing operator**, reducing high-frequency noise while preserving edges.
- The convolution operation with a 3×3 kernel is expressed as:

  \[
  h[i,j] = \sum_{k=1}^9 W_k P_k
  \]

  where \(W_k\) are kernel weights and \(P_k\) are pixel values in the window.
  
- Approximate adders reduce logic complexity by simplifying sum and carry calculations, with different APFAs defined by specific logical functions (see Table 1 in the paper).
- Image quality metrics used include:

  - **PSNR (Peak Signal-to-Noise Ratio)**: measures signal fidelity.
  - **MSE (Mean Squared Error)**: quantifies average pixel error.
  - **SSIM (Structural Similarity Index)**: assesses perceptual similarity.
  - **Error Distance (ED), Mean Error Distance (MED), Normalized Error Distance (NED)**: measure spatial pixel discrepancies.
  
- These metrics provide a comprehensive evaluation of the trade-off between approximation and image quality[1].

---

### 4. **Key Diagrams or Visual Elements**

![Figure 1](output_images/figure_1.png)
- **Figure 1:** Delay line buffer structure showing how row buffers and window buffers store pixels for convolution, optimizing memory access.
![Figure 2](output_images/figure_2.png)
- **Figure 2:** Illustration of the convolution operation on an image using a 3×3 Gaussian kernel.
![Figure 3](output_images/figure_3.png)
- **Figure 3:** The weighted 3×3 Gaussian filter kernel applied to image pixels.
![Figure 4](output_images/figure_4.png)
- **Figure 4:** Block diagram of the Gaussian filter implementation, highlighting adders and shifters used for multiplication/division by powers of two.
![Figure 5](output_images/figure_5.png)
- **Figure 5:** Pipeline Gaussian filter architecture divided into four stages to enhance processing speed.
![Figure 6](output_images/figure_6.png)
- **Figure 6:** 16-bit carry ripple adder structure used to implement approximate adders.
![Figure 7](output_images/figure_7.png)
- **Figure 7:** Two raw underwater images (flatfish and diver) used for simulation, along with their histograms showing pixel intensity distributions.
![Figure 8](output_images/figure_8.png)
- **Figure 8:** Graphs showing image quality metrics (PSNR, SSIM, etc.) for different approximate adders and bit approximations, demonstrating acceptable quality up to 5-6 bits of approximation.
![Figure 9](output_images/figure_9.png)
- **Figure 9:** Power-Delay Product (PDP) comparison for different approximate Gaussian filters, identifying APFA4, APFA5, and APFA6 as most power-efficient.
- **Tables 1-3:** Logical functions of approximate adders, and synthesis results comparing power, delay, and area for exact and approximate Gaussian filters, showing trade-offs between speed, power, and spatial requirements[1].

---

### 5. **Conclusion**

The study presents a **pipeline Gaussian filter architecture** enhanced with **approximate adders** for underwater image noise reduction on FPGA. This design achieves:

- Over **150% speed improvement**.
- More than **34% power consumption reduction**.
- Increased spatial resource usage as a trade-off.

The approach is well-suited for **error-resilient applications** like underwater image and video processing, where slight quality degradation is acceptable in exchange for faster and more power-efficient processing. The paper highlights the importance of balancing approximation level and output quality, recommending up to 5-6 bits of approximation for optimal results. This work contributes a practical hardware solution advancing underwater image enhancement techniques[1].