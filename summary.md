# Summary of Research Paper

1. **Introduction / Abstract**

The paper addresses the challenge of underwater image degradation caused by natural phenomena such as absorption, scattering, and haziness, which reduce visual quality. It proposes a novel approach to enhance underwater images by improving the efficiency of Gaussian filters used to reduce Gaussian noise. The key innovation is implementing a pipeline structure for the Gaussian filter on FPGA hardware and employing approximate adders to optimize performance. Simulation results show a speed increase of over 150% and power consumption reduction exceeding 34%, though at the cost of increased spatial resource usage. The design is particularly suited for error-resilient applications like image and video processing where speed and power savings outweigh spatial constraints[1].

2. **Methodology**

The methodology involves designing a pipeline Gaussian filter (PGF) architecture on FPGA, which uses line buffers and window buffers to process image pixels sequentially with minimal memory usage. The Gaussian filter kernel is applied via convolution with a 3×3 mask, and the pipeline divides the computation into four stages to enhance throughput. To further improve efficiency, the study integrates various approximate full adders (APFAs) into the filter’s arithmetic units, reducing logic complexity and power consumption by allowing controlled approximation in addition operations. The approximation is applied selectively to the least significant bits of 16-bit adders, balancing output quality and hardware efficiency. The design is validated through MATLAB simulations and FPGA synthesis on Intel MAX10 devices[1].

3. **Theory / Mathematics**

- The Gaussian filter is based on the two-dimensional Gaussian function:

  \[
  f(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}
  \]

  where \(\sigma\) controls the spread of the Gaussian curve, affecting smoothing and noise reduction.

- The convolution operation applies a weighted sum of pixel intensities within the kernel window to compute each output pixel:

  \[
  h[i,j] = \sum_{k=1}^9 W_k P_k
  \]

  where \(W_k\) are kernel weights and \(P_k\) are pixel values in the 3×3 neighborhood.

- Approximate adders modify the logic of full adders to reduce transistor count and power, trading off some accuracy. Ten types of approximate full adders (APFA1 to APFA10) are characterized by their logic functions and error rates.

- Image quality metrics used to evaluate performance include:

  - Peak Signal-to-Noise Ratio (PSNR):

    \[
    PSNR = 20 \log_{10} \left(\frac{2^B - 1}{\sqrt{MSE}}\right)
    \]

  - Mean Square Error (MSE):

    \[
    MSE = \frac{1}{MN} \sum_{i=1}^M \sum_{j=1}^N (f_{ij} - g_{ij})^2
    \]

  - Structural Similarity Index (SSIM), which considers luminance, contrast, and structure similarity.

  - Error Distance (ED), Mean Error Distance (MED), and Normalized Error Distance (NED) measure spatial discrepancies between original and processed images.

- Power-Delay Product (PDP) is used to assess hardware efficiency, combining power consumption and processing delay[1].

4. **Key Diagrams or Visual Elements**

![Figure 1](output_images/figure_1.png)
- **Figure 1:** Delay line buffer structure showing the use of line buffers and window buffers to store pixels for convolution, minimizing memory access during filtering.

![Figure 2](output_images/figure_2.png)
- **Figure 2:** Illustration of the convolution operation on an image using a 3×3 Gaussian kernel, showing how pixel values are weighted and summed.

![Figure 3](output_images/figure_3.png)
- **Figure 3:** The weighted 3×3 Gaussian filter kernel applied to the image, indicating the specific weights used for smoothing.

![Figure 4](output_images/figure_4.png)
- **Figure 4:** Block diagram of the Gaussian filter implementation, detailing the use of adders and shifters for multiplication and division operations.

![Figure 5](output_images/figure_5.png)
- **Figure 5:** Pipeline Gaussian filter architecture with four stages, demonstrating how computation is divided to increase processing speed.

![Figure 6](output_images/figure_6.png)
- **Figure 6:** 16-bit carry ripple adder structure used to implement approximate adders by selectively approximating least significant bits.

![Figure 7](output_images/figure_7.png)
- **Figure 7:** Two raw underwater images (a flatfish and a diver) used for simulation, along with their histograms showing pixel intensity distributions.

![Figure 8](output_images/figure_8.png)
- **Figure 8:** Graphs showing evaluation metrics (PSNR, SSIM, etc.) for image enhancement quality using different approximate adders and bit approximations.

![Figure 9](output_images/figure_9.png)
- **Figure 9:** Power-Delay Product (PDP) values for various approximate Gaussian filters, highlighting the trade-offs between power efficiency and speed.

- **Tables 1-3:** Logical functions of approximate full adders, and comparative synthesis results showing power, delay, and area metrics for Gaussian filters with and without pipeline and approximation techniques[1].

5. **Conclusion**

The study presents a novel FPGA-based pipeline Gaussian filter architecture enhanced with approximate adders to improve underwater image enhancement efficiency. The pipeline structure combined with 2 to 8-bit approximation in adders achieves over 150% speed improvement and more than 34% power reduction, albeit with increased area requirements. The approach effectively balances the trade-off between output image quality and hardware performance, making it suitable for error-resilient applications such as underwater image and video processing. This work contributes a practical solution for accelerating underwater image enhancement while reducing power consumption on FPGA platforms[1].