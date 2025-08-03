**Underwater Image Enhancement Using FPGA-Based Gaussian Filters with Approximation Techniques**  
*Summary Date: August 3, 2025*

1. **Introduction / Abstract**  
The paper addresses the problem of poor visual quality in underwater images caused by natural effects like light absorption and scattering, which create haziness and noise. To improve these images, the study proposes an efficient method using **Gaussian filters** (a mathematical tool for smoothing images and reducing noise) implemented on **FPGA** (Field-Programmable Gate Array, a type of reconfigurable hardware that allows fast and parallel processing). The key innovation is a pipeline structure combined with approximate adders (simplified arithmetic units that trade some accuracy for better speed and lower power use). The main findings show that this approach speeds up processing by over 150% and reduces power consumption by more than 34%, though it requires more hardware area. This tradeoff is suitable for error-tolerant applications like image and video processing.

2. **Methodology**  
The researchers designed a hardware architecture for the Gaussian filter on an FPGA platform. They used a pipeline structure, which breaks the filtering process into stages that operate simultaneously to increase speed. The filter uses a 3×3 kernel (a small matrix applied to each pixel and its neighbors) to smooth images and reduce noise. To further improve efficiency, they replaced exact adders with various types of approximate adders that simplify calculations by allowing small errors. They tested ten different approximate adder designs, focusing on how many least significant bits (LSBs) were approximated. The system was simulated in MATLAB with real underwater images corrupted by Gaussian noise, and then synthesized on an Intel MAX10 FPGA to measure power, speed, and area.

3. **Theory / Mathematics**  
The Gaussian filter is based on the two-dimensional Gaussian function:  
$$ f(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}} $$  
where $(x,y)$ are coordinates relative to the center pixel, and $\sigma$ controls the spread of the smoothing effect. This function weights pixels near the center more heavily, effectively blurring the image to reduce noise. The filter is separable, meaning a 2D convolution can be done as two 1D convolutions, improving efficiency. The convolution operation for a 3×3 kernel is:  
$$ h[i,j] = \sum_{m=1}^3 \sum_{n=1}^3 w[m,n] \cdot p[i+m-2, j+n-2] $$  
where $w$ is the kernel weight and $p$ is the pixel value.  

Image quality was evaluated using:  
- **PSNR (Peak Signal-to-Noise Ratio):** Measures how close the filtered image is to the original, higher is better.  
- **MSE (Mean Squared Error):** Average squared difference between original and filtered pixels, lower is better.  
- **SSIM (Structural Similarity Index):** Assesses perceived image quality considering luminance, contrast, and structure, higher is better.  
- **Error Distance (ED), Mean Error Distance (MED), Normalized Error Distance (NED):** Measure spatial differences between images to assess accuracy.

4. **Key Diagrams or Visual Elements**  
![Figure 1](output_images/figure_1.png)
- **Figure 1:** Shows the delay line buffer structure used to store rows of pixels for efficient convolution, minimizing memory access.  
![Figure 2](output_images/figure_2.png)
- **Figure 2:** Illustrates the convolution operation with a 3×3 kernel sliding over the image pixels to compute filtered values.  
![Figure 3](output_images/figure_3.png)
- **Figure 3:** Displays the weighted 3×3 Gaussian kernel used for smoothing.  
![Figure 4](output_images/figure_4.png)
- **Figure 4:** Block diagram of the Gaussian filter implementation, showing adders and shifters used for multiplication and division by powers of two.  
![Figure 5](output_images/figure_5.png)
- **Figure 5:** Pipeline Gaussian filter architecture divided into four stages to increase processing speed by parallelizing operations.  
![Figure 6](output_images/figure_6.png)
- **Figure 6:** Diagram of a 16-bit carry ripple adder used in the design, highlighting where approximation is applied.  
![Figure 7](output_images/figure_7.png)
- **Figure 7:** Two example raw underwater images (a flatfish and a diver) with their histograms, used for testing.  
![Figure 8](output_images/figure_8.png)
- **Figure 8:** Graphs showing image quality metrics (PSNR, SSIM, etc.) for different approximate adders and bit approximations, indicating that up to 5-6 bits of approximation maintain good quality.  
![Figure 9](output_images/figure_9.png)
- **Figure 9:** Power-Delay Product (PDP) comparison for different approximate adders, identifying the most power-efficient designs.  
- **Tables 1-3:** Present logic functions of approximate adders, and synthesis results comparing power, speed, and area for exact and approximate Gaussian filters.

5. **Conclusion**  
The study successfully demonstrates that implementing a pipeline Gaussian filter on FPGA with approximate adders significantly improves processing speed (over 150%) and reduces power consumption (over 34%) for underwater image enhancement. The tradeoff is an increased hardware area and some loss in output quality, which remains acceptable up to about 5-6 bits of approximation. This makes the approach well-suited for error-resilient applications like underwater image and video processing, where faster and more energy-efficient hardware is critical. The results, supported by simulation and FPGA synthesis data, highlight the practical benefits of combining pipeline architectures with approximate computing in real-time image enhancement tasks. This research advances underwater imaging technology, enabling better monitoring and analysis of marine environments with efficient hardware solutions.