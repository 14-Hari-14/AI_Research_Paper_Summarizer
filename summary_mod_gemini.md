# Underwater Image Enhancement Using FPGA-Based Gaussian Filters with Approximation Techniques
*Summary Date: October 26, 2023*

### 1. Introduction / Abstract

Underwater images often suffer from poor quality, appearing hazy, blurry, and with distorted colors. This is caused by how light scatters and gets absorbed in water, making it difficult to use these images for scientific research, like studying coral reefs or monitoring marine life.

This paper proposes a new method to clean up this "noise" in underwater images using a specialized hardware approach. The researchers designed a highly efficient **Gaussian filter** (a common image processing technique that blurs an image to reduce noise) to run on an **FPGA** (**Field-Programmable Gate Array**; a special type of computer chip that can be reconfigured for specific, high-speed tasks). Their design uses a "pipeline" structure, like an assembly line, to process images faster, and "approximate adders" (simplified circuits that trade a tiny bit of mathematical precision for speed and energy savings).

The main finding is that this new approach is remarkably effective, achieving a speed increase of over 150% and reducing power consumption by more than 34% compared to standard methods. However, this comes with a trade-off: the design requires more physical space on the chip, and the use of approximation slightly reduces the final image quality. This makes the solution ideal for applications where speed and battery life are more critical than perfect, pixel-by-pixel accuracy, such as in real-time video processing on underwater drones.

### 2. Methodology

The researchers followed a multi-step process to design and test their system:

1.  **Hardware Selection:** They chose to implement their filter on an **FPGA**. Unlike a general-purpose CPU in a laptop, an FPGA can be programmed to perform a specific task, like image filtering, with massive parallelism (doing many calculations at once), making it extremely fast and efficient for this kind of work.

2.  **Filter Architecture:** The core of their method is a **Gaussian filter**, which smooths an image by performing a **convolution** (**Convolution**; a mathematical process where a small grid of numbers, called a kernel, is slid across an image to calculate a new value for each pixel based on its neighbors). To make this process faster on the FPGA, they designed a "pipeline" architecture. This breaks the complex filter calculation into four smaller, sequential stages. As a piece of data finishes Stage 1, it moves to Stage 2, allowing a new piece of data to enter Stage 1. This assembly-line approach ensures the hardware is always busy and dramatically increases processing speed.

3.  **Approximate Computing:** To reduce power consumption, the researchers replaced the standard, precise addition circuits with ten different types of "approximate adders." These are simplified circuits that are faster and use less energy but may produce tiny errors in their calculations. The idea is that for image processing, these small errors are often unnoticeable to the human eye but provide significant gains in performance.

4.  **Testing and Evaluation:** They took two real underwater images (a flatfish and a diver) and digitally added Gaussian noise (a type of random static) to them. They then processed these noisy images using their new filter designs. To measure performance, they compared the output image quality against the original, noise-free image using several standard metrics. They also measured the actual speed, power consumption, and physical area their designs used on the FPGA chip.

### 3. Theory / Mathematics

The research is based on the principles of Gaussian filtering and image quality assessment.

The **Gaussian filter** uses a mathematical function to create its blurring effect. The two-dimensional Gaussian function is given by:

$$f(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$$

*   **What it means:** This equation describes a bell-shaped curve in 3D. When used as a filter, it means the pixel at the very center of the filter's focus is given the most importance (the peak of the bell), while surrounding pixels are given less importance the farther they are from the center. The symbol `σ` (sigma) controls the width of the bell, which determines how much the image is blurred.

The filter is applied to the image through **convolution**. For a 3x3 filter, the new value of a pixel is a weighted average of itself and its eight immediate neighbors. This can be represented simply as:

$$h[i, j] = AP_1 + BP_2 + CP_3 + DP_4 + EP_5 + FP_6 + GP_7 + HP_8 + IP_9$$

*   **What it means:** The new pixel value `h[i, j]` is calculated by multiplying each of the nine pixels in a block (`P1` through `P9`) by their corresponding weights (`A` through `I` from the Gaussian kernel) and adding them all together.

To measure the quality of the final image, the researchers used several metrics, including:

*   **Peak Signal-to-Noise Ratio (PSNR):** This measures the ratio between the maximum possible value of a pixel and the amount of error, or noise. A higher PSNR value means the image is of higher quality.
    $$PSNR = 20 \log_{10} \left( \frac{MAX_I}{\sqrt{MSE}} \right)$$
*   **Mean Squared Error (MSE):** This calculates the average of the squares of the differences between the pixels of the original and the processed images. A lower MSE value means the processed image is closer to the original.
    $$MSE = \frac{1}{mn} \sum_{i=0}^{m-1} \sum_{j=0}^{n-1} [I(i,j) - K(i,j)]^2$$

### 4. Key Diagrams or Visual Elements

![Figure 1](output_images/figure_1.png)
-   **Figure 1: Delay line buffer structure:** This diagram shows an efficient memory structure used by the FPGA. Instead of constantly fetching data from a large memory bank, it stores just the few rows of pixels needed for the current calculation in small, fast-access buffers. This minimizes memory access time and is key to achieving high-speed processing.

![Figure 2](output_images/figure_2.png)
-   **Figure 2: Convolution Operation on an image with a 3x3 kernel:** This visual explains the core concept of the filter. It shows a 3x3 grid (the kernel) sliding over the image pixels. The value of the center pixel is replaced by a weighted average of itself and its neighbors, effectively smoothing the image.

![Figure 5](output_images/figure_5.png)
-   **Figure 5: Pipeline Gaussian filter:** This block diagram illustrates the "assembly line" design. The entire calculation is broken into four distinct stages. Data flows from one stage to the next, allowing different parts of the calculation for different pixels to happen simultaneously, which greatly increases the overall throughput.

![Figure 8](output_images/figure_8.png)
-   **Figure 8: Evaluation Metrics for Image Enhancement:** These graphs display the key experimental results. They plot image quality (using metrics like PSNR and SSIM) against the number of "approximate bits" used in the adders. The graphs show that image quality remains high for up to 5 or 6 approximate bits but then drops off sharply. This helps identify the "sweet spot" that balances performance gains with acceptable image quality. They also show that certain approximate adders (APFA9, APFA2) maintain much better quality than others (APFA1).

-   **Table 2 & 3: Comparison on physical properties:** These tables quantify the hardware performance. Table 2 shows that the pipelined filter is 48% faster and 12% more power-efficient than a non-pipelined version, but uses 27% more chip area. Table 3 details the performance of the different approximate adders, showing that some designs can boost speed by over 70% (APFA7) or reduce power by over 20% (APFA5), demonstrating the significant trade-offs between speed, power, and area.

### 5. Conclusion

The researchers successfully designed and demonstrated a novel hardware architecture for enhancing underwater images. By implementing a pipelined Gaussian filter with approximate computing techniques on an FPGA, they achieved dramatic improvements in performance, with processing speeds over 150% faster and power consumption reduced by over 34%.

The key takeaway is that for error-resilient tasks like image processing, sacrificing a small, often imperceptible, amount of accuracy can lead to massive gains in speed and energy efficiency. While this approach requires more physical chip area, it offers a powerful solution for applications where performance is paramount.

**Why It Matters:** This research provides a practical method for building high-speed, low-power vision systems for devices like autonomous underwater vehicles or aerial drones. In these real-world scenarios, the ability to process video in real-time and conserve battery life is often far more important than ensuring every single pixel is mathematically perfect.