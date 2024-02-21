# Image Processing

**Overview:**

This project, titled Image Processing, leverages the PIL, math, and random libraries to provide a versatile set of image modification functions. The primary codebase resides in the `lab.py` file, offering a variety of image processing capabilities.

**Functionality:**

The `lab.py` file encompasses numerous functions for image modification, including:

- Correlating
- Inverting
- Blurring
- Sharpening
- Box blurring
- Edge emphasis
- Converting from grayscale to color images
- Seam carving
- Sparkles
- Filter cascade: combines multiple filters into one simulating their cumulative effect.

**Testing:**

The `test.py` file features a comprehensive set of test cases, combining those created by the project author and those provided by the MIT 6.101 course staff. The `test_images` folder contains a collection of test images, and the `test_results` folder contains the correct solutions corresponding to the test images. The 'test_seam_carving_helpers.py' file was created by me and MIT 6.101 course staff to test the seam carving helper functions and ensure that they were working properly. 

**Getting Started:**

To use Image Processing in your projects, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/imageprocessing.git`
2. Navigate to the project directory: `cd imageprocessing`
3. Open the `lab.py` file to explore available functions for image processing.
4. Check the `test.py` file for usage examples and test cases.

**Usage:**

This project is particularly useful for effortlessly adding filters to images without modifying the original version. The `sparkle` filter, highlighted as a personal favorite, can also serve as a snow filter, transforming any image into a winter wonderland.
