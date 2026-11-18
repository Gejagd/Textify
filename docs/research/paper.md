<div align="center">
    <h1> TEXTIFY </h1>
    Simplified Cam-Scanner project with implementation.
</div>

---

## Abstraction
In everyday college life, students often need to submit physical 
handwritten notes or assignments into digital formats like PDF or 
images. However, taking photos directly with a regular phone or 
laptop camera often leaves unwanted shadows, uneven background 
lighting, tilted paper angles, or even fingers holding the paper edges. 
This project, titled TEXTIFY, is a simple document scanner desktop 
application developed using Python and OpenCV. The program detects 
document corners, applies perspective warping to flatten the page, 
cleans up small finger patches using skin masking and inpainting, and 
turns the document into a high-contrast black-and-white scanned result 
using adaptive thresholding. The final output is automatically saved 
as sequentially numbered PNG files. Overall, the program fulfills our 
midterm computer vision project objectives and works well for scanning 
standard documents.


## Introduction
Document scanner applications on smartphones, such as CamScanner or Adobe Scan, are very popular among university students. When we study and take notes on paper, converting those notes into neat digital files makes them much easier to read, archive, and submit to learning management systems.

When we simply take a picture of a paper on a desk, several common issues usually happen:
1. The paper position is often rotated or skewed.
2. The room lighting is rarely uniform, creating annoying shadows.
3. Our fingers frequently show up along the paper borders while trying to hold it flat.

Therefore, for this Computer Vision midterm assignment, we built TEXTIFY. Our main goal is to recreate a lightweight version of a document scanner from scratch using fundamental computer vision techniques in Python, focusing on modular programming, clean code structure, and reliable test verification.


## Methods
The processing flow of TEXTIFY consists of several main steps implemented in an object-oriented and modular pipeline:

1. **Image Acquisition**:
   The user can choose either to pick an existing image through a file explorer dialog or capture an image directly using an OpenCV camera preview. During camera preview, the program displays a real-time green box contour when a four-corner document is recognized.

2. **Document Detection and Perspective Transform**:
   - The input frame is converted to grayscale and smoothed with Gaussian blur ($5 \times 5$).
   - Canny edge detection finds the paper boundaries.
   - Using contour approximation (`approxPolyDP`), the largest 4-point polygon is extracted.
   - We calculate the perspective transform matrix and warp the image so the document becomes a straight rectangular view.

3. **Finger / Skin Removal**:
   - When holding documents, fingertips often appear on the paper edges.
   - We segment skin color in the HSV color space with a predefined range.
   - Morphological operations (opening and dilation) refine the skin mask.
   - Fast image inpainting (`cv2.inpaint` with Telea's method) is applied to restore the paper background behind the fingers before binarization.

4. **Binarization & Noise Cleaning**:
   - The straightened, cleaned image is transformed to grayscale and blurred.
   - We apply Adaptive Gaussian Thresholding to handle local illumination differences across the page, making text clearly black and the background pure white.
   - A morphological opening filter is applied to remove isolated salt-and-pepper noise pixels without damaging character strokes.

5. **Exporting Result**:
   The final black-and-white image is saved inside `docs/results/` using an automatic incremental naming scheme (e.g., `scan_result_01.png`, `scan_result_02.png`), ensuring previous scan files are never overwritten.


## Results
The results were quite as expected, where the algorithm output were decent enough for the assignment goal. As far as the result were quite the same as in our Computer Vision practices. 

When testing with sample handwritten pages under standard desk lamps, the adaptive thresholding successfully separated text ink from paper background even when soft shadows were present. The perspective warp also straightened tilted papers effectively, and the incremental file naming prevented any accidental file loss during multiple scan sessions.


## Conclusion
The algorithm itself was proving its capabilities, while utilies its limitation to be the only options of perfect results, as in a lot of Cam-Scanner apps. 

Through this project, we learned how classical computer vision techniques—such as edge detection, contour approximation, perspective transformation, color masking, and adaptive thresholding—can work together to build a functional, real-world utility application. For future improvements, adding automatic page orientation detection and support for multi-page PDF generation would make TEXTIFY even more useful.

---

<div align="center">
    (c) All right reserved. TEXTIFY Project.
</div>
