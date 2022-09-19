# Digital-Image-Processing

## About The Project

The following algorithms were implemented:


### Assignment 1 (Segmentation by color)

Contours search using the Suzuki algorithm (Suzuki S. et al. Topological structural analysis of digitized binary images by border);

- After finding contours, they were obtained by bounding boxes;

- In the result image only green toys should be obtained by bounding boxes.

Result:
![image](https://user-images.githubusercontent.com/113569606/191006904-8eebb249-43ec-484e-8fb9-5b325a812e99.png)


### Assignment 2

Contrast Limited Adaptive Histogram Equalization Algorithm (CLAHE), 

The paper: 

    Pizer S. M. et al. Adaptive histogram equalization and its variations;


### Assignment 3

Otsu and Sauvola binarization algorithms;

Results examples:
- Otsu binarization
![image](https://user-images.githubusercontent.com/113569606/191007222-50fc2085-9870-4717-a6be-91e3054dfbe7.png)

- Sauvola binarization
![image](https://user-images.githubusercontent.com/113569606/191007291-68d66025-c7d7-4861-b7de-835574f5fca9.png)

All results are in:

    assignment-3/

### Assignment 4

Digital Image Forgery Detection Using JPEG Features and Local Noise Discrepancies 

The paper: 
    
    https://www.hindawi.com/journals/tswj/2014/230425/


### Assignment 5

Canny Edge search algorithm, Two Pass Connected Components Labeling;


### Assignment 6

Algorithm for finding the “correct” text orientation using the Fourier transform for images.


## Getting Started

Implementation for each of assignments are in assignment-x directory;

Parameters of algorithms can be changed in:

        assignment-x/config.py
