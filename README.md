<div align="center">
    <h1> TEXTIFY </h1>
    Midterms mini-project of Computer Vision with Python implementation.
</div>

---

## Project Overview
This project is about making an simple cam-scanner from computer 
vision algorithm with implementation of python-opencv2 library. 
So the user can make use of this simple program to turn any 
handwritting to be an digital scanned format.


## Requirement
The requirement to run this project is to have python installed 
and to make sure it is installed or not, you can follow these 
command.
- windows : 
```bash
python --version
```
- unix based os (mac & linux) :
```bash
python3 --version
```
As for the version, it was build & compatible with 3.14+ python version.

## Architechture
This is our (current) project architechture.
```
Textify/                        : Root folder of this project.
├── docs/                           : Non-code files contains.
│   ├── flowchart/                      : Contains flowchart of the program.
│   ├── research/                       : Contains our research paper.
|   └── results/                        : Contains scan results from the program.
├── src/                            : Source code of the program.
├── test_unit/                      : Testing module for the program.
├── .gitignore
├── requirement.txt
├── LICENSE.md
└── README.md
```


## Build & Execution
First of all, you must download requirement of the library its needed, 
for that you can use this following command :
```bash
pip install -r requirement.txt
``` 
And don't worry, it was compatible with all the os, just one of python 
deffault command.
To use of this program, you can follow these command :
- windows :
```bash
python main.py
```
- unix based os (macOS & linux) :
```bash
python3 main.py
```
nb.
And i expect, you were on the root of the folder.

## Verification & Testing
To see and check if the program has any bugs and errors, you can use 
these command to see if anything broken.
- windows :
```bash
python test_unit/unit_test.py
```
- unix based os (macOS & linux) :
```bash
python3 test_unit/unit_test.py
```
nb.
And i expect, you were on the root of the folder.

## Experimental Results
As for our results wouldn't be included in the `docs/results/' folder 
as is most of our samples were private work documents, so you could 
try it yourself, go give it a try! :D


## Limitations & Risks
Since this is an early mini-project for our semester 3 Computer Vision 
class, there are still some limitations and risks that we noticed while 
testing:
- **Lighting Conditions**: If the room is way too dark or has extreme 
harsh glare, the edge detection might get confused finding the 4 corners 
of the paper.
- **Background Contrast**: The paper needs to be placed on a background 
that has a different color (like a dark desk). If you put white paper 
on a white table, the algorithm will struggle to separate the borders.
- **Finger Detection**: The skin color masking works pretty well under 
normal yellowish or daylight lamps, but under weird lighting (like RGB 
LEDs or very dim blueish light), it might miss some finger spots or 
accidentally blur dark ink that looks similar to shadows.
- **Webcam Resolution**: Cheap laptop webcams usually have blurry focus 
for tiny handwritten notes, so using good lighting or feeding clearer 
photos from the file option gives much crisper scans.


## References
Our main inspiration and benchmark for this project is:
- **CamScanner (INTSIG Information Co., Ltd.)**: A widely used mobile 
scanner app. We basically wanted to prove to ourselves whether computer 
science students like us could replicate its core magic—such as 
auto-detecting page borders, perspective straightening, and turning 
messy paper into clean black-and-white text—using purely classical 
OpenCV algorithms without relying on heavy paid cloud APIs.
- OpenCV Official Documentation on Contour Approximations, Perspective 
Transform, and Adaptive Thresholding.


## Research Paper
This is our simple [paper research](./docs/research/paper.md) of this 
project.

---

<div align="center">
    Made with confusion and lot of tea :D
</div>