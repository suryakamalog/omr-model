# OMRScan
An application that lets you scan and grade OMR sheets quickly.

## Features
* Scan multiple OMR sheets at once.
* Handle multiple subjects through single app.
* Store all results in a downloadable format.
* Rescan of same paper won't create unnecessary documents.
* Change actual answers of question paper anytime.
* Easy re-evaluation process.


## Working of model

### Flowchart
![Flowchart](https://github.com/suryakamalog/omr-model/blob/master/readmeImages/model_flowchart.png)

### Processing Steps
#### 1. Upload Image
#### 2. Get Bird View
![Bird_view](https://github.com/suryakamalog/omr-model/blob/master/readmeImages/getBirdView.jpeg)
#### 3. Extract Boxes
![extract_boxes](https://github.com/suryakamalog/omr-model/blob/master/readmeImages/extract_boxes.PNG)
#### 4. Finding Bubble contours
![bubble_contours](https://github.com/suryakamalog/omr-model/blob/master/readmeImages/finding_bubble_contours.PNG)
#### 5. Evaluation using bit masking
![bit_masking_1](https://github.com/suryakamalog/omr-model/blob/master/readmeImages/evaluation_using_bitmasking_1.gif)
![bit_masking_1](https://github.com/suryakamalog/omr-model/blob/master/readmeImages/evaluation_using_bitmasking_2.gif)
#### 6. Result
![result](https://github.com/suryakamalog/omr-model/blob/master/readmeImages/result.png)


## Installation
### Install Dependencies


```bash
pip install opencv-python
pip install numpy
pip install scipy
pip install imutils
pip install argparse
```
### Usage

```bash

python3 main.py [--image path/to/image] [--template path/to/template.json] [--answer path/to/answer]
```
