# covid19_detection
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?style=flat-square)](https://github.com/armiro/Covid19-Detection/blob/master/LICENSE)
![license](https://img.shields.io/badge/development-20%25-yellow?style=flat-square)

Detecting the novel coronavirus (aka 2019-nCov or CoVid19) from frontal chest X-ray images 
using deep convolutional neural nets

The project is in early stages and a pretty huge amount of new images are still in progress
Some test snapshots of the basic model on random data from the external dataset:
![case_1](https://github.com/armiro/Covid19-Detection/blob/master/documents/case%231.png)
![case_2](https://github.com/armiro/Covid19-Detection/blob/master/documents/case%232.png)
![case_3](https://github.com/armiro/Covid19-Detection/blob/master/documents/case%233.png)
![case_4](https://github.com/armiro/Covid19-Detection/blob/master/documents/case%234.png)

The covid19 viral pneumonia often has the image features of patchy consolidation, 
perihilar/peripheral distribution, and rarely pleural effusion. Hence, the features extracted by the 
base model, which are seen in grad-cam heatmaps, are wrong in more than half of the samples. One reason 
might be the fact that most normal samples are pediatrics. Even though, evaluating the model on external
dataset of 60 images resulted in this confusion matrix:

|        | normal | covid |
|--------|--------|-------|
| normal | 21     | 9     |
| covid  | 3      | 27    |

Dataset of normal CXR images must be recollected with images from adult lungs.
