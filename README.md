# COVID-19 Detection
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?style=flat-square)](https://github.com/armiro/Covid19-Detection/blob/master/LICENSE)
![license](https://img.shields.io/badge/development-40%25-yellow?style=flat-square)

Detecting the novel coronavirus (aka 2019-nCov or COVID-19) from frontal chest X-ray images 
using deep convolutional neural nets

## Data Collection
Chest X-ray Images of patients with PCR-positive COVID-19 are collected from different sources, such as [SIRM](https://www.sirm.org/category/senza-categoria/covid-19/).
Please cite this repo if you are using CXR images:
```
@article{armiro2020covidcxr,
  title={COVID-19 Chest X-ray Image Data Collection},
  author={Arman Haghanifar},
  url={https://github.com/armiro/Covid19-Detection},
  year={2020}
}
```
There are currently ~410 images with different sizes and formats, and the data will be updated regularly. Metadata will be added as soon. **This dataset is the largest to best of my knowledge, as of 26/Apr/2020.** Normal CXRs are collected from different datasets, without a periatric image bias. To get the complete dataset consisting of 410 COVID-19 pneumonia and 600 normal images, [send me an email](mailto:arman@haghanifar.com).

## Project Progress
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
