# COVID-CXNet
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?style=flat-square)](https://github.com/armiro/Covid19-Detection/blob/master/LICENSE)
![license](https://img.shields.io/badge/development-100%25-yellow?style=flat-square)

Detecting the novel coronavirus (aka 2019-nCov or COVID-19) from frontal chest X-ray images 
using deep convolutional neural nets

## Data Collection
Chest X-ray Images of patients with PCR-positive COVID-19 are collected from different sources, such as [SIRM](https://www.sirm.org/category/senza-categoria/covid-19/).
Please cite this repo if you are using CXR images:
```
@article{haghanifar2020covidcxnet,
  title={COVID-CXNet: Detecting COVID-19 in Frontal Chest X-ray Images using Deep Convolutional Neural Networks},
  author={Arman Haghanifar, Mahdiyar Molahasani Majdabadi, Seokbum Ko},
  url={https://github.com/armiro/COVID-CXNet},
  year={2020}
}
```
There are currently **~720** images with different sizes and formats, and the data will be updated regularly. Metadata will be added soon. **This dataset is the largest to best of my knowledge, as of 01/Jun/2020.** Normal CXRs are collected from different datasets, without a pediatric image bias. Note that a `-` sign at the end of image name indicates that CXR did not reveal any abnormalities, but the patient had CT/PCR-proven COVID-19 infection (probably patient is in early stages od disease progression). Besides, a `p` letter at the ned of image name means that the image is taken from pediatric patient.
