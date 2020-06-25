# COVID-CXNet
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?style=flat-square)](https://github.com/armiro/Covid19-Detection/blob/master/LICENSE)
![license](https://img.shields.io/badge/development-100%25-yellow?style=flat-square)

Detecting the novel coronavirus pneumonia in frontal chest X-ray images using transfer learning of CheXNet with a focus on Grad-CAM visualiztions. Code repo for paper available on arXiv: [COVID-CXNet](https://arxiv.org/abs/2006.13807)

## Data Collection
Chest x-ray images of patients with (mostly) PCR-positive COVID-19 are collected from different publicly available sources, such as [SIRM](https://www.sirm.org/category/senza-categoria/covid-19/).
Please cite the associated paper if you are using CXR images. If this repo helped you with your research stuff, you can star it.
```
@article{haghanifar2020covidcxnet,
  title={COVID-CXNet: Detecting COVID-19 in Frontal Chest X-ray Images using Deep Learning},
  author={Arman Haghanifar, Mahdiyar Molahasani Majdabadi, Seokbum Ko},
  url={https://arxiv.org/abs/2006.13807},
  year={2020}
}
```
There are currently **~755** images with different sizes and formats, and the data will be updated regularly. Metadata will be added soon. **This dataset is the largest to the best of my knowledge, as of 23/Jun/2020.** Normal CXRs are collected from different datasets, without a pediatric image bias. Note that a `-` sign at the end of image name indicates that CXR did not reveal any abnormalities, but the patient had CT/PCR-proven COVID-19 infection (probably patient is in early stages of disease progression). Besides, a `p` letter at the end of image name means that the image is taken from pediatric patient.
