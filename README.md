# COVID-CXNet
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?style=flat-square)](https://github.com/armiro/COVID-CXNet/blob/master/LICENSE)
![license](https://img.shields.io/badge/development-100%25-yellow?style=flat-square)

Detecting the novel coronavirus pneumonia in frontal chest X-ray images using transfer learning of CheXNet with a focus on Grad-CAM visualiztions. 

- Preprint available on arXiv: [COVID-CXNet](https://arxiv.org/abs/2006.13807)
- Quick look at the final best results on paperswithcode: [leaderboard](https://paperswithcode.com/paper/covid-cxnet-detecting-covid-19-in-frontal)
- Article about detailed steps and challenges toward developing COVID-CXNet on Medium: [Thoughts on COVID-19 Pneumonia Detection in Chest X-ray Images](https://medium.com/@armiro/thoughts-on-covid-19-pneumonia-detection-in-chest-x-ray-images-59f8950e98bb)

## Requirements
- The code has been written in **Python 3.9.x** and requires **TensorFlow v2.6.0**
- Install the dependencies using `pip install -r requirements.txt`

## Data Collection
[![license](https://img.shields.io/badge/license-CC%20BY%204.0-red?style=flat-square)](https://creativecommons.org/licenses/by/4.0/)

Chest x-ray images of patients with (mostly) PCR-positive COVID-19 are collected from different publicly available sources, such as [SIRM](https://www.sirm.org/category/senza-categoria/covid-19/).
Please cite the associated paper if you are using CXR images. If this repo helped you with your research stuff, you can star it.
```
@article{haghanifar2020covidcxnet,
  title={COVID-CXNet: Detecting COVID-19 in Frontal Chest X-ray Images using Deep Learning},
  author={Arman Haghanifar, Mahdiyar Molahasani Majdabadi, Seokbum Ko},
  url={https://github.com/armiro/COVID-CXNet},
  year={2020}
}
```

If you are merging COVID-19 CXR images into your own datasets, please attribute the authors in any publications (DOI: [10.6084/m9.figshare.12580328](https://doi.org/10.6084/m9.figshare.12580328). You may include the version of the dataset found on the figshare webpage for reproducibility.

- View COVID-19 images in the directory: [chest-xray-images/covid19](https://github.com/armiro/COVID-CXNet/tree/master/chest_xray_images/covid19)
- Download COVID-19 images as a single ZIP file: [FigShare](https://figshare.com/articles/COVID-19_Chest_X-Ray_Image_Repository/12580328)
- Download the complete dataset from Kaggle: *coming soon*

There are currently **900** images with different sizes and formats, and the data will not be updated anymore. Normal CXRs are collected from different datasets, without a pediatric image bias. Note that a `-` sign at the end of image name indicates that CXR did not reveal any abnormalities, but the patient had CT/PCR-proven COVID-19 infection (probably patient is in early stages of disease progression). Besides, a `p` letter at the end of image name means that the image is taken from pediatric patient.
