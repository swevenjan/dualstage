# dualstage
A Novel Multi-Modal, Multi-Scale Dual-Stage Glioma Segmentation Network Using MRI Images

	A lightweight, multi-scale feature residual module as the backbone of U-Net, enabling the extraction of diverse and effective features.

	Integration of Receptive Field Blocks into the encoder and decoder parts as enhancement modules to capture multi-scale features and obtain richer contextual information.

	Modification of skip connections to a multi-scale skip connection structure, and employment of an attention mechanism-based decoder to overcome the limitations of receiving same-scale feature information.

Make dataset for training
The first dataset is the Brats2019 open dataset from the MICCAI competition, which includes 259 high-grade glioma (HGG) patients and 76 low-grade glioma (LGG) patients. The second dataset consists of 31 HGG patients and 16 LGG patients from 2020 to 2023.

 https://gitcode.com/Resource-Bundle-Collection/097d3

 This repository provides a download of a resource file, which is a .h5 format file of the Brats-2019 dataset. The Brats-2019 dataset is a dataset widely used for medical image analysis, especially for the segmentation and classification tasks of brain tumors. Dataset Introduction The Brats-2019 dataset contains multimodal brain MRI images, including four modalities: T1, T1ce, T2, and FLAIR. These images are used to train and test brain tumor segmentation algorithms. Each sample in the dataset contains annotation information of the tumor, which can help researchers develop and evaluate tumor segmentation models. File Format This resource file is provided in .h5 format, which is an efficient storage format suitable for storing large-scale medical image data. The .h5 file contains all the necessary image and annotation information and can be directly used for training and testing deep learning models. Usage Download the file: Get the .h5 file of the Brats-2019 dataset through the download link provided in this repository. Load the data: Use Python's h5py library or other tools that support the .h5 format to load the data file. Data preprocessing: Preprocess the data according to specific needs, such as normalization, cropping, etc. Model training: Use the loaded data to train the deep learning model. Notes The dataset is large. Please ensure that there is enough storage space and computing resources when downloading and processing. When using the dataset, please comply with the relevant data use agreement and copyright regulations. References For more information about the Brats-2019 dataset, please refer to related literature and research papers.

 More detailed comparison results will be updated synchronously after the paper is accepted.

 This work was supported in part by the Heilongjiang Provincial Natural Science Foundation of China under Grant LH2022E087, and in part by Heilongjiang Province Key Research and Development Program of China under Grant 2023ZX01A08.
