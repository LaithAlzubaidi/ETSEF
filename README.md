# ETSEF
Code for ETSEF for robust and efficient medical imaging task learning
ETSEF for robust and efficient medical imaging task learning
‘ETSEF’ is an ensemble framework that integrates Transfer Learning (TL) and Self-Supervised Learning (SSL) methods. This strategy leverages multiple pre-trained learning approaches and deep learning architectures to enhance model performance in data-scarce medical environments. The base models are pre-trained using TL and SSL techniques and then ensembled by fusing their features and training higher-level machine learning classifiers.
The provided code demonstrates how to implement ETSEF and includes several examples of Explainable AI (XAI) tools for model explanation and visualisation. The code is written in Jupyter Notebook, with versions available for two popular Python libraries: TensorFlow and PyTorch, offering a convenient reference for readers who wish to implement this framework.
Thanks to the GitHub - google-research/simclr: SimCLRv2 - Big Self-Supervised Models are Strong Semi-Supervised Learners gives a good example of contrastive self-supervised training with SimCLR. For more information and implementation samples of GradCAM in Pytorch, please refer to GitHub - jacobgil/pytorch-grad-cam: Advanced AI Explainability for computer vision. Support for CNNs, Vision Transformers, Classification, Object detection, Segmentation, Image similarity and more. For GradCAM in Tensorflow, please refer to Grad-CAM class activation visualization (keras.io). For more information and samples of SHAP explainer, please visit shap.DeepExplainer — SHAP latest documentation (shap-lrjball.readthedocs.io)
Technical Implementation
Our approach comprises the following steps:
1. Supervised representation learning on a large-scale non-medical dataset.
2. Supervised and self-supervised representation learning on an intermediate source domain dataset.
3. Supervised fine-tuning on the target domain dataset.
4. Feature extraction and fusion from pre-trained base models.
5. Feature selection and ensemble learning with higher-level ML classifiers.
6. Majority voting of ML classifiers.
We provide a full sample of the entire training process. For steps 1-3, refer to the ‘Data Preprocessing’ and ‘Base Model Preparation’ files. For steps 4-6, refer to the ‘Ensemble Learning’ file. The ‘XAI Tool’ file offers samples for utilising Explainable AI (XAI) techniques for visual model analysis.
A detailed research paper summarising our work and design will be provided later. This paper will include detailed descriptions of preprocessing, pretraining, finetuning, and hyperparameters for each task and model.
To address data imbalance and improve data quality, we utilised data augmentation techniques. Four augmentation methods—flip (F), rotation (R), zoom (Z), and random Gaussian blur (G)—are provided. The ‘Data_Split_Augmentation’ file demonstrates splitting the target dataset and applying augmentation methods to the samples.
We began with ResNet, Xception, InceptionResNet, MobileNet, and EfficientNet architectures, initialising them with ImageNet weights from TF and Torch libraries. The base models underwent supervised and self-supervised pretraining using Transfer Learning and Contrastive Self-Supervised Learning methods, with final fine-tuning based on supervised settings.
The ensemble learning step employed five machine learning classifiers—SVM, XGBoost, Random Forest, Naive Bayes, and KNN, to train with extracted and fused features from base models. The ensemble training environment is based on the Scikit-learn library in Python, reloading the pre-trained base models as feature extractors.
The model can be deployed using either TensorFlow or PyTorch, and it can accept base models trained from both libraries. The ‘ETSEF Sample’ file provides an example of utilising base models pre-trained from both libraries as feature extractors and performing the subsequent ensemble learning steps.
Training Data
The initial source domain dataset for all base models is ImageNet. Subsequent intermediate source domain datasets and target datasets are based on colourful and grey-scale medical imaging tasks across different modalities. For example, the target dataset Kvasirv2 contains gastrointestinal (GI) tract images and is suitable for classification and segmentation tasks. It is publicly available on Kaggle.
Our dataset structure follows this format: original_path/dataset_name/train_directory/class_name/sample_name.jpg. The provided code sample for data preprocessing is based on this structure. For other input data directories, please modify the code to fit the structure.

For more details and descriptions of the datasets used in this study and experiment, please refer to our paper.
Installation and Requirements
The provided code is primarily based on TensorFlow 2.10.1 and PyTorch 2.0.1. Note that TensorFlow versions beyond 2.11 do not support GPU running on Windows. For instructions on installing TensorFlow and PyTorch in your Python environment, please refer to the TensorFlow documentation.
Inference: These pre-trained base models can be used directly for effective medical prediction tasks. They can also serve as feature extractors for higher-level ensemble training to achieve better performance.
Finetuning: These models can be fine-tuned end-to-end using medical data from multiple modalities. The double fine-tuning technique employed during pre-training has reduced hardware requirements, but using a GPU or TPU is recommended for faster training.
Model Description
All base models consist of convolutional neural networks pre-trained on a non-medical source domain dataset, an intermediate source domain medical dataset, and fine-tuned on a target medical dataset. These models were trained at a resolution of 224x224 to fit the required input data size. The higher-level machine learning classifiers are based on various statistical methods, including linear, tree-based, boosting, and regression techniques, each with specific hyperparameters to optimise performance. For further details and usage instructions, please refer to our paper and the Scikit-learn library documentation.

Contact For any questions, issues, or requests regarding the code or trained models, please contact l.alzubaidi@qut.edu.au. We welcome feedback and collaboration opportunities.
