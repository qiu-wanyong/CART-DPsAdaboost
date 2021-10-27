# CART-DPsAdaboost Algorithm Framework


### Python-based Demo

* We provide a Python-based Demo of this project for readers' reference only.

Note: The core code for the project is ISD-Lab private.


### CART-DPsAdaboost  [![DOI](doi.org/10.1109/ACCESS.2020.2995058)](https://ieeexplore.ieee.org/abstract/document/9094704)

[Research on an Ensemble Classification Algorithm Based on Differential Privacy](https://ieeexplore.ieee.org/abstract/document/9094704)

差分隐私保护约束下集成分类算法的研究

The model publishing mechanism of ensemble classification algorithm under differential privacy constraint is shown in the figure. This scheme can protect private or sensitive information of data while making the published algorithm model play the maximum availability.

 ![](/result/Graphical_Abstract.png)
 Figure 1 Structure diagram of ensemble classification model under differential privacy protection

### Keyword

Privacy protection, differential privacy, machine learning, AdaBoost.

### Background

In the field of information security, privacy protection based on machine learning is currently a hot topic. 
For classification issues under privacy protection. We proposes an AdaBoost ensemble classification algorithm based on 
differential privacy protection: CART-DPsAdaBoost (CART-Differential Privacy structure of AdaBoost). 

## Requirements

Configuration

| **Quantity**            |    1                                                  |
| ----------------------  | ----------------------------------------------------- |
| **Before Configuration**| 8 core / 16G memory / 256G SSD                        |
| **Runtime Environment** | Version: Pycharm>=2019; Python>=3.7                   |
| **Hashrate**            | Need better                                           |

## Run Test

Test environment: Jupyter Lab

Laplace noise test on Adult：

> python [Adaboost_Adult-Laplace](Adaboost_Adult-Laplace)

Only test on Adult and CensusinCome:

> python [CART-DPsAdaBoost_Adult](CART-DPsAdaBoost_Adult)

> python [CART-DPsAdaBoost_CensusinCome](CART-DPsAdaBoost_CensusinCome)


### DataSet

Adult、Census Income from UCI Machine Learning Repository.

### Experimental

Adult and Census Income show that the model has good classification accuracy while taking into account privacy and usability. 

## In-depth Study of Ensemble Learning under Differential Privacy

In order to improve and further analyze the algorithm, we continue related experiments. 

In further experiments, the impact of the privacy level on the ensemble classification model under different tree
depths is analyzed, and the optimal tree depth value and privacy budget domain are obtained.

Article: [差分隐私保护约束下集成分类算法的研究-信息安全学报](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CJFD&dbname=CJFDLAST2021&filename=XAXB202104007&uniplatform=NZKPT&v=AAQtkV0Zi8V3c0w%25mmd2F3ddKMLSn46bRWjuP%25mmd2B5zm%25mmd2BfAVfqy0Y0HIfd6wyIRxPsxXwUvD)

## Cite As
[1] J. Jia and W. Qiu, "Research on an Ensemble Classification Algorithm Based on Differential Privacy," in IEEE Access, vol. 8, pp. 93499-93513, 2020, doi: 10.1109/ACCESS.2020.2995058.

[2] 贾俊杰,邱万勇,马慧芳.差分隐私保护约束下集成分类算法的研究[J].信息安全学报,2021,6(04):106-118.














