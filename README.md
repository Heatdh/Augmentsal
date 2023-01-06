## Abstract
Data Augmentation is a technique commonly used in Machine Learning to increase the size of a dataset by generating new, synthetic data samples from the existing data. In the context of saliency prediction, Data Augmentation can be used to leverage the performance of Deep Learning models by providing additional data to train on. In this work, we explore different augmentation techniques and investigate how augmentation can affect modern Neural Network Architectures. We were able to achieve an increase in the performance of Neural Networks trained using augmented data and classify the augmentation technique used based on their impact on the performance.


## Contribution 

Our study has demonstrated that the choice of Data Augmentation techniques can significantly impact the performance of a Deep Learning model for saliency prediction. By analyzing the effects of various techniques on network performance, we were able to identify those that were most effective and showed the correlation with the saliency retargeting of the techniques. Our results showed that augmentations with high mean intensity differences from the original, non-augmented maps had a negative impact on network performance, while those with low retargeting improved performance. 

In particular, we found that JPEG Compression was an effective method for improving performance, and that adjusting the quality of this technique had a significant impact on the results. By using low quality JPEG Compression, we were able to achieve the best performance while also significantly reducing the size of the dataset. These results highlight the importance of carefully considering Data Augmentation techniques in the design and training of Deep Learning models.

## Pipeline 
Our pipleline is mainly divided into: 
1. Generate New data using different Augmentation techniques
2. Optimizing the Neural Network used to generate the maps of the augmented data
3. Train new networks with the augmented images as input and the predicted map by the optimized Net as ground truth saliency
This can be seen in the following figures:
![netopt](readme_figures/netoptpipe.png)
![augment%20opt.png](readme_figures/netoptpipe.png)


