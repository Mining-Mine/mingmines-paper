# Mining Mines: A Case Study on Detecting Mining Activities in Eastern Congo Using Satellite Imagery and Scalable Computing Methods

## Introduction

The Eastern Congo is a region rich in minerals such as gold, tantalum, and tin, which are vital for various industries worldwide. However, the mining activities in this area are often unregulated, leading to significant environmental degradation and severe social issues, including human rights abuses and armed conflicts. Monitoring and regulating these activities is crucial for mitigating their negative impacts. Traditional methods of monitoring, such as on-ground inspections, are often impractical due to the region's remote and conflict-prone nature. Therefore, leveraging advanced technologies like satellite imagery and scalable computing methods offers a promising solution to this challenge.

Our project focuses on detecting mining activities in Eastern Congo using satellite imagery data from the Landsat dataset hosted on AWS. The primary research problem is to develop an accurate and efficient method for identifying active mining areas using remote sensing techniques. This task involves processing vast amounts of satellite data, which requires substantial computational resources. By employing scalable computing methods, we aim to create a model that can be used for continuous monitoring and regulation of mining activities, ultimately contributing to the region's sustainable development.

## Justification for Using Scalable Computing Methods

1. Volume of Data
The Landsat dataset comprises decades of satellite imagery, capturing vast geographical areas with high temporal frequency. Analyzing this data manually is not feasible due to its sheer volume. Scalable computing methods, such as cloud computing and distributed processing, enable us to handle and analyze large datasets efficiently.

2. High-Resolution Analysis
Detecting mining activities requires high-resolution analysis to distinguish between different land use types accurately. This precision demands significant computational power, which can be achieved using scalable methods like GPU computing and parallel processing.

3. Model Training and Evaluation
Developing a robust machine learning model necessitates training on large datasets to ensure accuracy and generalizability. This process involves numerous iterations and fine-tuning, which are computationally intensive. Scalable computing allows for faster training and evaluation, reducing the overall project timeline.

4. Real-Time Monitoring
For the model to be effective in real-world applications, it must be capable of real-time monitoring. This requires processing and analyzing new satellite images as they become available. Scalable computing infrastructures can support continuous data ingestion and real-time analysis, making timely interventions possible.

### Methodology

We will use Apache Spark to process and analyze NDVI and other spectral band data from Landsat images. The large-scale data processing capabilities of Spark are ideal for handling the vast amount of data involved in this project. 

We plan to use Spark's machine learning library (MLlib) to train and ensemble multiple models that identify changes in land cover indicative of mining activities, such as vegetation loss and alterations in water bodies.




## Develop Pipelines and Using Different Machine Learning Models for Predictions
main jupyter notebook: []

We use two sample datasets (radius 1km vs 5km) for machine model training. We have three main steps: 

Step 1: Data Processing and Visualization, Step 2: Data Transformations and Feature Engineering and

Step 3: Develop Pipelines and Employee Machine Learnig Models to Predict. Our main results are following:


We use three main models to train and predict data: logit model, random forest, and neutral network. Below are comparison performance of these three model:

### Sample Data Using Radius 1km as Cutoff

| Model               | Logit Model           | Random Forest        | Neural Network        |
|---------------------|-----------------------|-----------------------|-----------------------|
| **Test Accuracy**   | 0.817176979           | 0.873840681           | 0.873840681           |
| **Confusion Matrix**| \[ [3348, 680],<br>   [759, 3084] \]  | \[ [3344, 684],<br>   [309, 3534] \] | \[ [2965, 1063],<br>   [546, 3297] \] |
| **True Positive Rate** | 0.802498048        | 0.837837838           | 0.756192661           |
| **True Negative Rate** | 0.831181728        | 0.91541199            | 0.844488875           |


### Sample Data Using Radius 5km as Cutoff

| Model               | Logit Model           | Random Forests        | Neural Network        |
|---------------------|-----------------------|-----------------------|-----------------------|
| **Test Accuracy**   | 0.773996431           | 0.873840681           | 0.873840681           |
| **Confusion Matrix**| \[ [30073, 9902],<br>   [7703, 30219] \]  | \[ [3344, 684],<br>   [309, 3534] \] | \[ [28570, 11405],<br>   [4130, 33792] \] |
| **True Positive Rate** | 0.796872528        | 0.837837838           | 0.747660243           |
| **True Negative Rate** | 0.752295184        | 0.91541199            | 0.873700306           |

### Conclusion

- **Best Performing Model**: The Random Forests model outperforms the Logit Model and Neural Network in both accuracy and the ability to correctly identify mining and non-mining activities (highest true positive and true negative rates). This makes it the most reliable model for detecting potential mining activities.
- **Impact of Radius Cutoff**: Increasing the radius cutoff from 1km to 5km generally decreases the performance of the Logit Model and Neural Network, while the Random Forests model maintains consistent performance. This suggests that the Random Forests model is more robust to changes in spatial scale.
- **Model Recommendations**: For applications requiring high accuracy and reliability in detecting mining activities, the Random Forests model is recommended. The Logit Model, while simpler, performs adequately but not as well as the Random Forests model. The Neural Network shows potential but may require further tuning to handle larger spatial scales effectively.

In summary, for both radius cutoffs, the Random Forests model demonstrates superior performance, making it the preferred choice for detecting potential mining activities in the given dataset.






## Team Roles 

**Yu Hui**: Processing landsat data as well as related feature using AWS 

**Kevin**: Feature generatig and visualization 


**Charlotte**:  Responsible mainly for training the machine learning model, analyze the performance, and conducting respective visualization.



