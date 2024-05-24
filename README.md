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




## Team Roles 

**Yu Hui**: Processing landsat data as well as related feature using AWS 

**Charlotte**:  Data processinng and engineering a scalable data collection pipeline.

**Kevin**: Responsible mainly for training the machine learning model, analyze the performance, and conducting respective visualization.


