# course-recommend
*this is the final project of course **2022 ADL Fall**, topic: how how grand challenge*

## Introduction
    In this challenge, There are 2 main domains, seen and unseen. Each domain contains 2 sub-domain which is course-prediction and topic-prediction. Literally we are going to recommend the course or the topic for the customers based on different level of information.
    For both domain we have the customers' personal information such as the gender, occupation, interests and recreation. There are also some information related to the courses such as the course desciptions, instructors, introduction of instructors, topic of the coures and also some brief introduciton of chapters. Theres is purchase history of the seen custormes, howerver, there isn't any purchase history for unseen custormers and that's basically the "unseen" stands for.
    In terms of the methodology, in seen domain, I apply the traditional collaborative filtering(CF) method and also conduct few trials applying neural network framework. As for the unseen domain, I take advantage of the BM25 algorithm and neural network as well.
    The idea of CF is that it's reasonable to recommend based on a bunch of items purchased by the customers who share similarity. Hence, it's feasible for the seen customers to use CF since we can constuct the customer-course matrix based on the purchase history. I adopt three algorithms: alternating least squares(ALS), bayesian personalized ranking(BPR) and k nearest neighbors. The ALS and BPR are kind of tricky in that the assumption contains that there are latent representaion for both customers and courses. The goal is to find the best latent representation to approximate the customer-course matrix. With regard to BM25, it's a mature techique in the Field of information retrieval. The concept is given query and list of documents, find the best match based on the frequency. At last, there are various deep neural network framework, and I first try out multi-layer perceptron(MLP) with and without dropout layer, and also the "two tower" latent model. All the experiments result will be present.
    
 ## Experiments
 ### CF
 - ALS
 There are three major hyperparamers, alpha, regularizations, factors in ALS and in out case the alpha have minor impact. Hence, I first fix the number of iterations to 30. After find the optimal regularizations and factors, I further inspect the impact of number of iterations.
|domain\param | factors | regularizations | iterations | score|
| --- | --- | ---| --- | ---|
| course| 60 | 51.02939 | 30 | 0.07894 |
|topic from course | 370 | 489.7961 | 30 | 0.25506 |
|topic | 0 | 0 | 0 | 0|

 - BPR
 - KNN
 - ALS_BPR
 - ALS_KNN
 - BPR_ALS
 - BPR_KNN
 - KNN_ALS
 - KNN_BPR
 - ALS_BPR_KNN
 - BPR_ALS_KNN
 - KNN_BPR_ALS
 
 ### BM25
 
 ### nerual network
 #### seen
 - MLP(without dropout)
 - MLP(with dropout)
 - Two Tower
 #### unseen
 
 ## conclusion
