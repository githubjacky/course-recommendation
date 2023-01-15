# course-recommendation

*this is the final project of the course **2022 ADL Fall**, topic: how how grand challenge*

## Introduction

  In this challenge, there are 2 main domains, seen and unseen. Each domain contains 2 sub-domain which is course-prediction and topic-prediction. Literally, we are going to recommend the courses and the topics of course for the customers based on different level of information.
  
  For both domain we have the customers' personal information such as the gender, occupation, interests and recreation. There are also some information related to the courses such as the course desciptions, the name of instructors, introduction of instructors, topic of the coures and also some brief introduciton of courses' chapter. Theres is seen custormes' purchase history, howerver, there isn't any purchase history of unseen custormers and that's basically the "unseen" stands for.
  
  In terms of the methodology, in seen domain, I apply the traditional collaborative filtering(CF) method and also conduct few trials applying neural network framework. As for the unseen domain, I take advantage of the BM25 algorithm and neural network as well.
 
 ## Methods
  The idea of CF is that it's reasonable to recommend based on a bunch of items purchased by the customers who share similarity with the customer. Hence, it's feasible for the seen customers to use CF since we can constuct the customer-course matrix based on the purchase history. I adopt three algorithms: alternating least squares(ALS), bayesian personalized ranking(BPR) and k nearest neighbors. The ALS and BPR are kind of tricky in that the assumption contains that there are latent representaion for both customers and courses. The goal is to find the best latent representation to approximate the customer-course matrix.
  
   With regard to BM25, it's a mature techique in the Field of information retrieval. The concept is given query and list of documents, find the best match based on the frequency.
   
   At last, there are various deep neural network framework, and I first try out multi-layer perceptron(MLP) with and without dropout layer, and also the "two tower" latent model. All the experiments result will be present.
    
## Experiments
  For course-prediction task, I adopt various algorithms. While for topic-prediction task, I wheter borrow the course-prediciotion results and find their corresponding topics as the predicted topics or directyly apply the algorithm for topic-prediction task. Moreover, the evaluation metric is the mean average precesion at 50.
 
### Collaborative Filtering(seen domain)
  After try out the ALS, BPR and KNN, I further do the rearrangement wishing to improve the order of the recommendation further. The rearrangement idea is why not consider the recommendation result of three based method simultaneously? There are two rearrangement strategy, one is combing two of them and the other is combining all of them. The key difference among the rearrangement stratety is to choose the base result and one or two of the reference results since we have three kinds of CF methods. Take BPR_KNN for example, the BPR is the based method and the KNN is the reference while in terms of the BPR_ALS_KNN, BPR is still the based method and both ALS and KNN are referenced.
 
#### ALS
  There are three major hyperparamers, alpha, regularizations, factors in ALS and in out case the alpha have minor impact. Hence, I first fix the number of iterations to 30. After find the optimal regularizations and factors, I further inspect the impact of number of iterations.
 
| task\hparam | factors | regularization | iterations | score |
| --- | --- | ---| --- | --- |
| course| 60 | 51.02939 | 312 | 0.07909 |
| topic from course | 370 | 489.7961 | 30 | 0.25506 |
| topic | 40 | 51.02939 | 425 | 0.2437019 |

#### BPR
  There are two major hyperparamers, factors, learning rate in BPR. Hence, I first fix the number of iterations to 100. After find the optimal factors and learning rate, I further inspect the impact of number of iterations.
 
| task\hparam | factors | learning rate | iterations | score |
| --- | --- | --- | --- | --- |
| course | 0 | 0 | 0 | 0 |
| topic from course | 0 | 0 | 0 | 0 |
| topic | 0 | 0 | 0 | 0 |
 
#### KNN
 
| task\hparam | k neighbor | score |
| --- | --- | --- |
| course| 0 | 0 |
| topic from course | 0 | 0 |
| topic | 0 | 0 |
 
#### MIX2
 
| task\rearrangement_strategy | ALS_BPR | ALS_KNN | BPR_ALS | BPR_KNN | KNN_ALS | KNN_BPR |
| --- | --- | --- | --- | --- | --- | --- |
| course| 0 | 0 | 0 | 0 | 0 | 0 |
| topic from course | 0 | 0 | 0 | 0 | 0 | 0 |
| topic | 0 | 0 | 0 | 0 | 0 | 0 |

#### MIX3
 
| task\regarrangement_strategy | ALS_BPR_KNN | BPR_ALS_KNN | KNN_ALS_BPR |
| --- | --- | --- | --- |
| course| 0 | 0 | 0 |
| topic from course | 0 | 0 | 0 |
| topic | 0 | 0 | 0 |

### BM25(unseen domain)
 
### nerual network in seen domain
#### MLP(without dropout)
#### MLP(with dropout)
#### Two Tower
### nerual network in seen domain
 
## conclusion
