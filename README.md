# course-recommendation

*this is the final project of the course **2022 ADL Fall**, topic: how how grand challenge*

### Introduction

  In this challenge, there are 2 main domains, seen and unseen. Each domain contains 2 sub-domain which is course-prediction and topic-prediction. Literally, we are going to recommend the courses and the topics of course for the customers based on different level of information.
  
  For both domain we have the customers' personal information such as the gender, occupation, interests and recreation. There are also some information related to the courses such as the course desciptions, instructors, introduction of instructors, topic of the coures and also some brief introduciton of chapters. Theres is purchase history of the seen custormes, howerver, there isn't any purchase history for unseen custormers and that's basically the "unseen" stands for.
  
  In terms of the methodology, in seen domain, I apply the traditional collaborative filtering(CF) method and also conduct few trials applying neural network framework. As for the unseen domain, I take advantage of the BM25 algorithm and neural network as well.
  
  The idea of CF is that it's reasonable to recommend based on a bunch of items purchased by the customers who share similarity. Hence, it's feasible for the seen customers to use CF since we can constuct the customer-course matrix based on the purchase history. I adopt three algorithms: alternating least squares(ALS), bayesian personalized ranking(BPR) and k nearest neighbors. The ALS and BPR are kind of tricky in that the assumption contains that there are latent representaion for both customers and courses. The goal is to find the best latent representation to approximate the customer-course matrix.
  
   With regard to BM25, it's a mature techique in the Field of information retrieval. The concept is given query and list of documents, find the best match based on the frequency.
   
   At last, there are various deep neural network framework, and I first try out multi-layer perceptron(MLP) with and without dropout layer, and also the "two tower" latent model. All the experiments result will be present.
    
## Experiments
  For course-prediction task, I adopt various algorithms. While for topic-prediction task, I wheter take advantage of the course-prediciotion results and based on the course predicted, find their corresponding topics as the predicted topics or directyly apply the algorithm for topic-prediction task. Moreover, the evaluation metric is the mean average precesion at 50
 
### Collaborative Filtering(seen domain)
  After try out the ALS, BPR and KNN, we further do the rearrangement wishing to achieve the better score. The rearrangement idea is why not consider the recommendation result of three based method simultaneously? There are two rearrangement strategy, one is combing two of them and the other is combining all of them. The difference is to choose the base result and one or two of the reference result since we have three kinds of result. Take BPR_KNN for example, the BPR is the based method and the KNN is the reference while in terms of the BPR_ALS_KNN, BPR is still the based method and then reference both ALS and KNN.
#### ALS
  There are three major hyperparamers, alpha, regularizations, factors in ALS and in out case the alpha have minor impact. Hence, I first fix the number of iterations to 30. After find the optimal regularizations and factors, I further inspect the impact of number of iterations.
 
- result
 
| task\hparam | factors | regularizations | iterations | score |
| --- | --- | ---| --- | --- |
| course| 60 | 51.02939 | 312 | 0.07909 |
| topic from course | 370 | 489.7961 | 30 | 0.25506 |
| topic | 0 | 0 | 0 | 0 |

- tune factors and regularizations for course prediction
- tune factors and regularizations for topic prediction from course
- tune factors and regularizations for topic prediction

#### BPR
  There are two major hyperparamers, factors, learning rate in BPR. Hence, I first fix the number of iterations to 100. After find the optimal factors and learning rate, I further inspect the impact of number of iterations.
 
- result
 
| task\hparam | factors | learning rate | iterations | score |
| --- | --- | ---| --- | --- |
| course | 0 | 0 | 0 | 0 |
| topic from course | 0 | 0 | 0 | 0 |
| topic | 0 | 0 | 0 | 0 |

- tune factors and learning for course prediction
- tune factors and learning for topic prediction from course
- tune factors and learning for topic prediction
 
#### KNN
 
- result
 
| task\hparam | score |
| --- | --- |
| course| 0 |
| topic from course | 0 |
| topic | 0 |
 
#### ALS_BPR
 
- result
 
| task\hparam | score |
| --- | --- |
| course| 0 |
| topic from course | 0 |
| topic | 0 |

#### ALS_KNN
 
- result
 
| task\hparam | score |
| --- | --- |
| course| 0 |
| topic from course | 0 |
| topic | 0 |

#### BPR_ALS
 
- result
 
| task\hparam | score |
| --- | --- |
| course| 0 |
| topic from course | 0 |
| topic | 0 |

#### BPR_KNN
 
- result
 
| task\hparam | score |
| --- | --- |
| course| 0 |
| topic from course | 0 |
| topic | 0 |
 
#### KNN_ALS
 
- result
 
| tak\hparam | score |
| --- | --- |
| course| 0 |
| topic from course | 0 |
| topic | 0 |

#### KNN_BPR
 
 - result
 
| task\hparam | score |
| --- | --- |
| course| 0 |
| topic from course | 0 |
| topic | 0 |

#### ALS_BPR_KNN

- result
 
| task\hparam | score |
| --- | --- |
| course| 0 |
| topic from course | 0 |
| topic | 0 |

#### BPR_ALS_KNN

- result
 
| task\hparam | score |
| --- | --- |
| course| 0 |
| topic from course | 0 |
| topic | 0 |

#### KNN_ALS_BPR

- result
 
| task\hparam | score |
| --- | --- |
| course| 0 |
| topic from course | 0 |
| topic | 0 |
 
### BM25(unseen domain)
 
### nerual network
#### seen
- MLP(without dropout)
- MLP(with dropout)
- Two Tower
#### unseen
 
## conclusion
