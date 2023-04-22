# Introduction

Statistics plays a crucial role in machine learning, as many ML techniques are rooted in statistical concepts and methods. Statistical methods can be used to study machine learning in various ways, including the design of learning algorithms, the evaluation of model performance, and the interpretation of results. Here are some ways to use statistical methods to study machine learning:

1. Model selection and evaluation: Statistical methods are essential for comparing different ML models, selecting the best one, and evaluating its performance. Techniques such as cross-validation, bootstrapping, and hypothesis testing can be used to assess the performance of a model and ensure its generalization to new, unseen data.

2. Feature selection: Statistical methods can be employed to identify the most relevant features in a dataset, which can improve model performance and reduce computational complexity. Techniques such as correlation analysis, mutual information, and stepwise regression can help determine which features have the most significant impact on the target variable.

3. Regularization: Regularization techniques, such as L1 (Lasso) and L2 (Ridge) regularization, are used to control the complexity of ML models and prevent overfitting. These techniques are based on statistical concepts, such as penalizing large coefficients in linear regression models, which can lead to more robust and generalizable models.

4. Bayesian methods: Bayesian methods, rooted in probability and statistics, can be used in machine learning for tasks such as model selection, parameter estimation, and prediction. Bayesian techniques provide a principled way to incorporate prior knowledge and uncertainty into ML models and can be applied to various ML algorithms, including linear regression, neural networks, and clustering.

5. Dimensionality reduction: Statistical methods such as principal component analysis (PCA) and factor analysis can be used to reduce the dimensionality of a dataset, which can improve model performance and reduce computational costs.

6. Probability distributions and density estimation: Many ML algorithms rely on probability distributions and density estimation techniques, such as Gaussian mixture models and kernel density estimation, to model the underlying structure of the data.

7. Time series analysis: Statistical methods for time series analysis, including autoregressive and moving average models, can be applied to ML tasks involving time-dependent data, such as predicting stock prices or analyzing sensor data from IoT devices.

8. Experimental design: Statistical methods for experimental design, such as randomization and stratification, can be used to ensure that ML models are trained and tested on representative data samples, which can improve model performance and generalization.

To use statistical methods to study machine learning, it is essential to have a solid understanding of both statistical concepts and ML algorithms. You can start by familiarizing yourself with the relevant statistical methods and tools, such as R, Python's NumPy, SciPy, and scikit-learn libraries, and then apply these methods to various ML tasks and datasets to gain practical experience.

## Bayesian method

The Bayesian method is an approach to statistical inference based on Bayes' theorem, which provides a way to update the probabilities of hypotheses or parameters given observed data. The Bayesian method differs from traditional (frequentist) statistics in that it explicitly incorporates prior knowledge and uncertainty about the unknown quantities, represented as probability distributions, and combines them with the observed data to obtain updated (posterior) probability distributions.

Bayes' theorem is stated as follows:

P(A | B) = (P(B | A) * P(A)) / P(B)

Here, P(A | B) is the posterior probability of event A given that event B has occurred, P(B | A) is the likelihood of event B given that event A has occurred, P(A) is the prior probability of event A, and P(B) is the marginal probability of event B.

In the context of Bayesian statistics:

- A represents the unknown parameters or hypotheses.
- B represents the observed data.
- P(A) is the prior probability distribution, which represents our initial beliefs about the unknown parameters before observing the data.
- P(B | A) is the likelihood function, which describes how likely the observed data are given the parameters.
- P(A | B) is the posterior probability distribution, which represents our updated beliefs about the parameters after incorporating the observed data.

The Bayesian method has several advantages, such as:

1. Incorporating prior knowledge: Bayesian methods allow for the integration of prior knowledge and expert opinions into the analysis, which can be particularly useful when dealing with limited or noisy data.

2. Uncertainty quantification: Bayesian methods provide a natural way to quantify uncertainty by expressing the unknown parameters as probability distributions, making it easier to interpret the results and make decisions under uncertainty.

3. Flexibility: Bayesian methods can be applied to a wide range of statistical models, including linear and nonlinear regression, classification, and hierarchical models.

4. Sequential updating: Bayesian methods can be easily updated with new data, allowing for sequential learning and online adaptation in dynamic environments.

However, Bayesian methods also have some drawbacks, such as increased computational complexity and the need to choose suitable prior distributions, which can sometimes be subjective and lead to biased results if not chosen carefully.

Bayesian methods have been applied to various fields, including machine learning, physics, engineering, economics, and medicine. In machine learning, Bayesian techniques have been used for tasks such as model selection, parameter estimation, and prediction in algorithms such as linear regression, neural networks, and clustering.



**PAC(Probably Approximately Correct) learning model**. Find a good model with a high probability

$$P(|f(x)-y| \leq \epsilon) \geq 1-\delta$$

Science, Technology, Engineering

Science: what, why

Technology: how

Engineering: reduce the cost, improve the effiency, industrialization

What is Machine Learning: Using experience to improve the performance of the system.《Machine Learning》T.Mitchell


Summarize knowledge from experience, experience comes from information, information comes from data, data contains knowledge, knowledge contains wisdom.
从经验里面总结出知识，经验来自信息，信息来自数据，数据里面蕴藏着知识，知识里面还有智慧

Data—information—knowledge—wisdom, with knowledge and wisdom, improve efficiency, create value, and increase productivity
数据——信息——知识——智慧，有了知识和智慧，提高效率，创造价值，提高生产力

**AI & ML: data, algorithm, computational resource**

The optimization and progress of the model requires experience, which is stored and disseminated in the form of data in computers and the Internet

## Data


After humans entered the information age, big data was generated, and the ability and demand for data processing increased, which promoted the development of AI and ML.

Statistics, Math, data analysis, big data, How to gather, store, analyse, transform data increasingly efficiently？

Type of data：historical data(existing, known), future data (predict, judge)）

Different algorithms have different effects on different data. We need to analyze the characteristics of the data and use this as a basis to choose the appropriate algorithm.

Traing dataset and test dataset have the same distribution. That's why we can use the trained model to predict the unseen sample.

ML is not completely random, but based on solid theory. History background: Leslie Valiant(2010 Turing Award)：computational learning theory（1980s）

In his theory, one of the most important theory is **PAC(Probably Approximately Correct) learning model**. Find a good model with a high probability

$$P(|f(x)-y| \leq \epsilon) \geq 1-\delta$$

$x$ is a new sample/data point/observation, $y$ is the true answer/label, if the prediction is very accurate, then $|f(x)-y|\leq \epsilon$, where $\epsilon$ is a very small number. We hope to find such function $\boldsymbol{f}$ with a high probability: $P(|f(x)-y| \leq \epsilon) \geq 1-\delta$, where $\delta$ is a very small number.

**Note: the term probability used above, means that we can not solve the problem with 100% accuracy. Most events in our life are probabilistic, instead of deterministic, how can we solve such kind of problems with more confidence? We need experience. Based on the previous experience or historical data, we can minimize uncertainty to maximun extent. From this point of view, it's like the definition of information from Shannon. Actually, statistics reflect the limitations of human cognition, and it is an effort that human beings are unable to fully grasp the chance, but they still have to understand nature to their best under the constraints.**

### P=NP? 

P：finding the solution to the problem in polynomial time.
在多项式时间里，找到问题的解

NP：Can the solution of a problem given in polynomial time be verified?
在多项式时间里，给出一个问题的解，能否验证？

**For example, Google search：given a key word, can it give the best result in polynomial time complexity? Or even, given a possible result, can it be verified to be optimal in polynomial time? The answer to the question is NO
提供一个key word, 能否在多项式时间复杂度内给出最佳结果？甚至是，给出一个查询结果，能否在多项式时间里验证它就是最优的？No

Actually, a large number of problems in real life are beyond the scope of NP, and even if a solution to a problem is given, we cannot verify in polynomial time whether this is the optimal solution
现实生活中的大量问题，已经超出NP范围，哪怕给出一个问题的解，我们也无法在多项式时间里验证这个是否是最优解

**Wikipedia: P=NP?** 

The class of questions for which some algorithm can provide an answer in polynomial time is "P" or "class P".
The class of questions for which an answer can be verified in polynomial time is NP(nondeterministic polynomial time).

The P vs NP problem is an unsolved problem in computer science that has been the subject of intense research for over 50 years. It was first posed in 1971 by Stephen Cook and Leonid Levin, two computer scientists at the University of Toronto. 

It asks whether every problem whose solution can be quickly verified by a computer can also be quickly solved by a computer. 

P: The informal term quickly, used above, means the existence of an algorithm solving the task that runs in polynomial time, and the time to complete the task varies as a polynomial function instead of exponential time on the size of the input.

NP: For some questions, there is no known way to find an answer quickly, or in polynomial time, but if one is provided with information showing what the answer is, it is possible to verify the answer quickly, or in polynomial time. 

**If the answer is YES(P=NP)**, then many problems that are currently considered intractable could be solved in polynomial time, then it would have profound implications for mathematics cryptography, and artificial intelligence, as many problems that are currently difficult to solve could be solved much more easily. Despite decades of effort, the answer to this question remains unknown.

**If the answer is NO(P $\neq$ NP)**, then there are some problems which cannot be solved efficiently on a classical computer and would require quantum computers or other new technologies to solve them efficiently.

P $\neq$ NP, which is widely believed, it would mean that there are problems in NP that are harder to compute than to verify: they could not be solved in polynomial time, but the answer could be verified in polynomial time.

**Example-1: Sudoku**

A game where the player is given a partially filled-in grid of numbers and attempts to complete the grid following some certain rules. 

Given an incomplete Sudoku grid, of any size, is there at least one legal solution? Any proposed solution is easily verified, and the time to check a solution grows slowly (polynomially) as the grid gets bigger. 

However, all known algorithms for finding solutions take time that grows exponentially as the grid gets bigger. So, Sudoku is in NP (quickly checkable) but does not seem to be in P (quickly solvable). 

Thousands of other problems seem similar, in that they are fast to check but slow to solve. Researchers have shown that many of the problems in NP have the extra property that a fast solution to any one of them could be used to build a quick solution to any other problem in NP, a property called NP-completeness. Decades of searching have not yielded a fast solution to any of these problems, so most scientists suspect that none of these problems can be solved quickly. This, however, has never been proven.

**Example-2: Molecular Hamiltonian and Born-Oppenheimer Approximation**

$\hat{H}|\psi \rangle=E |\psi \rangle$ can't be solved exactly than a hydrogen atom, with a finite amount of computing resources in a finite amount of time. For $N$ charged particels, there are $C_n^2=\dfrac{N(N-1)}{2}$ interacting pairs, this is known for "many-body problem", which is unsolvable for $N>2$. That's why we can only solve it for $H$ atom, containg only one electron and one nucleus, but we can't solve it exactly for any molecular system which has more than two charged particles.

The difficulty of solving such a problem scales as an exponential in the number of particles: $\mathcal{O}(e^N)$, it's not twice as difficult to solve the problem if you go from 2 particles to 4 particles. This is where we have to start approximations in order to be able to get a handle on the molecular hamiltonian and try to get out some approximate solutions which will eventually end up at Hartree-Fock theory.

## Bayesian statistics

**Conditional probability** is a measure of the probability of an event occurring, given that another event (by assumption, presumption, assertion or evidence) has already occurred.

The probability of A under the condition B:

$P(A \mid B)=\dfrac{P(A \cap B)}{P(B)}$

$P(A \cap B)=P(B \mid A) P(A)=P(A \mid B) P(B) \Rightarrow$ **Bayes' theorem**


**Bayes' theorem**

$P(A \mid B)=\dfrac{P(B \mid A) P(A)}{P(B)}$

$P(A \mid B)$ is a conditional probability: the probability of event $A$ occurring given that B is true. It is also called the posterior probability of 
A given B.

If $A$ and $B$ are indenpendent, $P(A \mid B)=P(A), P(B \mid A)=P(B)$ 


### Naive Bayes classifier

Naive Bayes is a simple probabilistic algorithm used for **classification tasks**. It is based on the Bayes theorem, which is used to calculate the probability of a hypothesis based on prior knowledge.

In the context of classification, the Naive Bayes classifier assumes that the presence or absence of a feature in a class is independent of the presence or absence of any other feature in the same class. This is known as the "naive" assumption and is why the algorithm is called "Naive Bayes".

The Naive Bayes classifier works by first learning the probabilities of each class and the probabilities of each feature given each class from a training set of labeled examples. Then, given a new, unlabeled example, the algorithm calculates the probability of the example belonging to each class based on its features using Bayes' theorem. The class with the highest probability is then assigned to the example.

Naive Bayes classifiers are popular in text classification tasks, such as spam filtering, sentiment analysis, and topic classification, as well as in other domains such as image and speech recognition. They are fast, simple, and require relatively small amounts of training data.

### Bayesian network

A Bayesian network, also known as a belief network or a **probabilistic graphical model**, is a probabilistic model that represents a set of variables and their conditional dependencies using a directed acyclic graph (DAG).

In a Bayesian network, each node in the graph represents a random variable, and each edge represents a probabilistic dependency between the variables. The nodes and edges are annotated with conditional probability distributions that specify the probability of a node given its parent nodes in the graph.

Bayesian networks can be used for a variety of tasks, including classification, prediction, and decision making, and are particularly useful when dealing with uncertainty or incomplete information. For example, a Bayesian network can be used to diagnose a patient's illness based on their symptoms, or to predict the likelihood of an event occurring based on a set of observed variables.

One advantage of Bayesian networks is that they allow for efficient inference, which means that they can quickly and accurately calculate the probabilities of certain events or variables given other observed or unobserved variables. This makes them useful in a wide range of applications, including machine learning, data analysis, and decision support systems.

### Bayesian deep learning(BDL)

Bayesian deep learning is a subfield of machine learning that combines the powerful representation learning capabilities of deep neural networks with Bayesian inference methods.

In traditional deep learning, the parameters of the neural network are optimized using deterministic methods, such as gradient descent, to minimize a loss function. In contrast, Bayesian deep learning treats the **parameters as random variables** and applies Bayesian inference to estimate their posterior distributions given the data.

Bayesian deep learning provides a principled way to incorporate prior knowledge into the model and to quantify uncertainty in the predictions. This can lead to more robust and accurate models, especially in situations where data is limited or noisy.

Some common approaches to Bayesian deep learning include Bayesian neural networks, variational autoencoders, and Gaussian processes. These methods often require more computational resources and can be more challenging to implement than traditional deep learning methods. However, they offer the potential for improved performance and better insights into the underlying structure of the data.

In materials research, some parameters in a machine learning model represent the physical properties, and it's likely to find their distribution by modeling the parameters as random variables
