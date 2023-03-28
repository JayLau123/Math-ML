# Statistics-for-ML
Basic statistics for machine learning

# Introduction

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
