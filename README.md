# Machine Learning
## Introduction
### What is machine learning?
* Grew out of work in AI
* New capability for computers
* Examples:
1. Database minning
2. Application cat't program by hand
3. Self-customizing programs
4. Understanding human learning(brain, real AI)
### Machine Learning definition
* Arthur Samuel(1959). Machine Learning: Feild of study that gives computers the ability to learn without being explicitly programmed.
* Tom Mitchell(1998). Well-posed Learning Problem: Acomputer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T as measured by P improveswith experience E.
### Machine learing algorithms:
* Supervised learning
* Unsupervised learning
* Others: Reinforcement learning, recommender system
### Supervised learning
* Given the "right answer" for each example in the data
1. Regression problem: predict real-valued output
2. Classfication problem
### Unsupervised learning
* Cocktail party problem
* Clustering algorithm
************************
## Linear regression with one variable
### Model representation

Example: Training set(data set) of housing prices

| Size in feet^2 (x) | Price in 1000's (y) |
|:------------------:|:-------------------:|
|        2104        |         460         |
|        1416        |         232         |
|        1534        |         315         |
|        852         |         178         |
|        ...         |         ...         |

**Notation:**   
*m = Number of training examples*   
*x's = "input" variable / features*   
*y's = "output" variable / "target" variable*   
* Training set ---> Learning algorithm ---> h(hypothesis)
* Size of house(x) ---> h ---> Estimated price(y)
* h is a function: Linear regression with one variable(Univariable linear regression)
* ![alt Linear regression with one variable](res/linearRegressionWithOneVariable.png)
* theta: Parameters
### Cost function
* ![alt Cost function](res/costFunctionOfLinearRegressionWithOneVariable.png)
* Also called square error (cost) function
* Goal: ![alt goalOfCostFunctionOfLinearRegressionWithOneVariable](res/goalOfCostFunctionOfLinearRegressionWithOneVariable.png)
### Gradient descent
* Repeat until convergence: ![alt GradientDescentAlgorithm](res/GradientDescentAlgorithm.png)
* alpha: learning rate
* Gradient descent can converge to a local minimum, even with the learning rate alpha fixed
* As we approach a local minimum, gradient descent will automatically take samller steps. So, no need to decrease alpha over time.
