from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

prompt_notes =PromptTemplate(
    template="Generate a Short Notes from the following text \n {text}",
    input_variables=["text"]
)

prompt_qa =PromptTemplate(
    template="Generate a 5 question and answere from the following text \n {text}",
    input_variables=["text"]
)

final_prompt = PromptTemplate(
    
    template="Merge the following short notes \n {notes} and following question and answere \n {quiz} into a single document",
    input_variables=["notes","quiz"]
)

model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.4,
    api_key=api_key
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    
    'notes': prompt_notes|model|parser,
    'quiz': prompt_qa|model|parser
}
)


merged_chain = parallel_chain|final_prompt|model|parser

text= """
What is linear regression?
Linear regression is a data analysis technique that predicts the value of unknown data by using another related and known data value. It mathematically models the unknown or dependent variable and the known or independent variable as a linear equation. For instance, suppose that you have data about your expenses and income for last year. Linear regression techniques analyze this data and determine that your expenses are half your income. They then calculate an unknown future expense by halving a future known income.

Why is linear regression important?
Linear regression models are relatively simple and provide an easy-to-interpret mathematical formula to generate predictions. Linear regression is an established statistical technique and applies easily to software and computing. Businesses use it to reliably and predictably convert raw data into business intelligence and actionable insights. Scientists in many fields, including biology and the behavioral, environmental, and social sciences, use linear regression to conduct preliminary data analysis and predict future trends. Many data science methods, such as machine learning and artificial intelligence, use linear regression to solve complex problems.

How does linear regression work?
At its core, a simple linear regression technique attempts to plot a line graph between two data variables, x and y. As the independent variable, x is plotted along the horizontal axis. Independent variables are also called explanatory variables or predictor variables. The dependent variable, y, is plotted on the vertical axis. You can also refer to y values as response variables or predicted variables.

Steps in linear regression
For this overview, consider the simplest form of the line graph equation between y and x; y=c*x+m, where c and m are constant for all possible values of x and y. So, for example, suppose that the input dataset for (x,y) was (1,5), (2,8), and (3,11). To identify the linear regression method, you would take the following steps:

Plot a straight line, and measure the correlation between 1 and 5.
Keep changing the direction of the straight line for new values (2,8) and (3,11) until all values fit.
Identify the linear regression equation as y=3*x+2.
Extrapolate or predict that y is 14 when x is
What is linear regression in machine learning?
In machine learning, computer programs called algorithms analyze large datasets and work backward from that data to calculate the linear regression equation. Data scientists first train the algorithm on known or labeled datasets and then use the algorithm to predict unknown values. Real-life data is more complicated than the previous example. That is why linear regression analysis must mathematically modify or transform the data values to meet the following four assumptions.

Linear relationship
A linear relationship must exist between the independent and dependent variables. To determine this relationship, data scientists create a scatter plot—a random collection of x and y values—to see whether they fall along a straight line. If not, you can apply nonlinear functions such as square root or log to mathematically create the linear relationship between the two variables.

Residual independence
Data scientists use residuals to measure prediction accuracy. A residual is the difference between the observed data and the predicted value. Residuals must not have an identifiable pattern between them. For example, you don't want the residuals to grow larger with time. You can use different mathematical tests, like the Durbin-Watson test, to determine residual independence. You can use dummy data to replace any data variation, such as seasonal data.

Normality
Graphing techniques like Q-Q plots determine whether the residuals are normally distributed. The residuals should fall along a diagonal line in the center of the graph. If the residuals are not normalized, you can test the data for random outliers or values that are not typical. Removing the outliers or performing nonlinear transformations can fix the issue.

Homoscedasticity
Homoscedasticity assumes that residuals have a constant variance or standard deviation from the mean for every value of x. If not, the results of the analysis might not be accurate. If this assumption is not met, you might have to change the dependent variable. Because variance occurs naturally in large datasets, it makes sense to change the scale of the dependent variable. For example, instead of using the population size to predict the number of fire stations in a city, might use population size to predict the number of fire stations per person.

What are the types of linear regression?
Some types of regression analysis are more suited to handle complex datasets than others. The following are some examples.

Simple linear regression
Simple linear regression is defined by the linear function:

Y= β0*X + β1 + ε 

β0 and β1 are two unknown constants representing the regression slope, whereas ε (epsilon) is the error term.

You can use simple linear regression to model the relationship between two variables, such as these:

Rainfall and crop yield
Age and height in children
Temperature and expansion of the metal mercury in a thermometer
Multiple linear regression
In multiple linear regression analysis, the dataset contains one dependent variable and multiple independent variables. The linear regression line function changes to include more factors as follows:

Y= β0*X0 + β1X1 + β2X2+…… βnXn+ ε 

As the number of predictor variables increases, the β constants also increase correspondingly.

 Multiple linear regression models multiple variables and their impact on an outcome:

Rainfall, temperature, and fertilizer use on crop yield
Diet and exercise on heart disease
Wage growth and inflation on home loan rates
Logistic regression
Data scientists use logistic regression to measure the probability of an event occurring. The prediction is a value between 0 and 1, where 0 indicates an event that is unlikely to happen, and 1 indicates a maximum likelihood that it will happen. Logistic equations use logarithmic functions to compute the regression line.

These are some examples:

The probability of a win or loss in a sporting match
The probability of passing or failing a test 
The probability of an image being a fruit or an animal
How can AWS help you solve linear regression problems?
Amazon SageMaker is a fully managed service that can help you quickly prepare, build, train, and deploy high-quality machine learning (ML) models. Amazon SageMaker Canvas is a generic automatic ML solution for classification and regression problems, such as fraud detection, churn analysis, and targeted marketing. 

Amazon Redshift, a fast, widely used cloud data warehouse, natively integrates with Amazon SageMaker for ML. With Amazon Redshift ML, you can use simple SQL statements to create and train ML models from your data in Amazon Redshift. You can then use these models to solve all types of linear regression problems.
"""

result = merged_chain.invoke({"text":text})

print(result)
