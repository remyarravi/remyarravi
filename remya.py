# %% [markdown]
# 1. Measures of Central Tendency: 
# ● Given the dataset data = [12, 15, 14, 10, 18, 20, 22, 24, 17, 19], calculate the Mean, 
# Median, and Mode using both Excel/Google Sheets/ Python.

# %%
import statistics
data = [12,15,14,10,18,20,22,24,17,19]
mean_value = statistics.mean(data)
median_value = statistics.median(data)
mode_value = statistics.mode(data)
print(f"mean:{mean_value}")
print(f"median:{median_value}")
print(f"mode:{mode_value}")

# %% [markdown]
# 2. Percentiles and Quartiles: 
# ● Compute the 25th percentile (Q1), 50th percentile (Q2), and 75th percentile (Q3) 
# for the dataset using both tools.

# %%
import numpy as np
data = np.array([12,15,14,10,18,20,22,24,17,19])
q1 = np.percentile(data,25)
q2 = np.percentile(data,50)
q3 = np.percentile(data,75)
print(f"q1 (25th percentile): {q1}")
print(f"q2 (50th percentile): {q2}")
print(f"q3  (75th percentile): {q3}")



# %% [markdown]
# 3. Interquartile Range (IQR): 
# ● Find the IQR for the given dataset and explain its significance.

# %%
import numpy as np
data = np.array([12,15,14,10,18,20,22,17,19])
q1 = np.percentile(data,25)
q3 = np.percentile(data,75)
iqr = q3-q1
print(f"1qr: {iqr}")


# %% [markdown]
# 4. Min and Max: 
# ● Identify the minimum and maximum values from the dataset.

# %%
import numpy as np
data = np.array([12,15,14,10,18,20,22,17,19])
min_value = np.min(data)
max_value = np.max(data)
print(f"min: {min_value}")
print(f"max: {max_value}")

# %% [markdown]
# 5. Finding Outliers Using Quartiles: 
# ● Compute the Lower Bound and Upper Bound. 
# ● Identify any outliers in the dataset.

# %%
import numpy as np
data = np.array([12,15,14,10,18,20,22,17,19])
q1 = np.percentile(data,25)
q3 = np.percentile(data,75)
iqr = q3-q1
lower_bound = q1-(1.5*iqr)
upper_bound = q3-(1.5*iqr)
print(f"lower bound:{lower_bound}")
print(f"upper bound:{upper_bound}")

# %% [markdown]
# 6. Measures of Dispersion: 
# ● Compute the Range, Variance, and Standard Deviation using both Excel/Google 
# Sheets/Python.

# %%
import numpy as np
data = np.array([12,15,14,10,18,20,22,17,19])

range_value = np.max(data)- np.min(data)
varience_sample = np.var(data, ddof = 1)
varience_population = np.var(data)
std_dev_sample = np.std(data,ddof = 1)
std_dev_population = np.std(data)
print(f"range: {range_value}")
print(f"sample varience: {varience_sample}")
print(f"population varience:{varience_population}")
print(f"sample std deviation:{std_dev_sample}")
print(f"population std deviation:{std_dev_population}")



                        

# %% [markdown]
# 7. Z-score Standardization: 
# ● Compute the Z-scores for each value in the dataset and explain its significance in 
# data standardization. 

# %%
for i in data:
    z = ((i-mean_value)/std_dev_sample)
    print(f"{z:.2f}")

# %% [markdown]
# 8. Correlation Coefficient: 
# ● Given two datasets x = [10, 20, 30, 40, 50] and y = [5, 10, 15, 20, 25], compute the 
# Pearson correlation coefficient.

# %%
from scipy.stats import pearsonr
x = [10,20,30,40,50]
y = [5,10,15,20,25]
correlation_coefficient,_ = pearsonr(x,y)
print(f"correlation coefficient = {correlation_coefficient : .2f}")


# %% [markdown]
# 9. Scatter Plot Visualization: 
# ● Create a scatter plot using both Excel/Python to visually inspect the correlation 
# between x and y.

# %%
import matplotlib.pyplot as plt
x = [10,20,30,40,50]
y = [5,10,15,20,25]
plt.scatter(x,y)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('scatter plot of x vs. y')
plt.show()

# %% [markdown]
# 10. Box Plot Visualization: 
# ● Create a box plot for the dataset to visualize Q1, Q2, Q3, lower bound, upper 
# bound, and outliers.

# %%
plt.boxplot(data)
plt.title("box plot")
plt.ylabel("values") 
plt.grid(True)

# %%
11. Histogram Analysis: 
● Construct a histogram to show the frequency distribution of the dataset.

# %%
plt.hist(data,bins = 20,color = 'blue')


plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Data Distribution')

# %% [markdown]
# 12.c

# %% [markdown]
# Inferential statistics allows us to make educated guesses about a larger population based on data from a smaller sample. For example, instead of surveying every individual in a country to understand their average income, we can survey a representative group and infer the average income for the entire population. This approach is practical, cost-effective, and efficient, especially when studying large groups.
# 
# Correlation vs. Causation
# 
# It's essential to distinguish between correlation and causation:
# 
# Correlation means that two variables are related—they change together. However, this doesn't confirm that one causes the other
# Causation indicates that one event is the result of the occurrence of the other event; i.e., there is a cause-and-effect relationship between the two events.
# 
# Example: Mortgage Loans and Home Insurance
# 
# When individuals secure a mortgage to purchase a home, lenders typically require them to obtain home insurance. This practice protects the lender's interest in the property. Consequently, there's a correlation between having a mortgage and possessing home insurance. However, the act of borrowing (taking out a mortgage) doesn't directly cause one to buy insurance; rather, the lender's stipulation necessitates the insurance purchase.This scenario illustrates that while loans and insurance policies are correlated, the relationship is not directly causal but is mediated by external requirements.

# %% [markdown]
# 13. Population vs. Sample: 
# ● Why do we need sampling? Provide a real-world example.

# %% [markdown]
# Population vs. Sample: Why We Need Sampling
# Sampling is a fundamental concept in statistics that allows us to make inferences about a larger population without needing to gather data from every single member. Here’s why sampling is essential:
# 
# Why Do We Need Sampling?
# 
# Cost-Effective: Gathering data from an entire population can be expensive and resource-intensive. Sampling reduces costs significantly.
# Time-Saving: Collecting data from the entire population can be time-consuming. Sampling allows for quicker data collection and analysis.
# Feasibility: In many cases, it's simply not feasible to gather data from every member of a population. For example, conducting a survey of every citizen in a large country is impractical.
# Manageable Data: Smaller, manageable datasets can be easier to analyze and interpret without overwhelming computational resources.
# 
# Sampling for Educational Qualification Studies:
# To assess educational qualifications among India's diverse population, researchers often employ stratified sampling methods. This approach involves dividing the population into subgroups (strata) based on characteristics like age, gender, region, and socio-economic status, and then randomly sampling from each subgroup. This ensures that the sample accurately reflects the diversity of the population, leading to more reliable and generalizable findings.
# 
# By utilizing such sampling techniques, policymakers and educators can gain a nuanced understanding of educational attainment across different segments of society, allowing for targeted interventions to address disparities and promote educational advancement.

# %% [markdown]
# 14. Hypothesis Testing Concepts: 
# ● Define Null Hypothesis, Alternate Hypothesis, Significance Level (α), and P-value.

# %% [markdown]
# Scenario: Evaluating the Impact of Opening a New Franchise Location
# 
# Imagine a restaurant chain considering opening a new franchise in a specific city. The management wants to determine if this new location would achieve the company's average monthly sales of $100,000. To make an informed decision, they conduct hypothesis testing based on data from similar franchise locations.
# 
# 1. Null Hypothesis (H₀):
# The null hypothesis asserts that there is no significant difference between the new franchise's sales and the company's average sales. In this context:
# H₀: The new franchise's average monthly sales will be $100,000.
# 
# 2. Alternative Hypothesis (H₁):
# The alternative hypothesis suggests that there is a significant difference:
# H₁: The new franchise's average monthly sales will differ from $100,000.
# This is a two-tailed test, as the concern is with any significant deviation, whether an increase or decrease in sales.
# 
# 3. Significance Level (α):
# The significance level represents the threshold for rejecting the null hypothesis. A common choice is α = 0.05, indicating a 5% risk of concluding that a difference exists when there is none.
# 
# 4. P-value:
# The p-value indicates the probability of obtaining results at least as extreme as the observed ones, assuming the null hypothesis is true. A low p-value (typically less than α) suggests that such extreme results are unlikely under the null hypothesis, leading to its rejection.
# 
# Application:
# The company collects monthly sales data from 30 similar franchise locations. They calculate the sample mean and standard deviation of these sales figures. Using this data, they perform a t-test to compare the sample mean to the hypothesized population mean of $100,000.
# If the p-value ≤ 0.05: There is sufficient evidence to reject the null hypothesis, indicating that the new franchise's sales are likely to differ from $100,000
# If the p-value > 0.05: There isn't enough evidence to reject the null hypothesis, suggesting that the new franchise's sales are not significantly different from $100,000.
# 
# Conclusion:
# 
# By conducting this hypothesis test, the restaurant chain can make a data-driven decision about opening the new franchise location, assessing whether its expected performance aligns with company standards. This example illustrates how hypothesis testing serves as a valuable tool for businesses to evaluate potential outcomes and make informed decisions based on statistical evidence.

# %% [markdown]
# 15. Z-test Calculation: 
# ● Given a sample mean of 25, population mean of 22, population standard deviation 
# of 3, and sample size of 40, compute the Z-test statistic and interpret the results.

# %%
import math
sample_mean = 25
population_mean = 22 
population_stdev = 3
sample_size = 40
z = ((sample_mean - population_mean)/(population_stdev/math.sqrt(40)))
print(f"z-test value = {z:.2f}")


# %% [markdown]
# 16. P-value Computation for Z-test: 
# ● Using a standard normal table, find the p-value corresponding to the Z-test statistic 
# computed in the previous question and determine whether to reject the null 
# hypothesis at α = 0.05.

# %%
from scipy.stats import norm
p_value = 2* (1-norm.cdf(abs(z)))
print(f"pvalue = {p_value:}")

# %% [markdown]
# 17. One Sample T-test: 
# ● Given a sample of data = [45, 50, 55, 60, 62, 48, 52], test whether the mean is 
# significantly different from 50 using a one-sample t-test.

# %%
import numpy as np
from scipy import stats

# Sample data
data = [45, 50, 55, 60, 62, 48, 52]

# Sample mean
sample_mean = np.mean(data)

# Sample standard deviation
sample_std = np.std(data, ddof=1)  # Use ddof=1 for sample standard deviation

# Sample size
n = len(data)

# Population mean
population_mean = 50

# Calculate the t-statistic
t_statistic = (sample_mean - population_mean) / (sample_std / np.sqrt(n))

# Calculate the p-value
p_value = 2 * stats.t.sf(np.abs(t_statistic), df=n-1)

sample_mean, sample_std, t_statistic, p_value

# %%
18. Independent Sample T-test: 
● Two groups of students took a math test. Their scores are: 
○ Group 1: [85, 90, 88, 92, 86] 
○ Group 2: [78, 75, 80, 83, 79] 
● Perform an independent sample t-test to determine if there is a significant 
difference between the means. 

# %%
import numpy as np
from scipy import stats

# Scores for both groups
group1 = [85, 90, 88, 92, 86]
group2 = [78, 75, 80, 83, 79]

# Calculate the t-statistic and p-value
t_statistic, p_value = stats.ttest_ind(group1, group2)

t_statistic, p_value

# %% [markdown]
# 19.19. Critical T-value Lookup: 
# ● Using a t-table, find the critical t-value for α = 0.05 with degrees of freedom 
# appropriate for question 18 and interpret the results.

# %%
import scipy.stats as stats

# Sample sizes
n1 =[85,90,88,92,86]
n2 =[78,75,80,83,79 ]

# Degrees of freedom
df = len(n1) + len(n2) - 2

# Significance level
alpha = 0.05

# Find the critical t-value for a two-tailed test
critical_t_value = stats.t.ppf(1 - alpha / 2, df)

# Display the results
print(f"Degrees of Freedom: {df}")
print(f"Critical t-value for α = {alpha}: {critical_t_value:.3f}")

# %% [markdown]
# 20. Summary and Insights: 
# ● Summarize the key takeaways from the analysis performed above and describe 
# how descriptive and inferential statistics can be used in real-world data analysis.

# %% [markdown]
#  Descriptive statistics summarize and describe data features, such as mean or standard deviation. Inferential statistics use sample data to make predictions or generalizations about a larger population, employing techniques like hypothesis testing and regression analysis. 


