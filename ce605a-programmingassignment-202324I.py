## import modules
import numpy as np
import matplotlib.pyplot as plt
### Part 1 :  Inverse CDF method to drive PMF for discrete variable

# Define the PMF of the discrete distribution
pmf = 

# Cumulative distribution function (CDF)-
cdf = 

# Inverse transform method to sample from the discrete distribution
def inverse_transform_sampling(cdf):
    u = np.random.rand()
    for i in range(len(cdf)):
        if u <= cdf[i]:
            return i

# Number of samples to generate
sample_size = 

# Generate random samples using the inverse transform method
samples = 

# Display the sampled data
print(samples)

# Plot a histogram of the generated samples




### Part 2 :  Inverse CDF method to drive pdf for continous variable




# Define the CDF of the standard exponential distribution
def exponential_cdf(x, lamda):
    return 1 - np.exp(-lamda * x)

# Inverse CDF method to sample from the standard exponential distribution
def inverse_cdf_exponential(lamda, size=1):
    u = np.random.rand(size)
    x = -np.log(1 - u) / lamda
    return x

# Parameters for the standard exponential distribution
lamda = 
sample_size = 

# Generate random samples using the inverse CDF method
samples = 


# Plot a histogram of the generated samples






## Part 3 : Rejection method  Sample of unifrom distribution and target distribution is standard distribution


# Define the target distribution (PDF)
def target_distribution(x):
    return 


# Define the proposal distribution (uniform)
def proposal_distribution(x, lower_bound, upper_bound):
    return 

# Number of samples to generate
sample_size = 

# Define lower bound and upper bound of uniform distribution
lower_bound = 
upper_bound = 

# Rejection method to sample from the target distribution
samples = []
while len(samples) < sample_size:
    x = 
    y = 
    if y <= target_distribution(x):
        samples.append(x)

# Plot a histogram of the generated samples




##Sample for normal distribution and target distribution is standard distribution

# Define the target distribution (standard normal)


# Define the proposal distribution (normal with different parameters)


# Number of samples to generate


# Parameters for the proposal distribution
# Mean of the proposal distribution

# Standard deviation of the proposal distribution

# Rejection method to sample from the target distribution


# Plot a histogram of the generated samples


# Generate values for the x-axis

# Calculate the CDF values using the target distribution
cdf_values = [np.trapz([target_distribution(x) for x in x_values[:i+1]], x_values[:i+1]) for i in range(len(x_values))]

# Plot the CDF


##Bonus


## Solution:
* P(Disease): (prevalence rate) = 0.01 (1%).
* P(No Disease): 1 - P(Disease) = 0.99 (99%).
* P(Positive | Disease): (sensitivity) = 0.95 (95%).
* P(Negative | Disease): 1 - P(Positive | Disease) = 0.05.
* P(Positive | No Disease): (false positive rate) = 0.10 (10%).
* P(Negative | No Disease): (specificity) = 0.90 (90%).
* P(Disease∣Positive)= (0.95*0.01)/(0.95*0.01 + 0.1*0.99)  = 0.08755 = 8.755%
"""



# Define the parameters
sensitivity = 
specificity = 
prevalence = 

# Number of simulations
num_simulations = 

# Simulate the problem and calculate the probability using inverse CDF method
positive_given_disease = 
negative_given_no_disease = 
samples = 

disease_samples =
print(disease_samples)
# Calculate the probability that the person has the disease given a positive result
probability_estimation = 

print("Estimated probability that the person has the disease given a positive test result:", probability_estimation)

