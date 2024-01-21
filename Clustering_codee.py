# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 23:41:56 2024

@author: Prudhvi vardhan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import scipy.optimize as opt
from sklearn.metrics import silhouette_score  # Import silhouette_score

def cluster_and_polynomial_fit(file_path):
    """
    Perform clustering and polynomial fitting on selected indicators from the given dataset.

    Parameters:
    - file_path (str): The file path of the dataset in Excel format.

    Returns:
    None (displays plots for clustering and polynomial fitting).

    """
    # Load the dataset
    data = pd.read_excel(file_path)

    # Clustering Section

    # List of selected countries
    selected_countries = [
        "Argentina", "Austria", "China", "Germany", "France", 
        "United Kingdom", "Indonesia", "India", "Japan", 
        "New Zealand", "Singapore", "United States"
    ]

    # Selected indicators
    selected_indicators = [
        "Renewable energy consumption (% of total final energy consumption)",
        "CO2 emissions (metric tons per capita)"
    ]

    # Filtering the dataset for the selected countries, indicators, and years 2010-2020
    filtered_data = data[
        (data["Country Name"].isin(selected_countries)) & 
        (data["Indicator Name"].isin(selected_indicators))
    ]

    # Include only the years 2010-2020
    years_range = list(range(2010, 2021))
    filtered_data = filtered_data[["Country Name", "Indicator Name"] + years_range]

    # Melting the dataset to transform it into a long format for easier normalization
    data_melted = pd.melt(filtered_data, id_vars=["Country Name", "Indicator Name"], 
                          var_name="Year", value_name="Value")

    # Splitting the melted data into two separate dataframes for each indicator
    population_data = data_melted[data_melted["Indicator Name"] == "Renewable energy consumption (% of total final energy consumption)"].drop('Indicator Name', axis=1)
    electricity_data = data_melted[data_melted["Indicator Name"] == "CO2 emissions (metric tons per capita)"].drop('Indicator Name', axis=1)

    # Renaming the value columns for clarity
    population_data.rename(columns={"Value": "Renewable energy consumption (% of total final energy consumption)"}, inplace=True)
    electricity_data.rename(columns={"Value": "CO2 emissions (metric tons per capita)"}, inplace=True)

    # Merging the two dataframes on Country Name and Year
    merged_data = pd.merge(population_data, electricity_data, on=["Country Name", "Year"])

    # Applying Min-Max scaling
    min_max_scaler = MinMaxScaler()
    normalized_data = merged_data.copy()
    normalized_data[["Renewable energy consumption (% of total final energy consumption)", "CO2 emissions (metric tons per capita)"]] = min_max_scaler.fit_transform(merged_data[["Renewable energy consumption (% of total final energy consumption)", "CO2 emissions (metric tons per capita)"]])

    # Calculate Silhouette Score for clusters 2 to 10
    for n_clusters in range(2, 11):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        kmeans.fit(normalized_data[['Renewable energy consumption (% of total final energy consumption)', 'CO2 emissions (metric tons per capita)']])
        silhouette_avg = silhouette_score(normalized_data[['Renewable energy consumption (% of total final energy consumption)', 'CO2 emissions (metric tons per capita)']], kmeans.labels_)
        print(f"Silhouette Score for {n_clusters} clusters: {silhouette_avg}")

    # Using K-means clustering algorithm with 5 clusters
    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(normalized_data[['Renewable energy consumption (% of total final energy consumption)', 'CO2 emissions (metric tons per capita)']])

    # Adding a new column 'Cluster' to the original data
    normalized_data['Cluster'] = kmeans.labels_

    # Customizing cluster shapes and colors
    cluster_shapes = ['o', 's', 'D', '^', 'v']  # Example shapes: circle, square, diamond, triangle up, triangle down
    cluster_colors = ['red', 'green', 'blue', 'purple', 'orange']

    # Create a scatter plot for cluster centers
    plt.figure(figsize=(12, 6))

    # Scatter plot for data points
    sns.scatterplot(data=normalized_data, x='Renewable energy consumption (% of total final energy consumption)', y='CO2 emissions (metric tons per capita)', hue='Cluster', palette=cluster_colors, style='Cluster', markers=cluster_shapes, legend='full')

    # Scatter plot for cluster centers with "+" symbols
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='+', s=100, color='black', label='Cluster Centers')

    # Add circles to cover all similar points in each cluster
    for cluster_num in range(5):
        cluster_points = normalized_data[normalized_data['Cluster'] == cluster_num]
        cluster_center = kmeans.cluster_centers_[cluster_num]
        max_distance = np.max(np.linalg.norm(cluster_points[['Renewable energy consumption (% of total final energy consumption)', 'CO2 emissions (metric tons per capita)']].values - cluster_center, axis=1))
        circle = plt.Circle(cluster_center, max_distance, color=cluster_colors[cluster_num], fill=False, linestyle='dotted', alpha=0.5)
        plt.gca().add_artist(circle)

    plt.title('Renewable energy consumption vs CO2 emissions')
    plt.xlabel('Renewable energy consumption')
    plt.ylabel('CO2 emissions (metric tons per capita)')
    plt.legend(title='Cluster', loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.show()

    # Polynomial Fitting Section
    # Filter data for the two indicators
    population_growth_data = data[data['Indicator Name'] == 'Renewable energy consumption (% of total final energy consumption)']
    agricultural_land_data = data[data['Indicator Name'] == 'CO2 emissions (metric tons per capita)']

    # Reshape the data for each indicator
    def reshape_data(data):
        """
    Reshape the input DataFrame for a specific indicator.

    This function takes a DataFrame containing data for a particular indicator and transforms it into a long format
    where countries are represented in rows, years in columns, and the indicator values in the cells.

    Parameters:
    - data (pd.DataFrame): The input DataFrame containing data for a specific indicator.

    Returns:
    pd.DataFrame: Reshaped DataFrame with countries as rows, years as columns, and indicator values as cells.

    """
        data_reshaped = data.drop(['Indicator Name'], axis=1).set_index('Country Name').T
        data_reshaped.index = data_reshaped.index.astype(int)  # Convert year indices to integers
        return data_reshaped

    population_growth_reshaped = reshape_data(population_growth_data)
    agricultural_land_reshaped = reshape_data(agricultural_land_data)

    # Define the polynomial model fitting function
    def polynomial_with_error(x, a, b, c, d, e):
        """
    Evaluate the polynomial model with error at given x values.

    This function represents a 4th-degree polynomial model with error, where the coefficients
    (a, b, c, d, e) are determined through curve fitting. The model is centered around the year 2000
    to improve numerical stability.

    Parameters:
    - x (float or array-like): Input values for which the polynomial model should be evaluated.
    - a, b, c, d, e (float): Coefficients of the polynomial model.

    Returns:
    float or array-like: Evaluated values of the polynomial model at the given x values.
    
    """

        x = x - 2000  # Centering around the year 2000 to improve numerical stability
        return a + b * x + c * x**2 + d * x**3 + e * x**4

    # Partial Derivatives for Polynomial Model
    dfunc = [
        lambda x, a, b, c, d, e: 1,
        lambda x, a, b, c, d, e: (x - 2000),
        lambda x, a, b, c, d, e: (x - 2000)**2,
        lambda x, a, b, c, d, e: (x - 2000)**3,
        lambda x, a, b, c, d, e: (x - 2000)**4
    ]

    # Confidence Interval Calculation Function
    def confidence_interval(x, params, covar, func):
        """
    Calculate the 95% confidence interval for the predictions of a given function.

    Parameters:
    - x (float or array-like): Input values for which the confidence interval should be calculated.
    - params (array-like): Coefficients of the model.
    - covar (array-like): Covariance matrix of the model parameters.
    - func (callable): Function representing the model.

    Returns:
    tuple: Lower and upper bounds of the confidence interval.

    """
        pred = func(x, *params)
        J = np.array([[df(x, *params) for df in dfunc] for x in x])
        pred_se = np.sqrt(np.diag(J @ covar @ J.T))
        ci = 1.96 * pred_se  # 95% confidence interval
        return pred - ci, pred + ci

    # Sample countries to plot
    sample_countries = ['China', 'India', 'United States', 'Brazil']

    # Define custom colors for lines, points, and confidence intervals
    colors = ['red', 'blue', 'orange', 'yellow']

    # Loop over each country for Population Growth
    for i, country in enumerate(sample_countries, 1):
        """
    Loop over each selected country to fit the Polynomial Model, generate predictions, and plot the results.

    Parameters:
    - i (int): Iteration index.
    - country (str): Name of the country.
    """
        # Check if the country data is available
        if country in population_growth_reshaped.columns:
            """
        Check if the data for the selected country is available in the reshaped population growth data.

        Parameters:
        - country (str): Name of the country.
        """
            # Fit the Polynomial Model
            country_data = population_growth_reshaped[country].dropna()
            param_poly, covar_poly = opt.curve_fit(
                polynomial_with_error, 
                country_data.index, 
                country_data.values,
                maxfev=10000  # Increase the number of function evaluations
            )

            # Generate Predictions and Confidence Intervals
            year_range = np.arange(2000, 2026)  # Extended to include 2025
            low_poly, up_poly = confidence_interval(year_range, param_poly, covar_poly, polynomial_with_error)

            # Prediction for 2025
            prediction_2025 = polynomial_with_error(2025, *param_poly)

            # Create a new figure for each country
            plt.figure(figsize=(10, 5))

            # Plotting for Population Growth
            plt.plot(country_data.index, country_data, label=f"Actual Data - {country}", marker="o", color=colors[0])
            plt.plot(year_range, polynomial_with_error(year_range, *param_poly), label=f"Polynomial Fit - {country}", color=colors[2])
            plt.fill_between(year_range, low_poly, up_poly, color=colors[3], alpha=0.5, label=f"95% Confidence Interval - {country}")
            plt.plot(2025, prediction_2025, marker='o', markersize=8, label=f'Prediction for 2025: {prediction_2025:.2f}', color=colors[1])
            plt.title(f"Renewable energy consumption (% of total final energy consumption) - {country}")
            plt.xlabel("Year")
            plt.ylabel("Renewable energy consumption (% of total final energy consumption)")
            plt.legend()
            plt.tight_layout()
            plt.show()

# Example usage
cluster_and_polynomial_fit('world_bank_data_New.xlsx')

     