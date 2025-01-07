#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>

std::vector<std::vector<double>> instanceNormalize(
    const std::vector<std::vector<double>>& data, double epsilon = 1e-5) {

    size_t num_instances = data.size();
    size_t num_channels = data[0].size();

    std::vector<std::vector<double>> normalized_data = data;

    for (size_t i = 0; i < num_instances; ++i) {
        std::vector<double> instance = data[i];

        // Compute mean and variance for the current instance
        double mean = std::accumulate(instance.begin(), instance.end(), 0.0) / num_channels;
        double variance = 0.0;
        for (double val : instance) {
            variance += (val - mean) * (val - mean);
        }
        variance /= num_channels;

        // Compute instance normalization
        for (size_t j = 0; j < num_channels; ++j) {
            normalized_data[i][j] = (instance[j] - mean) / std::sqrt(variance + epsilon);
        }
    }

    return normalized_data;
}