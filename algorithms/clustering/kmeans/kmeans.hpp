#pragma once
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>
#include <Eigen/Dense>

namespace tpp {



struct Point {
    double x{0}, y{0};
};

using DataFrame = std::vector<Point>;

double square(double value) {
    return value * value;
}

double squared_l2_distance(Point first, Point second) {
    return square(first.x - second.x) + square(first.y - second.y);
}

DataFrame k_means(const DataFrame& data,
                  size_t k,
                  size_t number_of_iterations) {
    static std::random_device seed;
    static std::mt19937 random_number_generator(seed());
    std::uniform_int_distribution<size_t> indices(0, data.size() - 1);

    // Pick centroids as random points from the dataset.
    DataFrame means(k);
    for (auto& cluster : means) {
        cluster = data[indices(random_number_generator)];
    }

    std::vector<size_t> assignments(data.size());
    for (size_t iteration = 0; iteration < number_of_iterations; ++iteration) {
        // Find assignments.
        for (size_t point = 0; point < data.size(); ++point) {
            double best_distance = std::numeric_limits<double>::max();
            size_t best_cluster = 0;
            for (size_t cluster = 0; cluster < k; ++cluster) {
                const double distance =
                        squared_l2_distance(data[point], means[cluster]);
                if (distance < best_distance) {
                    best_distance = distance;
                    best_cluster = cluster;
                }
            }
            assignments[point] = best_cluster;
        }

        // Sum up and count points for each cluster.
        DataFrame new_means(k);
        std::vector<size_t> counts(k, 0);
        for (size_t point = 0; point < data.size(); ++point) {
            const auto cluster = assignments[point];
            new_means[cluster].x += data[point].x;
            new_means[cluster].y += data[point].y;
            counts[cluster] += 1;
        }

        // Divide sums by counts to get new centroids.
        for (size_t cluster = 0; cluster < k; ++cluster) {
            // Turn 0/0 into 0/1 to avoid zero division.
            const auto count = std::max<size_t>(1, counts[cluster]);
            means[cluster].x = new_means[cluster].x / count;
            means[cluster].y = new_means[cluster].y / count;
        }
    }

    return means;
}


Eigen::ArrayXXd k_means(const Eigen::ArrayXXd &data,
                        uint16_t k,
                        size_t number_of_iterations) {
    static std::random_device seed;
    static std::mt19937 random_number_generator(seed());
    std::uniform_int_distribution<size_t> indices(0, data.size() - 1);

    Eigen::ArrayX2d means(k, 2);
    for (size_t cluster = 0; cluster < k; ++cluster) {
        means.row(cluster) = data(indices(random_number_generator));
    }

    // Because Eigen does not have native tensors, we'll have to split the data by
    // features and replicate it across columns to reproduce the approach of
    // replicating data across the depth dimension k times.
    const Eigen::ArrayXXd data_x = data.col(0).rowwise().replicate(k);
    const Eigen::ArrayXXd data_y = data.col(1).rowwise().replicate(k);

    for (size_t iteration = 0; iteration < number_of_iterations; ++iteration) {
        // This will be optimized nicely by Eigen because it's a large and
        // arithmetic-intense expression tree.
        Eigen::ArrayXXd distances =
                (data_x.rowwise() - means.col(0).transpose()).square() +
                (data_y.rowwise() - means.col(1).transpose()).square();
        // Unfortunately, Eigen has no vectorized way of retrieving the argmin for
        // every row, so we'll have to loop, and iteratively compute the new
        // centroids.
        Eigen::ArrayX2d sum = Eigen::ArrayX2d::Zero(k, 2);
        Eigen::ArrayXd counts = Eigen::ArrayXd::Ones(k);
        for (size_t index = 0; index < data.rows(); ++index) {
            Eigen::ArrayXd::Index argmin;
            distances.row(index).minCoeff(&argmin);
            sum.row(argmin) += data.row(index).array();
            counts(argmin) += 1;
        }
        means = sum.colwise() / counts;
    }

    return means;
}

}
