#include "KMeans.hpp"
#include <cmath>
#include <limits>
#include <random>
#include <algorithm>

KMeans::KMeans(int k) : k(k) {}

double KMeans::distance(const Point& a, const Point& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

int KMeans::nearestCentroid(const Point& p) {
    int best = 0;
    double bestDist = std::numeric_limits<double>::max();
    for (int i = 0; i < k; i++) {
        double d = distance(p, centroids[i]);
        if (d < bestDist) {
            bestDist = d;
            best = i;
        }
    }
    return best;
}

std::vector<std::vector<int>> KMeans::fit(const std::vector<Point>& points, int maxIterations) {
    std::mt19937 rng(std::random_device{}());
    std::vector<int> centroidIndices;

    std::uniform_int_distribution<int> dist(0, (int)points.size() - 1);
    centroidIndices.push_back(dist(rng));

    for (int i = 1; i < k; i++) {
        std::vector<double> distances;
        for (const auto& p : points) {
            double minDist = std::numeric_limits<double>::max();
            for (int idx : centroidIndices)
                minDist = std::min(minDist, distance(p, points[idx]));
            distances.push_back(minDist * minDist);
        }
        std::discrete_distribution<int> weightedDist(distances.begin(), distances.end());
        centroidIndices.push_back(weightedDist(rng));
    }

    centroids.clear();
    for (int idx : centroidIndices)
        centroids.push_back(points[idx]);

    std::vector<int> assignments(points.size(), -1);

    for (int iter = 0; iter < maxIterations; iter++) {
        bool changed = false;
        for (int i = 0; i < (int)points.size(); i++) {
            int nearest = nearestCentroid(points[i]);
            if (nearest != assignments[i]) {
                assignments[i] = nearest;
                changed = true;
            }
        }
        if (!changed) break;

        std::vector<double> sumX(k, 0), sumY(k, 0);
        std::vector<int> count(k, 0);
        for (int i = 0; i < (int)points.size(); i++) {
            int c = assignments[i];
            sumX[c] += points[i].x;
            sumY[c] += points[i].y;
            count[c]++;
        }
        for (int i = 0; i < k; i++) {
            if (count[i] > 0) {
                centroids[i].x = sumX[i] / count[i];
                centroids[i].y = sumY[i] / count[i];
            }
        }
    }

    std::vector<std::vector<int>> clusters(k);
    for (int i = 0; i < (int)points.size(); i++)
        clusters[assignments[i]].push_back(points[i].patientIndex);

    return clusters;
}
