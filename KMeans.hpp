#pragma once
#include <vector>

struct Point {
    double x, y;
    int patientIndex;
};

class KMeans {
private:
    int k;
    std::vector<Point> centroids;

    double distance(const Point& a, const Point& b);
    int nearestCentroid(const Point& p);

public:
    explicit KMeans(int k);
    std::vector<std::vector<int>> fit(const std::vector<Point>& points, int maxIterations = 100);
};
