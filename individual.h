#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

#include <vector>
#include <random>

extern std::mt19937 gen;

class Individual {
private:
    double fitness;
    double penalty;
    double scalingFactor;
    std::vector<int> genes;

public:
    Individual();
    Individual(int n, double f);

    double getFitness() const;
    void setFitness(double f);
    const std::vector<int>& getGenes() const;
    void setGenes(const std::vector<int>& g);
    void setGenes(std::vector<int>&& g);
    double getPenalty() const;
    void setPenalty(double penalty);
    double getScalingFactor() const;
    void setScalingFactor(double scalingFactor);

    void swapPatients(size_t idx1, size_t idx2);
    void mutation(double mutationRate);
    void mutation(double mutationRate, std::mt19937& rng);
    void relocateMutation();
    void relocateMutation(std::mt19937& rng);
    void swapMutation();
    void swapMutation(std::mt19937& rng);
    void reset();

    static Individual crossover(const Individual& parent1, const Individual& parent2);
};

#endif
