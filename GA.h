#ifndef GA_H
#define GA_H

#include "individual.h"
#include <vector>
#include <random>

class HomeCare;

struct FitnessResult {
    double rawTravelTime;
    double penalty;
    bool isValid;
};

class GA {
private:
    HomeCare* problem;
    std::vector<Individual> population;
    int populationSize;
    double mutationRate;
    int eliteCount;
    int tournamentSize;

    int tournamentSelect() const;
    int tournamentSelect(std::mt19937& rng) const;
    int similarity(const Individual& a, const Individual& b) const;
    FitnessResult evaluateDetailed(const Individual& ind) const;
    double evaluateRoutes(const std::vector<std::vector<int>>& routes) const;
    void applyLocalSearch(Individual& ind) const;
    void apply2Opt(Individual& ind) const;
    void applyOrOpt(Individual& ind) const;
    void applyRelocate(Individual& ind) const;
    Individual crossoverFlat(const Individual& parent1, const Individual& parent2) const;
    Individual createRandomIndividual() const;
    Individual createRandomIndividual(std::mt19937& rng) const;
    Individual createGreedyIndividual() const;
    Individual createGreedyIndividual(std::mt19937& rng) const;
        void restartPartOfPopulation();

    double lastBestFitness;
    int generationsWithoutImprovement;
    int stagnationThreshold;
    double restartFraction;
    int generationCount;
    int islandId;
    double penaltyMultiplier;
    double greedyFraction;

public:
    GA(HomeCare* hc, int popSize = 100, double mutRate = 0.1,
      int elites = 2, int tournSize = 5,
      int stagnationGen = 50, double restartFrac = 0.5,
      int island = 0, double greedyFrac = 0.25);

    void initialize();
    void runGeneration();
    double evaluateFitness(const Individual& ind) const;
    void ruinAndRecreate(Individual& ind, double destructionRate, std::mt19937& rng) const;
    FitnessResult evaluateFitnessDetailed(const Individual& ind) const;
    void evaluatePopulation();

    const std::vector<Individual>& getPopulation() const;
    std::vector<Individual>& getPopulation();
    Individual getBest() const;

    int getPopulationSize() const;
    double getMutationRate() const;
    void setMutationRate(double r);
    int getEliteCount() const;
    void setEliteCount(int e);
    double getPenaltyMultiplier() const;
};

#endif
