#pragma once
#include "HomeCare.hpp"
#include "Similarity.hpp"
#include "individual.h"
#include <vector>

enum class Recombination {
  PMX
};

enum class Mutation {
  SWAP
};

class RuinAndRepair {
private:
    HomeCare& homeCare;
    int popSize;
    int kParents;
    int generations;
    double penalty;
    double crossoverRate;
    double mutationRate;
    double scalingFactor;
    int kElites;
    SimilarityFunc similarityFunc;
    mt19937 rng;

    std::vector<Individual> population;

    Individual randomIndividual();

    vector<Individual> eliteSelection();

    Individual tournamentParentSelection(vector<Individual>::iterator begin, vector<Individual>::iterator end);

    void orderCrossover(vector<Individual>& parents, vector<Individual>& children);

    void mutate(Individual& i);

    void twoOptMutation(Individual& individual);

    void relocateMutation(Individual& individual);

    void exchangeMutation(Individual& individual);

    void generalizedCrowding(vector<Individual>& parents, vector<Individual>& children, vector<Individual>& suvivors);

public:
    explicit RuinAndRepair(
        HomeCare& homeCare, 
        int popSize, 
        int kParents, 
        int generations, 
        double penalty, 
        double crossoverRate, 
        double mutationRate,
        double scalingFactor,
        int kElites,
        SimilarityFunc similarityFunc,
        unsigned int seed
        );
    void printPopulationStats();

    double edgeEntropy();

    double getMinFitness();

    Individual getBestSolution();

    void test();

    void initPopulation();

    void run();
    
    void runGenerations(int generations);

    vector<Individual> getBestIndividuals(int k) const;

    void injectIndividuals(vector<Individual>& immigrants);

    void setPopulation(vector<Individual>& individuals);
};
