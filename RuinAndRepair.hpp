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
    double epsilon;
    int kParents;
    int generations;
    double penalty;
    double crossoverRate;
    double mutationRate;
    double scalingFactor;
    int kElites;
    SimilarityFunc similarityFunc;

    std::vector<Individual> population;

    void printPopulationStats();

    void initPopulation();

    Individual randomIndividual(vector<vector<int>>& clusters);

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
        double epsilon, 
        int kParents, 
        int generations, 
        double penalty, 
        double crossoverRate, 
        double mutationRate,
        double scalingFactor,
        int kElites,
        SimilarityFunc similarityFunc
        );

    double getMinFitness();

    Individual getBestSolution();

    void test();

    void run();
};
