using namespace std;
#include <vector>
#include <random>

std::mt19937 gen(1234);

class Individual {
    private:
        double fitness;
        std::vector<int> genes;

    public:
        Individual() {};
        Individual(int genes, double fitness) {
            this -> fitness = fitness;
            this -> genes = std::vector<int>(genes);
        }

        double getFitness() {
            return this -> fitness;
        }

        void setFitness(double fitness) {
            this -> fitness = fitness;
        }

        std::vector<int> getGenes() {
            return this -> genes;
        }

        void setGenes(std::vector<int> genes) {
            this -> genes = genes;
        }

        void swapPatients(int idx1, int idx2) {
            if (genes[idx1] >= 0 && genes[idx2] >= 0) {
                std::swap(genes[idx1], genes[idx2]);
            }
        }

        void mutation(double mutationRate) {
            std::uniform_real_distribution<double> mutate(0.0, 1.0);
            for (size_t i = 0; i < this->genes.size(); ++i) {
        if (mutate(gen) < mutationRate) {
                this->swapPatients(i, i + 1);
            }
        }
        }
        void reset() {
            this -> reset();
        }
};