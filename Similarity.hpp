#ifndef SIMILARITY_H
#define SIMILARITY_H

#include <vector>

using namespace std;

using SimilarityFunc = double (*)(const vector<int>&, const vector<int>&);

double cosineSimilarity(const vector<int>& a, const vector<int>& b);
double kendallTauSimilarity(const vector<int>& a, const vector<int>& b);
double spearmanFootrule(const vector<int>& a, const vector<int>& b);

#endif
