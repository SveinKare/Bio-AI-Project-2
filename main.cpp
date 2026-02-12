#include <iostream>
#include "HomeCare.cpp"

using namespace std;

int main() {
  auto res = HomeCare();
  res.init("./data/train_0.json");
  return 0;
}
