#include <Rcpp.h>
#include <cmath>
#include <vector>
using namespace Rcpp;

// [[Rcpp::export]]
double monteCarloDownInCall(
    double S0, double K, double r, double sigma, double T, double barrier, int nPaths) {
  double dt = T / 100.0; // Time step
  double discountFactor = std::exp(-r * T);

  //  paths and calculate payoff
  std::vector<double> payoffs;
  for (int i = 0; i < nPaths; ++i) {
    double S = S0;
    bool hitBarrier = false;

    for (int t = 0; t < 100; ++t) {
     // normal random variable
     double Z = R::rnorm(0, 1);
      S *= std::exp((r - 0.5 * sigma * sigma) * dt + sigma * std::sqrt(dt) * Z);
      if (S <= barrier) hitBarrier = true;
    }

    if (hitBarrier) {
      payoffs.push_back(std::max(S - K, 0.0));
    }
  }

  // average payoff and discounted present value
  double averagePayoff = std::accumulate(payoffs.begin(), payoffs.end(), 0.0) / payoffs.size();
  return discountFactor * averagePayoff;
}
