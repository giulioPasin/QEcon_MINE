#' MC Pricing of Down-and-In Call Option
#'
#' @param S0 Initial stock price
#' @param K Strike price
#' @param r Risk-free rate
#' @param sigma Annualized volatility
#' @param T Time to maturity (in years)
#' @param barrier Barrier level
#' @param nPaths Number of Monte Carlo paths
#' @return Theoretical price of the down-and-in call option
#' @export
monteCarloDownInCall <- function(S0, K, r, sigma, T, barrier, nPaths = 10000) {
  .Call("_optionPricer_monteCarloDownInCall", PACKAGE = "optionPricer",
        S0, K, r, sigma, T, barrier, nPaths)
}
