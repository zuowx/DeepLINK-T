# FDP calculation
# Input:
#   S: array of discovered variables
#   beta_true: true coefficient vector
# Output:
#   false discovery proportion
def fdp(S, beta_true):
    return sum(beta_true[S] == 0) / max(1, len(S))


# Power calculation
# Input:
#   S: array of discovered variables
#   beta_true: true coefficient vector
# Output:
#   true discovery proportion
def pow(S, beta_true):
    return sum(beta_true[S] != 0) / sum(beta_true != 0)