def Viterbi(obs, states, start, trans, emit):
    V = [{}] # DP list: V[t][s] means the max probability of status s in time t.
    path = {} # Path list: path[s] means the optimal path to current status s.

    # Initialization: t = 0
    for s in states:
        V[0][s] = start[s] * emit[s].get(obs[0], 0)
        path[s] = [s]

    # Iteration: t > 0
    for t in range(1, len(obs)):
        V.append({})
        new = {}

        for curr in states:
            (prob, prev) = max(
                (V[t-1][p] * trans[p].get(curr, 0) * emit[curr].get(obs[t], 0), p)
                 for p in states
            )
            V[t][curr] = prob
            new[curr] = path[prev] + [curr]
        path = new

    # Termination: Searching for the final status of the max probability

    n = len(obs) - 1
    (final_p, final_s) = max((V[n][s], s) for s in states)

    return final_p, path[final_s]


# Example
if __name__ == "__main__":
    states = ['Rainy', 'Sunny']
    observations = ['walk', 'shop', 'clean']
    start_p = {'Rainy': 0.6, 'Sunny': 0.4}
    trans_p = {
        'Rainy': {'Rainy': 0.7, 'Sunny': 0.3},
        'Sunny': {'Rainy': 0.4, 'Sunny': 0.6}
    }
    emit_p = {
        'Rainy': {'walk': 0.1, 'shop': 0.4, 'clean': 0.5},
        'Sunny': {'walk': 0.6, 'shop': 0.3, 'clean': 0.1}
    }

    prob, path = Viterbi(observations, states, start_p, trans_p, emit_p)
    print("Most possible status sequence: ", path)
    print("Probability: ", prob)