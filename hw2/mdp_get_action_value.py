def get_action_value(mdp, state_values, state, action, gamma):
    states_to_get = mdp.get_next_states(state, action)
    Q = 0
    for s in states_to_get:
        Q += float(states_to_get[s]) * (mdp.get_reward(state,action,s)+ gamma * state_values[s])
    return Q
