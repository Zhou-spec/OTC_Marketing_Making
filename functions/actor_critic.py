from functions.simulation import *
import torch

def reward(net, S, q, t, buy_orders, sell_orders, T, dt, A, B, gamma, delta, z, Q):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = int(T / dt)
    reward = torch.zeros(N, device = device)
    for i in range(N - 1):
        mean = net.forward(t[i], S[i], q[i])
        hold1 = z * buy_orders[i]
        hold2 = z * sell_orders[i]
        # combine hold1 and hold2 into one vector
        hold = torch.cat((hold1, hold2), 0)
        reward[i] = torch.dot(mean, hold)
        reward[i] = reward[i] + (q[i + 1] * S[i + 1] - q[i] * S[i]) - delta * h(q[i], Q) * (q[i + 1] - q[i])
        reward[i] = reward[i] - (gamma * dt * torch.sum(gamma / (2 * z * B)))
        reward[i] = reward[i] - (gamma * dt * (len(A) * 1.7981798683))

    return reward


def critic_loss(policy_net, value_net, S, q, t, buy_orders, sell_orders, T, dt, A, B, gamma, delta, z, Q):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    r = reward(policy_net, S, q, t, buy_orders, sell_orders, T, dt, A, B, gamma, delta, z, Q)
    loss = torch.zeros(len(r), device = device)
    for i in range(len(r) - 1):
        loss[i] = r[i] + value_net.forward(t[i + 1], S[i + 1], q[i + 1]) - value_net.forward(t[i], S[i], q[i])
    
    scalar_loss = 0.5 * torch.sum(loss[:-1] ** 2)
    return scalar_loss


def probability(policy_net, S, q, t, bid_vectors, ask_vectors, T, dt, B, gamma, z):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = int(T / dt)
    p = torch.zeros(N, device = device)
    for i in range(N - 1):
        mean = policy_net.forward(t[i], S[i], q[i])
        hold1 = bid_vectors[i]
        hold2 = ask_vectors[i]
        hold = torch.cat((hold1, hold2), 0)
        diag = torch.cat((gamma / (2 * z * B), gamma / (2 * z * B)), 0)
        cov = torch.diag(diag)
        p[i] = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov).log_prob(hold)
    return p
    

def policy_loss(policy_net, value_net, S, q, t, buy_orders, sell_orders, bid_vectors, ask_vectors, T, dt, A, B, gamma, delta, z, Q):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = probability(policy_net, S, q, t, bid_vectors, ask_vectors, T, dt, B, gamma, z)
    r = reward(policy_net, S, q, t, buy_orders, sell_orders, T, dt, A, B, gamma, delta, z, Q)
    loss = torch.zeros(len(p), device = device)
    for i in range(len(p) - 1):
        loss[i] = p[i] * (r[i] + value_net.forward(t[i + 1], S[i + 1], q[i + 1]) - value_net.forward(t[i], S[i], q[i]))
    
    scalar_loss = -torch.sum(loss[:-1]) / len(loss[:-1])
    return scalar_loss

