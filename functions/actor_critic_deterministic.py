from functions.simulation import *
import torch

def reward_deterministic(net, S, q, t, buy_orders, sell_orders, T, dt, A, B, gamma, delta, z, Q):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = int(T / dt)
    reward = torch.zeros(N, device = device)
    for i in range(N - 1):
        b = net.forward(t[i], S[i], q[i])
        hold1 = z * buy_orders[i]
        hold2 = z * sell_orders[i]
        # combine hold1 and hold2 into one vector
        hold = torch.cat((hold1, hold2), 0)
        reward[i] = torch.dot(b, hold)
        reward[i] = reward[i] + (q[i + 1] * S[i + 1] - q[i] * S[i]) - delta * h(q[i], Q) * (q[i + 1] - q[i])

    return reward


def critic_loss_deterministic(policy_net, value_net, S, q, t, buy_orders, sell_orders, T, dt, A, B, gamma, delta, z, Q):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    r = reward_deterministic(policy_net, S, q, t, buy_orders, sell_orders, T, dt, A, B, gamma, delta, z, Q)
    loss = torch.zeros(len(r), device = device)
    for i in range(len(r) - 1):
        loss[i] = r[i] + value_net.forward(t[i + 1], S[i + 1], q[i + 1]) - value_net.forward(t[i], S[i], q[i])
    
    scalar_loss = 0.5 * torch.sum(loss[:-1] ** 2)
    return scalar_loss


def policy_loss_deterministic(policy_net, value_net, S, q, t, buy_orders, sell_orders, T, dt, A, B, gamma, delta, z, Q):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    r = reward_deterministic(policy_net, S, q, t, buy_orders, sell_orders, T, dt, A, B, gamma, delta, z, Q)
    loss = torch.zeros(len(r), device = device)
    for i in range(len(r) - 1):
        loss[i] =  r[i] + value_net.forward(t[i + 1], S[i + 1], q[i + 1]) - value_net.forward(t[i], S[i], q[i])
    
    scalar_loss = -torch.sum(loss[:-1]) / len(loss[:-1])
    return scalar_loss


