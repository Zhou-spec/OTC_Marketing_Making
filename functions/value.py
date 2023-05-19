from functions.simulation import *

def value_function_loss(net, S, q, t, dt):
    N = len(q)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss = torch.zeros(N, device = device)
    for i in range(N - 1):
        loss[i] = (net.forward(t[i + 1], S[i + 1], q[i + 1]) - net.forward(t[i], S[i], q[i])) / dt
    return loss

def inventory_loss(net, S, q, t, dt, buy_orders, sell_orders, z, delta, Q, A, B):
    N = len(q)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss = torch.zeros(N, device = device)
    for i in range(N):
        for k in range(len(A)):
            loss[i] = loss[i] + (buy_orders[i][k] - sell_orders[i][k]) * (z[k] * S[i] - delta * h(q[i], Q))
            loss[i] = loss[i] +  (A[k] / (2 * B[k])- (net.forward(t[i], S[i], q[i] + z[k]) - net.forward(t[i], S[i], q[i]) + z[k] * (S[i] + delta * h(q[i], Q))) / (2 * z[k])) * buy_orders[i][k]
            loss[i] = loss[i] + (A[k] / (2 * B[k]) - (net.forward(t[i], S[i], q[i] - z[k]) - net.forward(t[i], S[i], q[i]) - z[k] * (S[i] - delta * h(q[i], Q))) / (2 * z[k])) * sell_orders[i][k]
    return loss

def total_loss(net, S, q, t, dt, buy_orders, sell_orders, z, delta, Q, A, B, gamma):
    N = len(S)
    K = len(A)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss = torch.zeros(N, device = device)
    loss1 = value_function_loss(net, S, q, t, dt)
    loss2 = inventory_loss(net, S, q, t, dt, buy_orders, sell_orders, z, delta, Q, A, B)
    loss = loss1 + loss2 - gamma * ((K * 1.7981798683) + torch.sum(gamma / (2 * z * B)))
    
    scalar_loss = 0.5 * torch.sum(loss[:-1] ** 2) * dt
    return scalar_loss