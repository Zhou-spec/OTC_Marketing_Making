import torch 

def Stock_Prices_Simulation(T, dt, sigma, S0):
    # T: the total time needed
    # dt: the time interval
    # sigma: the volatility
    # S0: the initial stock price
    # the output is the simulated stock prices in torch tensor
    
    # let S be on the same device as S0
    N = int(T / dt)
    device = S0.device
    S = torch.zeros(N, device = device)
    S[0] = S0
    for i in range(1, N):
        S[i] = S[i - 1] + sigma * (torch.sqrt(torch.tensor([dt])) * torch.randn(1)).to(device)

    return S


def h(q, Q):
    # q: the current inventory
    # Q: the maximum inventory
    # this function is used to decide weather to externalize the inventory or not

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if q < Q and q > - 1 * Q:
        return torch.tensor([0], device = device)
    else:
        return torch.tensor([1], device = device)
    

def Market_Order_Generator(bid_vector, ask_vector, A, B, dt):
    # bid_vector: the bid price vector, torch tensor of size N
    # ask_vector: the ask price vector, torch tensor of size N
    # dt: the time interval
    # In this project, we assume that MO intensity lambda = A - B * epsilon

    N = len(A)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    buy_orders = torch.zeros(N, device = device)
    sell_orders = torch.zeros(N, device = device)
    
    bid_intensity = A - B * bid_vector
    ask_intensity = A - B * ask_vector
    buy_orders = torch.distributions.poisson.Poisson(bid_intensity * dt).sample()
    sell_orders = torch.distributions.poisson.Poisson(ask_intensity * dt).sample()

    return buy_orders, sell_orders


def Gaussian_Policy(t, S, q, net, A, B, Q, z, delta, gamma):
    # t: the current time
    # S: the current stock price
    # q: the current inventory
    # net: the neural network
    # This function is used to generate the bid and ask price vectors under Gaussian policy

    N = len(A)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    bid_vector = torch.zeros(N, device = device)
    ask_vector = torch.zeros(N, device = device)

    for i in range(N):
        bid_mean = (A[i] / (2 * B[i])) - (net.forward(t, S, q + z[i]) - net.forward(t, S, q) + z[i] * (S + delta * h(q, Q))) / (2 * z[i])
        ask_mean = (A[i] / (2 * B[i])) - (net.forward(t, S, q - z[i]) - net.forward(t, S, q) - z[i] * (S - delta * h(q, Q))) / (2 * z[i])
        variance = gamma / (2 * z[i] * B[i])
        std = torch.sqrt(variance)
        bid_vector[i] = torch.normal(bid_mean, std)
        ask_vector[i] = torch.normal(ask_mean, std)

    return bid_vector, ask_vector


def DNN_Policy(t, S, q, net):
    # t: the current time
    # S: the current stock price
    # q: the current inventory
    # net: the neural network that represent directly the policy
    # This function is used to generate the bid and ask price vectors under DNN policy (the most general one)
    
    vector = net.forward(t, S, q)
    bid_vector = vector[:int(len(vector) / 2)]
    ask_vector = vector[int(len(vector) / 2):]

    return bid_vector, ask_vector

def DNN_Policy_Stochastic(t, S, q, net, gamma, z, B):
    mean = net.forward(t, S, q)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bid_mean = mean[:int(len(mean) / 2)]
    ask_mean = mean[int(len(mean) / 2):]
    covariance_matrix = torch.diag(gamma / (2 * z * B))
    covariance_matrix = covariance_matrix.to(device)
    bid_vector = torch.distributions.multivariate_normal.MultivariateNormal(bid_mean, covariance_matrix).sample()
    ask_vector = torch.distributions.multivariate_normal.MultivariateNormal(ask_mean, covariance_matrix).sample()

    return bid_vector, ask_vector



def Train_Data_Simulation(T, dt, sigma, S0, A, B, Q, z, delta, gamma, net):
    # T: the total time needed
    # dt: the time interval
    # sigma: the volatility
    # S0: the initial stock price
    # net: the neural network that represent the value function
    # this function return the simulated stock prices, buy orders, sell orders, inventory and time for N time steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = int(T / dt)
    S = Stock_Prices_Simulation(T, dt, sigma, S0)
    buy_orders = torch.zeros(N, len(A))
    sell_orders = torch.zeros(N, len(A))
    buy_orders = buy_orders.to(device)
    sell_orders = sell_orders.to(device)
    q = torch.zeros(N, device = device)
    t = torch.zeros(N, device = device)
    bid_vectors = torch.zeros(N, len(A))
    ask_vectors = torch.zeros(N, len(A))
    bid_vectors = bid_vectors.to(device)
    ask_vectors = ask_vectors.to(device)
    for i in range(N - 1):
        bid_vector, ask_vector = Gaussian_Policy(t[i], S[i], q[i], net, A, B, Q, z, delta, gamma)
        bid_vectors[i] = bid_vector
        ask_vectors[i] = ask_vector
        buy_orders[i], sell_orders[i] = Market_Order_Generator(bid_vector, ask_vector, A, B, dt)
        for j in range(len(A)):
            q[i + 1] += (buy_orders[i][j] - sell_orders[i][j]) * z[j]
        q[i + 1] += q[i]
        t[i + 1] = t[i] + dt
        
    return S, buy_orders, sell_orders, q, t, bid_vectors, ask_vectors


def Test_Data_Simulation_NN(T, dt, sigma, S0, net, A, B):
    # T: the total time needed
    # dt: the time interval
    # sigma: the volatility
    # S0: the initial stock price
    # net: the neural network that represent the value function
    # this function return the simulated stock prices, buy orders, sell orders, inventory and time for N time steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = int(T / dt)
    S = Stock_Prices_Simulation(T, dt, sigma, S0)
    buy_orders = torch.zeros(N, len(A))
    sell_orders = torch.zeros(N, len(A))
    buy_orders = buy_orders.to(device)
    sell_orders = sell_orders.to(device)
    q = torch.zeros(N, device = device)
    t = torch.zeros(N, device = device)
    bid_vectors = torch.zeros(N, len(A))
    ask_vectors = torch.zeros(N, len(A))
    bid_vectors = bid_vectors.to(device)
    ask_vectors = ask_vectors.to(device)
    for i in range(N - 1):
        bid_vector, ask_vector = DNN_Policy(t[i], S[i], q[i], net)
        bid_vectors[i] = bid_vector
        ask_vectors[i] = ask_vector
        buy_orders[i], sell_orders[i] = Market_Order_Generator(bid_vector, ask_vector, A, B, dt)
        for j in range(len(A)):
            q[i + 1] += (buy_orders[i][j] - sell_orders[i][j]) * z[j]
        q[i + 1] += q[i]
        t[i + 1] = t[i] + dt
        
    return S, buy_orders, sell_orders, q, t, bid_vectors, ask_vectors


def Test_Data_Simulation_Stochastic(T, dt, sigma, S0, net, A, B, gamma, z):
    # T: the total time needed
    # dt: the time interval
    # sigma: the volatility
    # S0: the initial stock price
    # net: the neural network that represent the value function
    # this function return the simulated stock prices, buy orders, sell orders, inventory and time for N time steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = int(T / dt)
    S = Stock_Prices_Simulation(T, dt, sigma, S0)
    buy_orders = torch.zeros(N, len(A))
    sell_orders = torch.zeros(N, len(A))
    buy_orders = buy_orders.to(device)
    sell_orders = sell_orders.to(device)
    q = torch.zeros(N, device = device)
    t = torch.zeros(N, device = device)
    bid_vectors = torch.zeros(N, len(A))
    ask_vectors = torch.zeros(N, len(A))
    bid_vectors = bid_vectors.to(device)
    ask_vectors = ask_vectors.to(device)
    for i in range(N - 1):
        bid_vector, ask_vector = DNN_Policy_Stochastic(t[i], S[i], q[i], net, gamma, z, B)
        bid_vectors[i] = bid_vector
        ask_vectors[i] = ask_vector
        buy_orders[i], sell_orders[i] = Market_Order_Generator(bid_vector, ask_vector, A, B, dt)
        for j in range(len(A)):
            q[i + 1] += (buy_orders[i][j] - sell_orders[i][j]) * z[j]
        q[i + 1] += q[i]
        t[i + 1] = t[i] + dt
        
    return S, buy_orders, sell_orders, q, t, bid_vectors, ask_vectors




