"""
Wednesday 22nd October
Problem sheets 3 and 4
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import pandas as pd
from scipy.integrate import quad


"""
Problem sheet 4
"""
def ps4problem4():
    def metropolis(f, f_prop, x_0, N=10000, kwrgs = None):
        samples = []
        current = x_0
        samples.append(current)
        
        accepted_count = 0

        for _ in range(N - 1):
            proposal = [f_prop(i, kwrgs) for i in current]
            
            prob_current = f(current)
            prob_proposal = f(proposal)
            
            acceptance_ratio = prob_proposal / prob_current
            
            if np.random.uniform(0, 1) < min(1, acceptance_ratio):
                current = proposal
                accepted_count += 1                
            samples.append(current)
            
        # print(f"Acceptance Rate: {accepted_count / N:.2f}")
        return np.array(samples)
    
    def metropolis_integration(f, f_prop, f_comparison, x_0, N=10000, kwrgs = None):
        samples = metropolis(f_comparison, f_prop, x_0, N, kwrgs)

        burn_in = N // 10
        samples_afterburn = samples[burn_in:]

        f_samples = np.array([f(x) for x in samples_afterburn])
        f_samples_comparison = np.array([f_comparison(x) for x in samples_afterburn])

        arr = f_samples/f_samples_comparison
        return np.mean(arr)
    
    def gaussian(x_current, sigma=0.1):
        return np.random.normal(loc=x_current, scale=sigma)
        
    def func2d(x):
        x1, x2 = x
        r = 1
        if x1**2 + x2**2 <= r**2:
            return 1.
        else:
            return 0.
    
    def compare2d(x):
        x1, x2 = x
        r = 1
        if (-r <= x1 <= r) and (-r <= x2 <= r):
            return 1 / (4 * r * r)
        else:
            return 0.0
        
    def func3d(x):
        x1, x2, x3 = x
        r = 1
        if x1**2 + x2**2 + x3**2 <= r**2:
            return 1.
        else:
            return 0.
        
    def compare3d(x):
        x1, x2, x3 = x
        r = 1
        if (-r <= x1 <= r) and (-r <= x2 <= r) and (-r <= x3 <= r):
            return 1.0 / (8 * r * r * r)
        else:
            return 0.0
        
    
    # samples2d = metropolis(func2d, gaussian, [0.5,0.5])
    # error2d = []
    # error3d = []
    # N = np.arange(1000,100000,100)
    # for i in N:
    #     I2d = metropolis_integration(func2d, gaussian, compare2d, [0.5,0.5], N= i)
    #     I3d = metropolis_integration(func3d, gaussian, compare3d, [0.5,0.5,0.5], N=i)
    #     error3d.append(((4/3 * np.pi) - I3d)/(4/3 * np.pi))
    #     error2d.append((np.pi - I2d)/np.pi)

    # plt.plot(N, error2d, label = '2d')
    # plt.plot(N, error3d , label = '3d')
    # plt.grid()
    # plt.xlabel("N")
    # plt.ylabel("Error")
    # plt.legend()
    # plt.show()

    error2d = []
    error3d = []
    sigma = np.arange(1, 0.0001, -0.001)
    for i in sigma:
        I2d = metropolis_integration(func2d, gaussian, compare2d, [0.5,0.5], kwrgs = i)
        I3d = metropolis_integration(func3d, gaussian, compare3d, [0.5,0.5,0.5], kwrgs = i)
        error3d.append(((4/3 * np.pi) - I3d)/(4/3 * np.pi))
        error2d.append((np.pi - I2d)/np.pi)

    plt.plot(sigma, error2d, label = '2d')
    plt.plot(sigma, error3d , label = '3d')
    plt.grid()
    plt.xlabel("N")
    plt.ylabel("Sigma")
    plt.legend()
    plt.show()


    # plt.scatter(samples2d[:,0], samples2d[:,1])
    # plt.show()

    # samples3d = metropolis(func3d, gaussian, [0.5,0.5,0.5])
    

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(samples3d[:,0], samples3d[:,1], samples3d[:,2]) 
        
def ps4problem3():
    def MC_integrate_nd(f, nd, low, high, N = 10000):
        high = np.array(high)
        low = np.array(low)

        V = np.prod(high - low)

        X = np.random.uniform(low, high, size=(N, nd))
        fx = np.array([f(x) for x in X])
    
        mean =  sum(fx) / N
        error = V * np.sqrt(1 / (N * (N - 1)) * np.sum((fx - mean)**2))
        return mean * V, error
    
    
    def func1d(x):
        return np.sin(x**2) / x
    
    def func2d(x):
        x1, x2 = x
        r = 0.5
        if x1**2 + x2**2 <= r**2:
            return x1**2 + x2**2
        else:
            return 0

    def func3d(x):
        x1, x2, x3 = x
        r = 0.5
        if x1**2 + x2**2 + x3**2 <= r**2:
            return x1**2 + x2**2 +x3**2
        else:
            return 0
    
    I, err = MC_integrate_nd(func1d, 1, 1, 10)
    true, _ = quad(func1d, 1, 10)
    print(f"{I} +- {err}")
    print(f"Difference from true value: {I - true}")

    I, err = MC_integrate_nd(func2d, 2, [-0.5, -0.5], [0.5, 0.5], N=100000)
    print(f"Integral ≈ {I:.6f} ± {err:.6f}")    

    I, err = MC_integrate_nd(func3d, 3, [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5], N=100000)
    print(f"Integral ≈ {I:.6f} ± {err:.6f}")    



"""
Problem sheet 3
"""

def problem4():
    def gauss_inverse(A):
        A = np.array(A)
        rows, cols = A.shape
        I = np.eye(rows, cols)

        for c in range(0, cols):
            if A[c, c] == 0:
                print(c)
                r_swap = -1
                for r_find in range(c + 1, rows):
                    if A[r_find, c] != 0:
                        r_swap = r_find
                        break

                if r_swap != -1:
                    A[[c, r_swap]] = A[[r_swap, c]]
                    I[[c, r_swap]] = I[[r_swap, c]]
                
                else:
                    raise ValueError("Matrix is singular")

            factor = A[c, c]
            A[c, :] /= factor
            I[c, :] /= factor
            for r in range(c + 1, rows):
                factor = A[r,c]
                A[r, :] -= (factor * A[c, :])
                I[r, :] -= (factor * I[c, :])

        for c in range(cols - 1, -1, -1):
            for r in range(c):
                factor = A[r, c]
                A[r, :] -= (factor * A[c, :])
                I[r, :] -= (factor * I[c, :])
        return I


    def jacobian(F, x, h=1e-5):
        """
        Numerical Jacobian using central finite differences
        F: function R^n -> R^m
        x: array-like (length n)
        h: small step size
        """
        x = np.array(x, dtype=float)
        f0 = F(x)
        m = len(f0)
        n = len(x)
        J = np.zeros((m, n))
        
        for j in range(n):
            dx = np.zeros_like(x)
            dx[j] = h
            J[:, j] = (F(x + dx) - F(x - dx)) / (2 * h)
        return J
    
    def newton_raphson_2d(f, x_0, stop_diff = 1e-5, max_step = 100):
        step = 0 
        diff = 1
        x_i = x_0
        while (norm(diff) > stop_diff) and (step < max_step):
            x_prev = x_i 
            J = jacobian(f, x_i)
            x_i = x_i - np.dot(gauss_inverse(J), f(x_i))

            diff = np.abs(x_i - x_prev)
            step += 1 
            
        print(f"finished with {step} iterations")
        return x_i

    def E(x, Q0 = 1e-9, r0 = [0,0], r1 = [1, 0], r2 = [0,1], epsilon0=8.854e-12):
        r = np.array(x)
        def contrib(r, ri, weight=1.0, eps=1e-12):
            diff = r - np.array(ri)
            dist = np.linalg.norm(diff)
            if dist < eps: return np.zeros_like(diff)
            return weight * diff / dist**3
        term0 = contrib(r, r0)
        term1 = contrib(r, r1, 0.5)
        term2 = contrib(r, r2)
        return term0 + term1 + term2

    x_i = newton_raphson_2d(E,[1, np.pi/4])
    print(x_i)



def problem3():
    def jacobi_inverse(A, b, max_step = 100, stop_diff = 1e-6):
        L = np.zeros_like(A, dtype = float)
        U = np.zeros_like(A, dtype = float)
        D = np.zeros_like(A, dtype = float)

        N = A.shape[0]
        for i in range(N):
            D[i, i] = A[i, i]

            for j in range(i):  
                L[i, j] = A[i, j]

            for j in range(i + 1, N):
                U[i, j] = A[i, j]

        step = 0 
        x_k = np.zeros_like(b, dtype=float) # This did not work originally, due to the arrays being initialised as integers not floats! 
        x_prev = np.ones_like(b, dtype=float)
        errors = []
        error = stop_diff + 1 
        while (error > stop_diff) and (step < max_step):
            x_prev = x_k.copy()

            for i in range(N):
                sum = 0 
                for j in range(N):
                    if i==j:
                        continue 
                    sum += A[i,j] * x_prev[j] 

                x_k[i] = (1/A[i,i]) * (b[i] - sum)

            error = norm(np.dot(A, x_k) - b)
            errors.append(error)
            step += 1 
            
        print(f"finished after {step} steps ")

        # plt.scatter(np.arange(step), errors)
        # plt.xlabel('iterations')
        # plt.ylabel('error')
        # plt.show()

        return x_k, step
    def build_system(N):
        A = np.zeros((N, N))
        for i in range(N):
            A[i, i] = 3
            if i > 0:
                A[i, i-1] = -1  
            if i < N - 1:
                A[i, i+1] = -1 
        b = np.zeros(N)
        b[0] = -1
        b[-1] = 5

        return A, b
    
    A, b = build_system(5)

    steps = []
    indexes = np.arange(5,300)
    for i in indexes:
        A, b = build_system(i)
        x, step = jacobi_inverse(A,b,10000)
        steps.append(step)

    plt.scatter(indexes, steps)
    plt.xlabel('size of array')
    plt.ylabel('steps taken for convergent solution')
    plt.show()




def problem2():
    def largest_eigenvalue(A, stop_diff = 1e-10, iter_lim = 100):
        x_0 = np.random.rand(A.shape[1])
        x_i = x_0
        step = 0 
        eigenvalue = 0.
        prev_eigenvalue = 1.
        
        while np.abs(prev_eigenvalue - eigenvalue) > stop_diff and step < iter_lim:
            y = A.dot(x_i)
            prev_eigenvalue = eigenvalue
            eigenvalue = max(np.abs(y))
            x_i = y / eigenvalue
            step += 1 
        print(f"finished with {step} iterations")

        return eigenvalue, step

    def build_system(N):
        A = np.zeros((N, N))
        for i in range(N):
            A[i, i] = 2
            if i > 0:
                A[i, i-1] = -1/2
            if i < N - 1:
                A[i, i+1] = -1/2 
        b = np.zeros(N)
        b[0] = -1
        b[-1] = 5

        return A, b
    
    steps = []
    evals = [] 
    indexes = np.arange(1,500)
    for i in indexes:
        A, _ = build_system(i)
        eigenvalue, step = largest_eigenvalue(A,iter_lim=10000)
        steps.append(step)
        evals.append(eigenvalue)


    plt.plot(indexes, evals)
    plt.xlabel('size of array')
    plt.ylabel('Max eigenvalue')
    plt.show()


    """
    The above do not converge as they are singular matrices. The best way is to add on a multiple of the identity matrix, aI, so that the eigenvalues are scaled by a
    """

  


def problem1():
    def LU_decomposition(A, b):
        """
        LU decomposition using Doolittle convention
        """
        A = np.array(A)
        rows, cols = A.shape
        L = np.zeros_like(A)
        U = np.zeros_like(A)

        # initial calculations
        U[0,:] = A[0, :]
        L[1,0] = A[1,0]/U[0,0]
        U[1,1] = A[1,1] - (L[1,0] * U[0,1])

        for j in range(cols):
            for i in range(j+1):
                sum = 0
                for k in range(i):
                    sum += L[i,k] * U[k,j]
                
                U[i,j] = A[i,j] - sum 

            for i in range(j, cols):
                sum = 0
                for k in range(j):
                    sum += L[i, k] * U[k, j]
                L[i,j] = 1/(U[j,j]) * (A[i,j] - sum)

            L[j, j] = 1
        # print(np.isclose(A,np.dot(L,U)))

        # forward substitution to find y
        y = np.zeros_like(b, dtype=float)
        for i in range(len(y)):
            sum_ = np.dot(L[i, :i], y[:i])
            y[i] = (b[i] - sum_) / L[i, i]   

        # Backward substitution (U * x = y)
        x = np.zeros_like(y, dtype=float)
        for i in range(len(x) - 1, -1, -1):
            sum_ = np.dot(U[i, i+1:], x[i+1:])
            x[i] = (y[i] - sum_) / U[i, i]

        return x
    
    def LU_inverse(A):
        b = np.zeros(A.shape[0], dtype=float)
        inv = np.zeros_like(A, dtype=float)

        for i in range(0, A.shape[0]):
            b[i] = 1
            inv[i,:] = LU_decomposition(A, b)
            b[i] = 0 
        return inv



    A = (1/6) * np.array([
    [5, 4, 3, 2, 1],
    [4, 8, 6, 4, 2],
    [3, 6, 9, 6, 3],
    [2, 4, 6, 8, 4],
    [1, 2, 3, 4, 5]
    ])

    b = np.array([
    [0],
    [1],
    [2],
    [3],
    [4]
    ])
    x = LU_decomposition(A,b)
    inv = LU_inverse(A)
    # print(inv)
    # print(6*np.dot(L,U))
    
    

def problem5():
    def IterativeTaylorSeries(stop_error, stop_step, g, x_0):
        def x(x_i, g):
            return 1 - (g * x_i)
    
        x_i = x_0
        step = 0
        true = np.float64(1/(1+g))
        error = stop_error + 1

        while (np.abs(error) > stop_error) and (step < stop_step):        
            x_i = x(x_i, g)
            error = (true - x_i) / true
            step += 1

        return x_i, true, step
    
    g_vals = np.linspace(-1 + 1e-5, 1 + 1e-5, 1000)
    true = [1/(1+i) for i in g_vals]
    x_vals = [i + 1 for i in g_vals]

    iterations = []
    approximations = []
    # x_i, true, step = IterativeTaylorSeries(10e-6, 1e3, i, 1)
    # # print(f"True value: {true} \nFinal x_i: {x_i} \nnumber of iterations: {step}")
    vectorized = np.vectorize(IterativeTaylorSeries)
    approximations, _, iterations = vectorized(1e-7, 1e3, g_vals, 1)
    
    true = np.array(true)
    approximations = np.array(approximations)

    error = np.abs(true - approximations)
    fract_error = error / true
    
    plt.plot(x_vals, iterations)
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("Number of iterations")
    plt.show()

    plt.plot(x_vals, approximations)
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("Approximation")
    plt.show()

    plt.plot(x_vals, fract_error)
    plt.yscale('log')
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("Fractional Error")
    plt.show()
        
def problem6():
    def gauss_inverse(A):
        A = np.array(A)
        rows, cols = A.shape
        I = np.eye(rows, cols)

        for c in range(0, cols):
            if A[c, c] == 0:
                print(c)
                r_swap = -1
                for r_find in range(c + 1, rows):
                    if A[r_find, c] != 0:
                        r_swap = r_find
                        break

                if r_swap != -1:
                    A[[c, r_swap]] = A[[r_swap, c]]
                    I[[c, r_swap]] = I[[r_swap, c]]
                
                else:
                    raise ValueError("Matrix is singular")

            factor = A[c, c]
            A[c, :] /= factor
            I[c, :] /= factor
            for r in range(c + 1, rows):
                factor = A[r,c]
                A[r, :] -= (factor * A[c, :])
                I[r, :] -= (factor * I[c, :])

        for c in range(cols - 1, -1, -1):
            for r in range(c):
                factor = A[r, c]
                A[r, :] -= (factor * A[c, :])
                I[r, :] -= (factor * I[c, :])
        return I
    
    h = 1000
    mu_0 = 4 * np.pi * 1e-7 

    T = np.array([[1, 0.526, 0.257, 0, 0, 0 ],
         [0.526, 1, 0.64, 0, 0, 0 ],
         [0.257, 0.64, 1, 0, 0, 0],
         [0, 0, 0, -1, -0.581, -0.978],
         [0, 0, 0, -0.581, -1, -0.5],
         [0, 0, 0, -0.978, -0.5, -1]
        ])
    
    T_prime = np.array([
        [0.552, 0.998, 0.61, 0, 0, 0],
        [0, 0, 0, -0.988, -0.52, -0.998]
    ])

    T_inv = gauss_inverse(T) #* 1/((2 * np.pi * h) / mu_0)
    

    df_HER = pd.read_csv("./session2_data/HER.csv")
    df_HBK = pd.read_csv("./session2_data/HBK.csv")
    df_TSU = pd.read_csv("./session2_data/TSU.csv")

    df_KMH = pd.read_csv("./session2_data/KMH.csv")
    time = pd.to_datetime(df_HER['Date_UTC'])

    Bx = np.array([df_HER['Bx'], df_HBK['Bx'], df_TSU['Bx']])
    By = np.array([df_HER['By'], df_HBK['By'], df_TSU['By']])
    B_all = np.concatenate((Bx, By))  

    I_all = (T_inv @ B_all) 
    B_prime_all = T_prime @ I_all                        

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(time, df_KMH['Bx'], label='Measured Bx (KMH)', color='k')
    plt.plot(time, B_prime_all[0, :], label='Interpolated Bx (KMH)', color='tab:blue')
    plt.ylabel('Bx [nT]')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(time, df_KMH['By'], label='Measured By (KMH)', color='k')
    plt.plot(time, B_prime_all[1, :], label='Interpolated By (KMH)', color='tab:orange')
    plt.ylabel('By [nT]')
    plt.xlabel('Time [UTC]')
    plt.legend()
    plt.grid(True)

    plt.suptitle('Interpolated vs Measured Geomagnetic Field at KMH')
    plt.show()

def problem6d():
    def jacobi_inverse(A, b, max_step = 50, stop_diff = 1e-6):
        L = np.zeros_like(A, dtype = float)
        U = np.zeros_like(A, dtype = float)
        D = np.zeros_like(A, dtype = float)

        N = A.shape[0]
        for i in range(N):
            D[i, i] = A[i, i]

            for j in range(i):  
                L[i, j] = A[i, j]

            for j in range(i + 1, N):
                U[i, j] = A[i, j]

        step = 0 
        x_k = np.zeros_like(b, dtype=float) # This did not work originally, due to the arrays being initialised as integers not floats! 
        x_prev = np.ones_like(b, dtype=float)
        errors = []
        error = stop_diff + 1 
        while (error > stop_diff) and (step < max_step):
            x_prev = x_k.copy()

            for i in range(N):
                sum = 0 
                for j in range(N):
                    if i==j:
                        continue 
                    sum += A[i,j] * x_prev[j] 

                x_k[i] = (1/A[i,i]) * (b[i] - sum)

            error = norm(np.dot(A, x_k) - b)
            errors.append(error)
            step += 1 
            
        print(f"finished after {step} steps ")

        # plt.scatter(np.arange(step), errors)
        # plt.xlabel('iterations')
        # plt.ylabel('error')
        # plt.show()

        return x_k, step
    from scipy.linalg import block_diag
    T = np.array([[1, 0.526, 0.257, 0, 0, 0 ],
         [0.526, 1, 0.64, 0, 0, 0 ],
         [0.257, 0.64, 1, 0, 0, 0],
         [0, 0, 0, -1, -0.581, -0.978],
         [0, 0, 0, -0.581, -1, -0.5],
         [0, 0, 0, -0.978, -0.5, -1]
        ])
    
    T_prime = np.array([
        [0.552, 0.998, 0.61, 0, 0, 0],
        [0, 0, 0, -0.988, -0.52, -0.998]
    ])
    N = 1000
    m=6
    T_massive = np.zeros([N*m, N*m])
    for i in range(N):
        T_massive[i*m :(i+1)*m, i*m : (i+1)*m] = T

    # use the first 1000 timesteps - this will require reformatting T and input vectors 
    df_HER = pd.read_csv("./session2_data/HER.csv")
    df_HBK = pd.read_csv("./session2_data/HBK.csv")
    df_TSU = pd.read_csv("./session2_data/TSU.csv")

    df_KMH = pd.read_csv("./session2_data/KMH.csv")
    time = pd.to_datetime(df_HER['Date_UTC'])

    Bx = np.array([df_HER['Bx'], df_HBK['Bx'], df_TSU['Bx']])
    By = np.array([df_HER['By'], df_HBK['By'], df_TSU['By']])
    B_all = np.concatenate((Bx, By)).T
    B_sliced = B_all[:N, :]
    B = B_sliced.flatten().reshape(-1, 1)

    I = jacobi_inverse(T_massive, B)
    I_reshaped = I.reshape(N, m)
    
    I_timeseries = I_reshaped.T
    B = T_prime @ I_timeseries

    time_sliced = time.iloc[:N]
    df_KMH_sliced = df_KMH.iloc[:N]

    plt.figure(figsize=(12, 8))
    plt.suptitle(f'Interpolated (Jacobi) vs Measured Geomagnetic Field (N={N})')

    plt.subplot(2, 1, 1)
    plt.plot(time_sliced, df_KMH_sliced['Bx'], label='Measured Bx (KMH)', color='k')
    plt.plot(time_sliced, B[0, :], label='Interpolated Bx (Jacobi)', color='tab:blue', linestyle='--')
    plt.ylabel('Bx [nT]')
    plt.legend() 
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(time_sliced, df_KMH_sliced['By'], label='Measured By (KMH)', color='k')
    plt.plot(time_sliced, B[1, :], label='Interpolated By (Jacobi)', color='tab:orange', linestyle='--')
    plt.ylabel('By [nT]')
    plt.xlabel('Time [UTC]')
    plt.legend()
    plt.grid(True)

if __name__ == "__main__":
    ps4problem4()