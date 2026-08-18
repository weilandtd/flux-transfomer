from optlang import Variable, Constraint, Objective
from cobra.util.solver import check_solver
import numpy as np
from scipy import linalg
from scipy.optimize import linprog

def right_kernel_stoich_matrix_qr(S):
    """
    Find the right nullspace using QR factorization 
    this results in a non-sparse null space
    """

    rank_S = np.linalg.matrix_rank(S)
    rank_deficiency = S.shape[1] - rank_S

    Q, R, P = linalg.qr(S.T, pivoting=True)

    kernel = Q[:,-rank_deficiency:]

    return kernel

def right_kernel_stoich_matrix_milp(S, eps=0.1, M = 1000, solver='cplex'):
    """Find a sparse right nullspace of S via an elementary-flux-mode MILP formulation. Requires cplex or gurobi."""
    if solver not in ["cplex", "gurobi"]:
        raise NotImplementedError("Only cplex and gurbi interfaces support the indicator constraint!")

    # Charateristics of the stoichometric matrix
    rank_S = np.linalg.matrix_rank(S)
    rank_deficiency = S.shape[1] - rank_S
    num_reactions = S.shape[1]
    num_mass_balances = S.shape[0]

    # Solution stoarge
    kernel_vectors = [] 
    objective_val = []
    binary_solution = []
    linear_dependence = []

    # Define model with the desired solver
    interface = check_solver(solver)
    model =interface.Model(name="kernel-problem")

    # Devine variables
    kp = [Variable(f'k_pos_{i}', lb=0, ub=M) for i in range(num_reactions)]
    kn = [Variable(f'k_neg_{i}', lb=0, ub=M) for i in range(num_reactions)]

    Zp = [Variable(f'Z_pos_{i}', lb=0, ub=1, type='binary') for i in range(num_reactions)]
    Zn = [Variable(f'Z_neg_{i}', lb=0, ub=1, type='binary') for i in range(num_reactions)]

    binary_vars = Zp + Zn

    # Equality constraints 
    # [S |-S | 0 | 0  ]
    # [0 | 0 | I | I  ].[kp;kn;Zp;Zn] = 0

    constraints = []
    # Stoichiometric constraints 
    for i in range(num_mass_balances):  
        expression = 0
        for j in range(num_reactions):
            expression += S[i,j] * kp[j] - S[i,j] * kn[j]
        c = Constraint(expression, ub=0, lb=0)
        constraints.append(c)

    # Indicator constraints
    # Zi = 0 ki =0 / Zi =1,  kp > 1 
    for i in range(num_reactions):
        cp = Constraint( kp[i], indicator_variable=Zp[i], active_when=1, lb=1)
        constraints.append(cp)
    
        cp = Constraint( kp[i], indicator_variable=Zp[i], active_when=0, ub=0)
        constraints.append(cp)

        cn = Constraint( kn[i], indicator_variable=Zn[i], active_when=1, lb=1)
        constraints.append(cn)
    
        cn = Constraint( kn[i], indicator_variable=Zn[i], active_when=0, ub=0)
        constraints.append(cn)

    # Exclusion constraint 
    # Zp + Zn <= 1
    for i in range(num_reactions):
        expression = Zp[i] + Zn[i]
        c = Constraint(expression, ub=1)
        constraints.append(c)

    # Exclude trivial solution 
    expression = sum(Zp) + sum(Zn)
    trivial_constraint = Constraint(expression, lb=1)
    constraints.append(trivial_constraint)

    # minimize 
    expression = sum(kp) + sum(kn)
    obj = Objective(expression, direction='min')

    model.objective = obj
    model.add(constraints)

    # Finding linerly independent kernel vectors 
    print("Finding linerly independent kernel vectors")

    this_rank = 0

    while this_rank < rank_deficiency:

        if not kernel_vectors == []:
            # Add exclusion constraints 
            # sum(Zi^sol * Zi) <= sum(Zi^sol) - 1 
            bs = np.array(binary_solution[-1])
            obj_val = objective_val[-1]
            expression = sum([ s*v for s,v in zip(bs,binary_vars) ] )
            c = Constraint(expression, ub=sum(bs)-1 )
            model.add([c]) 
            # Exclude complementary binary solution: 
            s = bs[:num_reactions] + bs[num_reactions:]
            zp_comp = np.abs(bs[:num_reactions] - s)
            zn_comp = np.abs(bs[num_reactions:] - s)
            bs_comp = np.concatenate([zp_comp,zn_comp])
            expression = sum([ s*v for s,v in zip(bs_comp,binary_vars) ] )
            c = Constraint(expression, ub=sum(bs_comp)-1 )
            model.add([c]) 

        # Solve the model
        status = model.optimize()
        print(f"Solver {status}: {model.objective.value}")

        if status == "optimal":

            bs = [b.primal for b in binary_vars]
            obj_val = model.objective.value
            objective_val.append(obj_val)
            binary_solution.append(bs)

            # Get the kernel vector 
            k = np.array([kp[i].primal - kn[i].primal 
                          for i in range(num_reactions)])
            
            kernel_vectors.append(k)
            
            # Test for linear depdence
            kernel = np.stack([v for v in kernel_vectors]).T

            rank = np.linalg.matrix_rank(kernel)
            if rank > this_rank:
                linear_dependence.append(True)
            else:
                linear_dependence.append(False)

            print(f"Current rank {this_rank}")
            this_rank = rank

        else:
            raise RuntimeError("MILP infeasible cant find Kernel")

    # final kernel 
    kernel = np.stack([v for v, _ld in zip(kernel_vectors, linear_dependence) if _ld]).T

    if np.linalg.matrix_rank(kernel) < rank_deficiency:
         raise RuntimeError("Kernel is not full rank!")
         
    # Test if kernel is truly a kernel of S
    null = np.abs(S.dot(kernel)).sum()
    toll = S.shape[0]*S.shape[1]*1e-9
    if null > toll:
        raise RuntimeError(f"Kernel is not a kernel of S {null} > {toll}")

    return kernel            
        

# Legacy formulation 
def right_kernel_stoich_matrix_legacy(S, eps=1e-3):
    """Legacy MILP-based right kernel of S; superseded by right_kernel_stoich_matrix_milp."""

    rank_S = np.linalg.matrix_rank(S)
    # kernel = np.zeros((S.shape[1], S.shape[1] - rank_S))
    rank_deficiency = S.shape[1] - rank_S

    num_reactions = S.shape[1]

    kernel_vectors = [] 
    objective_val = []
    binary_solution = []
    # Equality constraints 
    # [S |-S | 0 | 0 | 0 ]
    # [0 | 0 | I | I |-I ].[kp;kn;Zp;Zn;U] = 0

    # Where k is continous and Z a binary variable 
    # Constraint the 0 <= k <= 1
    # [I | I ].[k;Z] <= [1]

    # Maximize k 

    # Add exclusion constraints 
    # [0 | sol_Z ].[k;Z] = [ob-1]


    Zero = np.zeros_like(S)

    A1 = np.concatenate([S, -S, Zero, Zero, Zero], axis=1)

    #I = np.eye(num_reactions)
    #Zero = np.zeros_like(I)
    #A2 = np.concatenate([Zero, Zero, I, I, -I], axis=1)

    # sum_constraints = np.concatenate([np.ones((1,S.shape[1])), 
    #                                   np.ones((1,S.shape[1])), 
    #                                   np.zeros((1,S.shape[1])),
    #                                   np.zeros((1,S.shape[1])),
    #                                   np.zeros((1,S.shape[1])),], axis=1)
    
    A = np.concatenate([A1], axis=0)
            
    b = np.zeros(A.shape[0])
    #b[-1] = 1.0
    
    eye = np.eye(num_reactions)
    Zero = np.zeros_like(eye)

    A_ub_0 = np.concatenate([-eye, Zero, -2 * eps * eye, Zero, Zero], axis=1)
    b_ub_0 = -eps * np.ones(num_reactions)

    A_ub_1 = np.concatenate([Zero, -eye, Zero, -2 * eps * eye, Zero], axis=1)
    b_ub_1 = -eps * np.ones(num_reactions)

    A_ub_2 = np.concatenate([eye, Zero, eye, Zero, Zero], axis=1)
    b_ub_2 = np.ones(num_reactions)

    A_ub_3 = np.concatenate([Zero, eye, Zero, eye, Zero], axis=1)
    b_ub_3 = np.ones(num_reactions)

    A_ub_4 = np.concatenate([Zero, Zero, -eye, -eye, 2 * eye], axis=1)
    b_ub_4 = np.zeros(num_reactions)

    A_ub_6 = np.concatenate([Zero, Zero, eye, eye, -eye], axis=1)
    b_ub_6 = np.ones(num_reactions)

    A_ub_7 = np.concatenate([Zero, Zero, -eye, -eye, Zero], axis=1)
    b_ub_7 = -np.ones(num_reactions)

    A_ub_5 = -1 * np.concatenate([  np.ones((1,S.shape[1])), 
                                    np.ones((1,S.shape[1])), 
                                    np.zeros((1,S.shape[1])),
                                    np.zeros((1,S.shape[1])),
                                    np.zeros((1,S.shape[1])),], axis=1)
    b_ub_5 = -np.ones(1)



    for k in range(rank_deficiency):
        if kernel_vectors == []:

            A_ub = np.concatenate([A_ub_0, A_ub_1, A_ub_2, A_ub_3, A_ub_4, A_ub_5, A_ub_6, A_ub_7, ], axis=0)
            b_ub = np.concatenate([b_ub_0, b_ub_1, b_ub_2, b_ub_3, b_ub_4, b_ub_5, b_ub_6, b_ub_7, ], axis=0)
        else:
            A_ub_ex_1 = np.stack([np.concatenate([np.zeros(4*num_reactions), v])
                                for v in binary_solution])
            b_ub_ex_1 = [ -o-1 for o in objective_val]

            A_ub = np.concatenate([A_ub_0, A_ub_1, A_ub_2, A_ub_3, A_ub_4, A_ub_5, A_ub_6, A_ub_7,
                                   A_ub_ex_1], axis=0)
            b_ub = np.concatenate([b_ub_0, b_ub_1, b_ub_2, b_ub_3, b_ub_4, b_ub_5, b_ub_6, b_ub_7,
                                   b_ub_ex_1], axis=0)


        # Objective function
        c = np.zeros(A.shape[1]) 
        c[-num_reactions:] = -1.0 # Maximize the number of binary variables

        integrality = np.zeros_like(c)
        integrality[2*num_reactions:] = 1


        res = linprog(c=c, A_eq = A, b_eq = b, 
                      A_ub = A_ub, b_ub = b_ub, 
                      bounds=(0,1),
                      integrality=integrality,)

        if res.x is not None:
            # Nomrmalize the 
            objective_val.append(res.fun)
            k = res.x[:num_reactions] - res.x[num_reactions:2*num_reactions] #np.array([x if abs(x) > eps else 0.0 for x in res.x[:num_reactions]])
            Z = res.x[-num_reactions:]

            # DEGBUG
            #print(Z)
            #print(res.x[2*num_reactions:3*num_reactions])
            #print(res.x[3*num_reactions:4*num_reactions])

            binary_solution.append( Z )
            kernel_vectors.append(k)
        else:
            # Compute the kernel to test stuff
            kernel = np.stack([v for v in kernel_vectors]).T
            raise ValueError(f"No solution found for kernel vector {res.status}")
    # final kernel 
    kernel = np.stack([v for v in kernel_vectors]).T
    return kernel
