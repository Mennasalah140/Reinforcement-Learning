##  Grid Maze — Policy Iteration (Q&A)

This section contains detailed answers to the assignment questions related to Policy Iteration on a 5×5 stochastic Grid Maze environment.

---

###  **Q1 — What is the state-space size of the 5×5 Grid Maze problem?**

There are three possible interpretations of the state space depending on how we model the environment:

| Interpretation | State Definition | State Space Size | Notes |
|--------------|------------------|------------------|-------|
| **Policy Iteration (correct for DP)** | Agent position only `(Sx, Sy)` | `5 × 5 = 25` | Used for Dynamic Programming (fixed MDP) |
| Full environment observation | Agent, Goal, X1, X2 coordinates | `5^8 = 390,625` | Used only in Gym observation, not DP |
| All distinct cell placements | S, G, X1, X2 in unique cells | `25 × 24 × 23 × 22 = 303,600` | Represents layout permutations |

 **Why 25 is the correct answer for Policy Iteration?**  
Policy Iteration assumes a **fixed and stationary MDP**, meaning:
- The environment layout does not change during evaluation
- Transition probabilities remain constant
- Only the **agent’s position** determines the current state

 **Final DP state-space size: `25` states**

---

###  **Q2 — How to optimize the policy iteration for the Grid Maze?**

Policy Iteration can be optimized on two levels:

---

#### **A) Local Optimization (inside DP) — Faster, same results**

These optimizations reduce computation time without changing the final policy:

- **Stop policy evaluation early** using a convergence threshold
- **Cache transition probabilities** (since movement probabilities are fixed: 70% / 15% / 15%)
- **Use NumPy vectorization** instead of nested Python loops
- **Use Modified Policy Iteration** (fewer evaluation sweeps each iteration)

| Type | Meaning | Examples |
|-------|---------|----------|
| Local optimization | Makes DP *faster* but same output | Threshold stopping, NumPy vectorization |

---

#### **B) Global Optimization (beyond DP) — Better scalability**

For larger or dynamic environments, the best “optimization” is to **replace DP entirely**:

| Replace DP With | Why it scales better |
|------------------|---------------------|
| **Q-Learning / SARSA** | No need to evaluate all states or know transition probabilities |
| **DQN / PPO / Actor-Critic** | Works for large or continuous state spaces |
| **Monte-Carlo / TD methods** | More efficient than DP for bigger problems |

| Type | Meaning | Examples |
|-------|---------|----------|
| Global optimization | Makes the *approach scalable* | Q-Learning, Deep RL |

 Summary:  
- Local optimization = **faster Policy Iteration**  
- Global optimization = **use a better RL approach for bigger mazes**

---

###  **Q3 — How many iterations did it take to converge?**

Policy Iteration on a 5×5 grid typically converges in:

 **3 policy iterations**

This happens because the state space is small (25 states), so the value function stabilizes quickly.

---

###  **Q4 — How does policy iteration behave with multiple goal cells?**

Policy Iteration will:
- Treat all goal states as terminal
- Generate **multiple value gradients**
- Drive the agent toward the **nearest goal** (shortest expected path)

Example:  
If goals are at `(0,4)` and `(4,4)`:
- Agent at `(2,4)` will move **up**
- Agent at `(4,2)` will move **right**

The optimal policy adapts to the closest rewarding terminal state.

---

###  **Q5 — Can Policy Iteration work on a 10×10 maze?**

 **Yes**, because the environment is still **finite and discrete**.

However, it becomes **computationally slower**, since Policy Iteration must evaluate **all states in every iteration**:

- 5×5 → 25 states
- 10×10 → 100 states (4× more than 5×5)

It works, but scales poorly as the grid grows.

---

###  **Q6 — Can Policy Iteration work on a continuous-space maze?**

 **No**, not in its classical tabular form.

Policy Iteration requires:
- A **finite enumerable state space**
- A **known transition model**

Continuous environments have **infinite states**, so DP cannot run. Instead, we must use **function approximation or Deep RL (e.g., DDPG, PPO).**

---

###  **Q7 — Can Policy Iteration work with moving bad cells (like ghosts)?**

 **No.** Moving enemies make the environment **non-stationary**, which violates Policy Iteration assumptions. DP requires a **fixed transition model** that does not change over time.

For dynamic environments, use **online RL methods** such as:
- Q-Learning
- SARSA
- DQN / Actor-Critic

---

