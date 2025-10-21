## Assignment Answers

### **Q1 — What is the state-space size of the 5×5 Grid Maze?**

There are three possible interpretations of the state space:

| Interpretation | State Definition | Size |
|--------------|------------------|------|
| **Policy Iteration (correct for DP)** | Agent position only | `5 × 5 = 25` |
| Full Gym observation | Agent + Goal + 2 Bad Cells | `5^8 = 390,625` |
| All distinct cell placements | S, G, X1, X2 in unique cells | `25 × 24 × 23 × 22 = 303,600` |

For **Policy Iteration**, the environment must be a *fixed MDP*, so only the agent’s position counts as the state.

 **Final Answer: `25` states**

---

### **Q2 — How to optimize the policy iteration for the Grid Maze?**

There are **two levels of optimization**:

####  *Local Optimization (inside DP)* — Faster, same result
- Use a convergence threshold
- Cache transition probabilities
- Use NumPy vectorization
- Use Modified Policy Iteration

| Optimization Type | Meaning | Example |
|-------------------|---------|---------|
| Local Optimization | Makes Policy Iteration faster | Threshold stopping, NumPy vectorization |

####  *Global Optimization (beyond DP)* — Scales better
| Replace DP With | Why |
|------------------|------|
| Q-Learning / SARSA | No need to evaluate all states or know transition probabilities |
| DQN / PPO / Actor-Critic | Works for large or continuous environments |
| Monte-Carlo / TD | More scalable than DP |

| Optimization Type | Meaning | Example |
|-------------------|---------|---------|
| Global Optimization | Makes the solution scalable | Replace DP with Model-Free RL |

---

### **Q3 — How many iterations to converge?**
Policy Iteration in a 5×5 grid converges in:

 **3 iterations**

---

### **Q4 — How does policy iteration behave with multiple goal cells?**
The agent will follow the value gradient toward the **nearest goal**, creating the shortest-path policy.

---

### **Q5 — Can policy iteration work on a 10×10 maze?**
 **Yes**, it still works (100 states), but will be **slower** because DP updates every state in every iteration.

---

### **Q6 — Can policy iteration work on a continuous-space maze?**
 **No.** Continuous spaces have **infinite states**, so DP cannot enumerate or evaluate them. Requires **Deep RL** instead.

---

### **Q7 — Can policy iteration work with moving bad cells (ghosts)?**
 **No.** Moving enemies make the environment **non-stationary**, which breaks DP assumptions. Use **Q-Learning, SARSA, DQN**, etc.

---
