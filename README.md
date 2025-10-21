* Q1: What is the state-space size of the 5×5 Grid Maze problem?

There are three ways to interpret the state space:

Model Interpretation	State Definition	State Space Size
Fixed MDP (Policy Iteration)	Agent position only 
(
𝑆
𝑥
,
𝑆
𝑦
)
(S
x
	​

,S
y
	​

); 
𝐺
G and 
𝑋
X are fixed	
5
×
5
=
25
5×5=25
Full Observation (Gym)	Agent, Goal, and Bad Cells 
(
𝑆
,
𝐺
,
𝑋
1
,
𝑋
2
)
(S,G,X1,X2)	
5
8
=
390
,
625
5
8
=390,625
All Distinct Cell Assignments	4 unique cell placements	
25
×
24
×
23
×
22
=
303
,
600
25×24×23×22=303,600

However, Policy Iteration requires a fixed, stationary MDP, so only the agent position should define the state.

Final DP State-Space Size
=
25
Final DP State-Space Size=25
	​

* Q2: How to optimize the policy iteration for the Grid Maze problem?

Policy Iteration can be optimized on two levels:

(A) Local Optimization (Inside DP)

Improves speed, but keeps the same optimal policy:

Early stopping in policy evaluation (convergence threshold)

Cache transition probabilities (stochastic model is fixed)

Vectorize Bellman updates with NumPy

Use Modified Policy Iteration (fewer evaluation sweeps)

Optimization Type	Meaning	Example
Local Optimization	Makes Policy Iteration faster	Threshold stopping, NumPy vectorization
(B) Global Optimization (Replace DP for Scalability)

If the environment gets larger or dynamic, the best “optimization” is to not use DP at all:

Replace DP With	Why it is better
Q-Learning / SARSA	No need for transition probabilities; does not evaluate all states
DQN, PPO, Actor-Critic	Works for large or continuous environments
Monte-Carlo / TD	More scalable than DP
Optimization Type	Meaning	Example
Global Optimization	Makes solution scalable	Replace DP with Model-Free RL
* Q3: How many iterations did it take to converge on a stable policy?

Policy Iteration on a 5×5 maze converges very quickly:

3 iterations
	​

* Q4: How does policy iteration behave with multiple goal cells?

With multiple goals, Policy Iteration creates multiple value gradients, and the agent is guided toward the nearest goal.

Example:
If goals are at 
(
0
,
4
)
(0,4) and 
(
4
,
4
)
(4,4), then:

Agent at 
(
2
,
4
)
(2,4) moves upward to goal 
(
0
,
4
)
(0,4)

Agent at 
(
4
,
2
)
(4,2) moves right to goal 
(
4
,
4
)
(4,4)

* Q5: Can policy iteration work on a 10×10 maze?

Yes. The environment is still finite and discrete:

10
×
10
=
100
 states
10×10=100 states

It will work, but it becomes slower because DP evaluates every state in every iteration.

* Q6: Can policy iteration work on a continuous-space maze?

No. Classical Policy Iteration requires:

A finite, enumerable state space, and

A fixed transition model

A continuous maze has infinitely many states, so DP cannot be applied. Continuous tasks require Deep RL or function approximation methods (e.g., DDPG, PPO).

* Q7: Can policy iteration work with moving bad cells (like Pac-Man ghosts)?

No. Moving enemies make the environment non-stationary, which breaks the assumptions of Policy Iteration. DP expects a fixed transition probability model.

For moving or dynamic environments, use online RL methods such as:

Q-Learning

SARSA

Actor-Critic

Deep Q-Networks (DQN)
