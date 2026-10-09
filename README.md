# Reinforcement Learning Algorithms and Analysis

**Project Level:** Postgraduate  

**Author:** Padma Michela Ricca  
**Institution:** University College London (UCL)  
**Degree:** MSc Artificial Intelligence for Biomedicine and Healthcare  
**Module:** COMP0089 (Reinforcement Learning)  
**Academic Year:** 2025/2026  
**Assessment Outcome:** 91.58% (Distinction)  
**Overall Module Mark:** Pending  
**Project Type:** Reinforcement learning algorithm implementation, experimental evaluation and theoretical analysis project

## Project Overview

Implemented and evaluated a range of reinforcement learning algorithms across four postgraduate coursework components, progressing from multi-armed bandits and tabular temporal-difference learning to neural actor–critic methods and off-policy multi-step value estimation.

The project combined algorithm development in Python, NumPy and JAX with experimental investigations and mathematical analysis, using course-provided bandit, grid-world and Catch environments. Particular emphasis was placed on understanding how exploration strategies, changing reward distributions, function approximation, optimisation methods and off-policy corrections influence learning performance and stability.

The findings highlighted important trade-offs between exploration and adaptation, convergence speed and training stability, and variance reduction and the reliability of bootstrapped value estimates.

## Contributions

- Implemented Upper Confidence Bound (UCB), ε-greedy and REINFORCE agents using NumPy, incorporating incremental reward estimation, configurable exploration schedules, a square-max policy parameterisation and optional reward baselines.
- Compared cumulative and instantaneous regret across stationary and non-stationary Bernoulli bandits, investigating reward transformations, premature exploitation and the differing abilities of learning agents to adapt following abrupt changes in reward structure.
- Proposed sliding-window UCB, discounted UCB and change-detection-based resets as potential improvements for non-stationary environments, examining how limiting the influence of historical observations could improve responsiveness to distribution shifts.
- Implemented one-step temporal-difference (TD) learning for on-policy state-value estimation in a grid world, alongside theoretical analysis of iterative policy improvement and the distinction between random-policy evaluation and optimal value estimation through Q-learning.
- Derived and implemented expected off-policy TD(λ) updates with linear function approximation, examining 25 target-policy and trace-parameter configurations and analysing convergence behaviour through update matrices, stability conditions and eigenvalue-based reasoning.
- Developed a JAX-based neural actor–critic architecture with a shared 50-unit ReLU hidden layer and separate state-value and action-preference outputs, implementing bootstrapped policy gradients, a value-function baseline and semi-gradient TD updates.
- Implemented an alternative ε-greedy preference-learning agent and a custom Adam-style optimiser, conducting 40,000-episode training comparisons with stochastic gradient descent (SGD) in the Catch environment to investigate convergence speed, training instability and the limitations of persistent exploration.
- Developed four off-policy multi-step return estimators using full importance sampling, per-decision importance sampling, control variates and adaptive bootstrapping. In a comparison across 1,000 sampled trajectories, adaptive bootstrapping achieved the lowest observed mean squared value error (approximately 1.89), compared with approximately 655 for full importance sampling under the tested conditions.
- Analysed the convergence and variance properties of two-step Bellman-style learning updates, using contraction arguments, importance-weighted corrections and state-action visitation assumptions to examine the effects of greedy, uniformly random and ε-greedy behaviour policies.

## Skills Demonstrated

- **Reinforcement Learning:** Understanding of sequential decision-making, stochastic policies, exploration–exploitation trade-offs, value estimation and policy optimisation.
- **Scientific Programming:** Practical experience with Python, NumPy, JAX automatic differentiation, just-in-time compilation and structured numerical computation.
- **Deep Learning and Optimisation:** Neural network parameterisation, online gradient-based training, custom update rules and adaptive optimisation techniques.
- **Mathematical and Statistical Analysis:** Applying Bellman operators, contraction mappings, matrix-based stability analysis and bias–variance reasoning to investigate learning behaviour.
- **Experimental Evaluation and Technical Documentation:** Interpreting learning curves, regret measurements and error distributions, critically assessing methodological limitations and communicating quantitative findings.

## Report Availability

The original assessed Jupyter notebooks are currently withheld from public access  to uphold academic integrity.

This repository provides an overview of the project's objectives, implemented algorithms, experimental methodology, evaluation findings and skills demonstrated.

Further documentation may be made available upon request, in accordance with the University's policies.
