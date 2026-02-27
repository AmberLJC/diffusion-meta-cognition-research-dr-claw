# Initial Research Directions

_Generated at project start. To be refined by deep search._

## Core Focus
Meta-cognition in language models: how models "think about" their own uncertainty, knowledge boundaries, and reasoning — and how this differs between diffusion-based and autoregressive architectures.

## Candidate Directions (to be evaluated)

### 1. Uncertainty Geometry in Diffusion vs AR
- AR models express uncertainty via next-token probability distributions
- Diffusion models express uncertainty via the denoising trajectory itself
- **Novel angle:** Do diffusion LLMs develop richer internal uncertainty representations?
- **Why unexplored:** Most uncertainty work focuses on AR models (Bayesian NNs, conformal prediction)

### 2. Meta-Cognitive Tokens in Masked Diffusion
- MDLM (masked diffusion language models) predict all tokens simultaneously
- **Novel angle:** Can specific "meta-tokens" emerge that regulate the global denoising plan?
- **Why unexplored:** Interpretability work on diffusion LLMs is nascent

### 3. Planning vs Reactive Generation
- AR = fundamentally reactive (each token depends on past)
- Diffusion = globally planned from the start
- **Novel angle:** Does the global view enable better multi-hop reasoning or task decomposition?
- **Experiment:** Compare chain-of-thought accuracy across MDLM vs GPT-style models

### 4. Knowledge Boundary Awareness
- When does a model "know it doesn't know"?
- **Novel angle:** Diffusion models' iterative refinement may encode epistemic uncertainty differently
- **Why unexplored:** No paper has directly studied knowledge boundary signals in diffusion LLMs

### 5. Transfer of Meta-Learning Properties
- MAML and similar meta-learning frameworks were designed for AR/discriminative models
- **Novel angle:** Can diffusion LLMs be meta-trained to adapt faster via their denoising objective?

## Priority Ranking (TBD after literature search)
[ ] Direction 1: Uncertainty Geometry
[ ] Direction 4: Knowledge Boundary Awareness  
[ ] Direction 3: Planning vs Reactive
[ ] Direction 2: Meta-Cognitive Tokens
[ ] Direction 5: Meta-Learning Transfer
