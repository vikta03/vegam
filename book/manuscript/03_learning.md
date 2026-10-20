# 8. What the network was asked to learn
@objectives Understand the training examples and five targets; calculate cross-entropy; distinguish teacher predictions from game outcomes; explain masking and duration censoring.

Everything so far explains how fixed weights transform a board. It does not yet explain why those weights make useful chess predictions. That requires the training process. The same differentiable computation is run on many recorded positions; its outputs are compared with recorded targets; derivatives tell the optimizer how parameter changes would alter the loss. The resulting weights are retained and eventually exported.

VEGAM's weights originate from random initialization and training on Leela self-play records. That is different from claiming that VEGAM generated all its own training games by playing itself during this run. A Leela search and game record supply supervision. VEGAM learns to approximate useful predictions from that data with its own architecture. The distinction matters when asking where a preference such as Bg7 ultimately came from. [S4, S5, T1]

## 8.1 What a fresh random network knows

At initialization, some structure is built in: there are 64 square tokens; the encoder identifies pieces, side, history and rule state; the move generator supplies legal candidates; projections and attention can combine the numbers; residual and output heads have their specified forms. None of this means the random parameters prefer good moves.

The fixed attention bias starts at zero. Position embeddings and adaptive left/right factors start with small random values. Linear layers use their implementation's initialization, and LayerNorm begins with its standard learned scale and offset initialization. Before training, a prediction can be accidentally plausible, but no chess competence should be inferred from the vocabulary of its tensors.

A random network already has inductive structure: shared square transformations make the same local input pattern reusable, and the move decoder shares its scoring function across candidates. Training does not need to invent matrix multiplication or legal move generation. It adjusts a structured family of functions until its outputs better fit the data.

## 8.2 One position supplies several kinds of supervision

@table targets

The policy target is a distribution over the legal moves represented by the self-play record. The raw record has 1,858 policy entries; illegal entries use a sentinel, and the decoder collects legal candidates into the padded 256-row contract. The legal mass is checked and normalized. The target need not be a single move with probability one. A teacher search may distribute support among several moves. [S4]

The state-WDL target is formed from the record's **best_q** and **best_d** fields. Here q=W-L and d=D. It is a searched teacher estimate, not automatically the final game result and not a statement that the root visit average was used. The separate outcome head learns from result fields, subject to validity masking. Different heads can legitimately disagree because their targets differ.

The action-WDL targets in version-6 records are sparse. They are available for the recorded played and best move indices; when both identify the same candidate, that does not create two independent labels for every move. Unlabeled candidates do not become losses or draws. A mask prevents their placeholder arrays from contributing to the action loss. Version-5 records lack these action targets in this decoder.

This sparsity is consequential. VEGAM emits action estimates for all legal moves at inference, but was not trained with equally strong direct action supervision on every alternative in every position. Generalization through the shared decoder supplies estimates for unlabeled cases. The action head is useful advice to search, not a proof about each candidate.

## 8.3 Converting q and draw rate into three targets

If W+D+L=1 and q=W-L, then W=(1-D+q)/2 and L=(1-D-q)/2. With q=0.4 and D=0.3, W=0.55 and L=0.15. These values are a consistent distribution. A malformed q=0.9 and D=0.4 would imply L=-0.15; the decoder validates finite and in-range values and masks invalid targets rather than learning from a negative probability.

@equation wdl

These equations define the conversion. They do not prove the teacher was correct. Labels can be noisy, bounded by its search budget or affected by adjudication. A student can learn consistent useful structure from imperfect supervision, but reducing agreement loss is not the same as proving optimal chess.

## 8.4 Cross-entropy: a penalty with a useful derivative

Take three illustrative legal moves, called Bg7, Bf4 and Kc3 only to keep the example concrete. Let the model predict p=[0.2,0.5,0.3], while the teacher target is y=[0.7,0.2,0.1]. Cross-entropy is the negative target-weighted sum of log predicted probabilities.

@equation crossentropy

The natural logarithms are log(0.2)=-1.609438, log(0.5)=-0.693147 and log(0.3)=-1.203973. Therefore the loss is -(0.7 times -1.609438 + 0.2 times -0.693147 + 0.1 times -1.203973) = **1.385633**. The model pays heavily for assigning only 0.2 to the target's dominant move.

Why a logarithm? A prediction that nearly excludes a target-supported event should be penalized strongly. The logarithm also turns products of probabilities across examples into sums, making accumulation convenient. Most usefully for our present bridge, softmax plus cross-entropy has a simple gradient with respect to each logit: **p-y**. The next chapter derives it rather than taking it on faith.

A soft target has nonzero entropy even when predicted perfectly. For y=[0.7,0.2,0.1], entropy H(y) is approximately 0.801819. If the model matches y exactly, its cross-entropy equals that value, not zero. KL divergence is CE-H(y), here about 0.583815 before the update. Comparing CE across different target sets can be misleading because their entropies differ; comparing paired predictions on the same targets avoids that particular problem.

## 8.5 Five losses share one body

The implemented objective combines policy CE, state-WDL CE, 0.5 times action-WDL CE when enabled, 0.25 times outcome CE and 0.05 times duration loss. Each valid-label term is averaged over its relevant mask. Missing labels are not silently counted as correct predictions. The cloud trainer handles denominator accounting across ranks and valid rows. [S1: losses; S5: objective_sums, update]

@equation loss_total

Shared transformer parameters receive the sum of gradient contributions from the heads. A feature useful for move policy may also help state value; two heads can also pull a shared representation in different directions. Loss coefficients affect their numerical influence, but do not directly specify percentages of chess understanding. Gradient magnitudes, target coverage and nonlinear dependencies matter too.

Turning action-Q use off during engine play does not remove the action head's past influence on the shared trained representation. It changes how search consumes a current prediction. Retraining without the action loss would be a different experiment. Similarly, outcome and duration can have shaped training even though the daily launcher disables their extra search uses.

## 8.6 Duration and the difference between an observation and a bound

A completed game can tell us how many recorded plies remained. An adjudicated or capped record may tell us only that the process continued at least to a boundary, not when it would have ended under unrestricted play. The training code distinguishes these cases. It does not treat an estimated teacher moves-left field as an observed final duration. [S4]

Duration is modeled on y=log(1+t), where t is remaining plies. The network emits mu and log_sigma. After clamping log_sigma to [-3,3], z=(y-mu)exp(-log_sigma). For an observed duration, the implemented loss is 0.5z^2+log_sigma. Additive constants and the target-only Jacobian do not change the gradients with respect to model parameters. For a right-censored duration, the loss is -log Phi(-z), the negative log probability of surviving beyond the bound.

@equation duration

For example, if the lower bound is t=19 plies, y=log(20)=2.995732. Suppose mu=3 and sigma=1. Then z is about -0.004268, and survival probability Phi(-z) is about 0.5017. A censored example penalizes assigning too little probability beyond that boundary; it does not ask the network to predict exactly nineteen plies.

The current Bg7 run emits mu=4.806509 and log_sigma=-0.442774. The quantity exp(mu)-1 is roughly 121 plies, a median-style duration prediction under the modeled transform, not a forced-mate length. The actual game ended 41 plies after this position. A single miss is not a calibration study; and the daily engine's moves-left option is disabled, so this output is not the source of the displayed evaluation or the chosen Bg7 move.

@check 11|If an action has no valid WDL label in a training record, is a placeholder draw vector a target the network should learn? Explain the mask's role.

@check 12|Why does a perfectly matched soft policy target usually have nonzero cross-entropy?

# 9. How an error changes a weight
@objectives Derive softmax cross-entropy gradients; execute a numerical parameter update; apply the chain rule through attention; understand how repeated supervised examples can shape relational computation.

A gradient answers a local numerical question: if I nudge this parameter slightly, how does the loss change? It does not name a concept or tell a narrative. But because a parameter participates in computations on board features, repeatedly nudging it in useful directions can create a function sensitive to chess relationships.

## 9.1 A derivative is a measured slope in the limit

For a one-variable function f(w), a small change delta produces approximately f(w+delta)=f(w)+f'(w)delta. If f'(w)>0, decreasing w slightly tends to reduce f. If f'(w)<0, increasing w tends to reduce f. A partial derivative holds all other variables fixed while measuring this slope for one coordinate.

For f(w)=w^2, f(2)=4 and f(2.01)=4.0401. The finite-difference slope is 4.01, approaching the derivative 4 as the step shrinks. Gradient descent takes w_new=w-eta times derivative, with eta the learning rate. From w=2 and eta=0.1, the update gives 1.6 and reduces f to 2.56. This local logic is what the multidimensional optimizer generalizes.

A neural model has millions of parameters, so a gradient is a same-shaped collection of partial derivatives. It is not obtained by trying two full forward passes for every weight. Backpropagation reuses intermediate values and applies the chain rule efficiently through the computational graph.

## 9.2 Derive the policy-logit gradient

Let logits be l_i, softmax probabilities p_i and target distribution y_i. Because log p_i = l_i - log(sum_j exp(l_j)), the cross-entropy becomes -sum_i y_i l_i + log(sum_j exp(l_j)), using sum_i y_i=1. Differentiating with respect to l_k gives -y_k + exp(l_k)/sum_j exp(l_j) = **p_k-y_k**.

@equation ce_gradient

For the preceding toy example, p-y=[-0.5,+0.3,+0.2]. Gradient descent therefore raises the Bg7 logit and lowers the other two. The direction follows the discrepancy between prediction and target. No instruction such as "bishops belong on g7" is supplied separately.

Choose initial logits log(p)=[-1.609438,-0.693147,-1.203973]. Their softmax is exactly p because p already sums to one. With a direct logit learning step eta=0.1, the new logits are [-1.559438,-0.723147,-1.223973]. Softmax gives approximately **[0.212477,0.490354,0.297169]**. Cross-entropy falls from **1.385633 to 1.348115**. This is a small move toward the target, not an immediate perfect solution.

@figure toy_training|ILLUSTRATIVE. Before prediction, after one computed gradient step and teacher target. The step reduces the explicitly calculated cross-entropy. These are teaching numbers, not a historical update from the training run.

## 9.3 Make it a weight update, not a magical logit edit

Suppose the last layer uses one hidden activation h=2, and each move's illustrative logit is w_i h. Set w_i=log(p_i)/2. The derivative with respect to w_i is (p_i-y_i)h, so the weight gradient is [-1,+0.6,+0.4]. Use weight learning rate 0.025: the weights change by [+0.025,-0.015,-0.010]. Multiplying by h=2 produces exactly the logit changes [+0.05,-0.03,-0.02] above.

The same update behaves differently on another example with a different hidden activation. If h=0, these weights have no direct effect on its logits. If h=-2, the same weight change has the opposite effect. That is the basic bridge to feature learning: **the input-dependent activation determines which examples a parameter update affects and in what direction**.

Real VEGAM shares its decoder across moves rather than assigning an independent trainable weight to each named move. Bg7 and Bf4 have different hidden move vectors produced from source, destination and context. The policy-output weight vector is common. A gradient that favors a feature active for Bg7 can also favor that feature in a different move on a different board. Generalization depends on how those hidden features organize examples.

## 9.4 The chain rule carries error backward through products

If a loss L depends on y, and y depends on w, then dL/dw=(dL/dy)(dy/dw). For example, y=2w and L=(y-3)^2 give dL/dw=2(y-3) times 2. At w=1, y=2, so the derivative is -4: increasing w reduces the error locally.

In a network, a variable often affects many later variables. Add the contributions from all paths. For a linear layer y_j=sum_i W_ji x_i+b_j, an upstream derivative g_j=dL/dy_j gives dL/dW_ji=g_j x_i, dL/db_j=g_j and dL/dx_i=sum_j W_ji g_j. These expressions explain why the same weights that route information forward route error information backward with a transpose.

For GELU, the derivative is Phi(x)+x phi(x), where phi is the standard normal density. At x=0 it is 0.5; at x=1 it is about 1.0833. Nonlinearities change how much gradient passes at a given activation. For a residual sum y=x+F(x), the derivative includes an identity route plus the derivative through F. That direct route is one reason residual networks are easier to optimize than forcing every signal through a long chain of transformations alone.

## 9.5 Backpropagate through our attention row

We can compute a local attention derivative with no neural mysticism. Let J be the **first coordinate of the toy g7 head output**, so J=sum_j a_j v_j,0. The value first coordinates are [1,-1,0]. Therefore dJ/da=[1,-1,0]. Recall a=[0.391954,0.174869,0.433176] and J=0.217085.

The softmax derivative is da_i/dl_j=a_i(delta_ij-a_j), where delta_ij is 1 when i=j and 0 otherwise. Combining the terms gives dJ/dl_j=a_j(v_j,0-J). The three results are **[+0.306867,-0.212831,-0.094036]**. Increasing the h6 logit slightly would reduce J here, even though h6 has the largest attention weight: its transmitted first coordinate is zero, below the current mixture's 0.217085.

@equation attention_gradient

This is a particularly useful antidote to reading attention as importance. The relevant derivative depends on the supplier's **value relative to the current mixture**, not merely on the attention coefficient. For a general vector output and scalar loss, replace the simple first-coordinate reading by the downstream gradient's dot product with each value vector.

Content logits are q dot k_j / sqrt(2). Thus dJ/dq=sum_j (dJ/dl_j) k_j / sqrt(2), yielding approximately **[0.150494,-0.066494]**. The key gradients are [0.216988,0] for g7, [-0.150494,0] for h7 and [-0.066494,0] for h6. These numbers have been checked against finite differences; the maximum discrepancy in that check is below 4 x 10^-12.

If a downstream loss wanted to increase J, it would carry a negative derivative with respect to J, reversing the gradient-descent directions just described. In the toy row z_g7=[1,0], the relevant first-row W_Q gradient follows directly from dJ/dq. In real training, z itself depends on earlier projections, attention and normalization, so the gradient continues backward into those computations as well.

## 9.6 How static and adaptive geometry learn

A fixed attention bias is directly added to its logit, so the derivative with respect to S_ij equals the derivative with respect to that logit. It is updated repeatedly across positions using that square pair. If the same positional routing tendency helps many examples, their gradients can reinforce it. If it helps only occasionally or hurts others, gradients can oppose it.

For adaptive D_ij=sum_r L_ir g_r R_jr, the derivative with respect to g_r is sum_ij (dL/dD_ij)L_ir R_jr. The derivatives for L and R similarly multiply by the other factor and the gate. The gate derivative then passes through tanh, the gate MLP, the compressor and normalized square states. This is the full mechanism by which a position-dependent routing adjustment can be shaped by move and value prediction error.

A gate can become useful for a broad context without being labeled "endgame" or "trapped rook." To verify such a label later, we need counterexamples and held-out tests. Optimization supplies numerical usefulness; interpretation is a separate scientific task.

@pagebreak

## 9.7 Repetition across examples creates stable relationships

Imagine two otherwise comparable training situations. In one, a pawn supports the bishop's destination and the teacher strongly favors the move. In the other, the pawn is absent and the teacher rejects it. A model that makes the same prediction in both suffers systematic error. Parameters that make its representation sensitive to the supporting-pawn difference can lower the combined loss, provided the rest of the computation uses that signal appropriately.

That is the logical route from data to chess-relevant structure: repeated input differences correlate with target differences; the current model's errors generate gradients; shared parameters are adjusted to exploit useful distinctions; those distinctions affect new inputs through the same computation. It is neither a guarantee of a clean concept nor mere memorization by definition. A large enough model can also exploit irrelevant correlations or overfit particular configurations.

A "learned defense feature" would be an operational claim: its activation or a direction in representation space reliably distinguishes defended from undefended pieces under controls, generalizes to held-out boards, and contributes causally to relevant predictions when manipulated. Training makes such features plausible. It does not certify their names.

@check 13|The model predicts 0.2 for a move with target 0.7. What is the loss derivative with respect to its logit, and which way does gradient descent move the logit?

@check 14|In the toy attention example, why is dJ/dl_h6 negative even though h6 receives the largest attention weight?

# 10. From updates to your deployed weights
@objectives Understand AdamW, clipping and EMA; reconcile the old document with the completed run; distinguish accepted exposures from unique positions; separate prediction improvement from engine strength.

A single gradient step is easy to inspect. The actual run accumulates information across hundreds of millions of examples while controlling step sizes and preserving resumable state. Your current model is the result of that particular process, not a generic transformer with arbitrary dimensions. [T1]

## 10.1 AdamW keeps running statistics of gradients

Plain gradient descent uses the current gradient directly. AdamW maintains a moving average of gradients m and squared gradients v. It corrects their initial bias toward zero, divides the corrected first moment by the square root of the corrected second moment plus epsilon, and applies decoupled weight decay. [S5; R4]

@equation adam

Here g_t is the current gradient; beta_1 and beta_2 control the smoothing; m_hat and v_hat are bias-corrected estimates; eta is the learning rate; lambda is weight decay. The decay term shrinks the parameter independently of the adaptive gradient division. It is not simply equivalent to adding an L2 penalty inside Adam's gradient under all conditions.

A one-parameter first-step example makes the units concrete. Let w=0.5, g=-0.2, beta_1=0.9 and beta_2=0.999. Starting moments are zero, so m=-0.02 and v=0.00004. Bias correction gives m_hat=-0.2 and v_hat=0.04. Ignoring the tiny epsilon, the normalized update is -1. With eta=0.001 and lambda=0.01, decay reduces w by 0.000005 and the gradient step increases it by 0.001, giving **0.500995**. This illustrates the rule; the real configured peak learning rate is 0.0003.

The cloud trainer clips the total gradient norm to at most 1 before a successful optimizer update. If a gradient vector [3,4] has norm 5, clipping to 1 scales it to [0.6,0.8]. This changes magnitude while preserving direction. It limits unusually large updates; it does not prove the direction is correct or eliminate all instability.

## 10.2 A learning-rate schedule and an averaged model

The source uses a warmup factor min(1,(step+1)/1000), multiplied by a cosine schedule based on accepted position exposures. The peak is 3 x 10^-4 and the final schedule approaches 10% of that peak. Warmup counts optimizer steps; cosine progress counts accepted exposures relative to the requested total. [S5]

The EMA is an exponential moving average of model weights: ema_new=decay times ema_old + (1-decay) times current_weights, with decay=min(0.999,(1+step)/(10+step)). Once decay reaches 0.999, each new update contributes 0.1% immediately, with older contributions fading geometrically. For an old averaged weight 2 and current weight 3, the next value is 2.001. It is an average in parameter space, not an ensemble that performs thousands of separate forward passes.

The exported model uses the selected checkpoint's EMA. It does not perform training while you play a GUI game. Your new Catalan game does not automatically update these weights. Search statistics may change and a tree may be reused, but the neural parameters stay fixed unless a separate training/export process changes them.

## 10.3 What "three passes" means here

The training partition contains 244,312,890 position records. The completed main run records **732,938,670 accepted exposures**, exactly three passes over that partition, and **715,761 optimizer steps**. A separate five-million-position pilot supplied initial model weights with a fresh optimizer for the main run. Counting both gives 737,938,670 recorded exposures in this lineage; it does not give that many distinct chess positions. [T1]

Games were split as groups rather than scattering consecutive positions from one game across train and validation. That reduces a serious leakage route. It does not ensure that common openings or transpositions never overlap across different games. The same opening can recur many times, and three passes deliberately reuse the same partition. "Exposures," "records," "games" and "unique board positions" are different counts.

The final receipt says latest-checkpoint EMA was selected, with zero partial-pass positions and zero recorded hard-example replay exposures. It was not selected by winning a giant checkpoint tournament. Recorded migrations preserved training state while changing execution, including a move from distributed to single-GPU execution at the same global batch; they do not claim bit-identical numerical continuation across hardware paths.

## 10.4 What changed since VEGAM8.md?

@table revision

The old document describes the earlier one-pass state and warns that the cosine schedule had not reached its end. The current receipt confirms the later passes completed. The old network mechanics remain recognizable: 120 channels, square transformer, adaptive bias and joint legal-action decoder. But runtime packaging, TensorRT execution, search fixes and the daily Q-enabled launcher have their own later provenance.

The neural graph and the search graph must not be conflated. A DAG search can share evaluation-related information across transpositions without changing the eight neural blocks. Classic search can reuse a retained subtree without becoming the DAG preview. Changing either search implementation does not automatically retrain the model or change its policy head.

## 10.5 Did the extra training improve prediction?

A saved paired comparison evaluated epoch-1 and epoch-3 models on the same 32,768 validation records from 308 games. Policy CE fell from 1.912619 to 1.860015; policy KL fell from 0.376269 to 0.323665; teacher top-move agreement rose from 53.467% to 56.467%. State-WDL CE fell from 0.542700 to 0.532218. The comparison reports a paired whole-game bootstrap for uncertainty. [T2]

This supports improved prediction on that sampled validation distribution. It does not translate directly into Elo, establish the independent contribution of epoch 2 versus epoch 3, or prove the architecture reached its limit. Rare tactical errors, time management and search interaction can dominate games despite small average loss changes. The measured improvement is real; the scope of the conclusion must remain real too.

## 10.6 Why useful structure can be distributed

A hidden feature may be represented by a direction spanning many coordinates rather than by one neuron. Several features may overlap in the same coordinates, especially when they rarely need to be active together. This possibility is often called superposition. It explains why inspecting one neuron can reveal a mixed bag of high-activation positions rather than a clean concept. It does not establish that every confusing neuron is a meaningful superposition. [R5]

Parameter sharing encourages reusable computations: the same MLP processes each square and the same move decoder scores every candidate. Learned square embeddings and pair tables also permit location-specific behavior. The architecture balances reuse with specificity; it is not perfectly translation-invariant or a pure lookup table. Generalization must be assessed on inputs withheld by game and by relevant structural relationships, not inferred from parameter count alone.

@check 15|Does switching off action-Q use in search erase the action loss's influence on the trained transformer? What experiment would change that training influence?

@check 16|Why is 732,938,670 an exposure count rather than evidence that the model saw that many unique positions?
