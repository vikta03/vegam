# Appendix A. The implementation contract
@objectives Locate every axis and parameter family; distinguish mathematical equations from runtime execution choices; preserve the precise model identity.

## A.1 Tensor ledger for one forward pass

@table tensors

The leading batch axis B is carried through the computation. It is omitted from several figures for readability. In attention, the two 64 axes are not interchangeable: the first is the receiver and the second the supplier. In the move decoder, 256 is a maximum padded candidate count, not the number of legal moves in every position.

The combined QKV layer returns B x 64 x 1152. The code views 1152 as 3 x 12 x 32 and permutes to 3 x B x 12 x 64 x 32, then separates the leading three-way axis into Q, K and V. This reshape changes indexing, not the learned numbers. After attention, B x 12 x 64 x 32 is transposed to B x 64 x 12 x 32 and concatenated as B x 64 x 384.

The main MLP expands 384 to 576 and returns to 384. Its two biases are learned. QKV and the attention output projection have no biases in this implementation. LayerNorm has learned coordinate scales and offsets. The static attention biases, position embeddings, adaptive factors and gate-network weights are also trained parameters.

## A.2 One block as executable mathematics

Read this sequence with the ledger rather than memorizing a generic transformer diagram. First z=LN1(x). Compute q,k,v from z. Build fixed bias S and adaptive D from the same z. Compute A=softmax(qk^T/sqrt(32)+S+D) across suppliers. Form twelve Y=AV results, concatenate them and apply W_O. Add that attention update to x. Normalize the result with LN2, pass it through Linear-GELU-Linear and add the MLP update. Repeat using the next block's distinct weights. Finally apply the network's final LayerNorm and output heads. [S1]

@figure architecture|CODE. The deployed neural computation and its two board readout routes. Shapes omit the batch axis. The search tree lies outside this forward graph; it asks for repeated evaluations of different encoded positions.

There is no positional rotation embedding, no text-token vocabulary, no causal language mask and no convolutional trunk in this graph. These absences matter because importing a familiar transformer explanation can silently add operations your model never performs. The learned square table and the two attention-bias mechanisms supply this architecture's position handling.

## A.3 Parameter count and computational scale

The trusted EMA state contains **9,800,012 scalar parameters**. At two bytes per float16 parameter, their raw storage is about 19.6 MB in decimal units, before graph metadata and runtime structures. The deployed ONNX and TensorRT artifacts have separate hashes because graph transformations and execution packaging are distinct files, not because every file represents a newly trained net.

For one block, QKV alone costs approximately 64 x 384 x 1152 = 28.31 million multiply-accumulates. The output projection costs about 9.44 million. The two main-MLP linear layers together cost about 28.31 million. QK^T and AV together cost about 6.29 million. The adaptive and gate path adds roughly another million, excluding activation and normalization work. These are algebraic operation counts, not measured latency.

Batching, memory transfers, kernel fusion, TensorRT tactics, graph capture, CPU search overhead, caching and retained visits all affect wall time. A theoretical MAC count cannot predict the GUI's NPS by itself. Nor does an activation dump have the same overhead as normal play. The textbook's saved diagnostics are selected for observability, not claimed as peak-throughput runs.

## A.4 Score conventions and indexing conventions

Head and block names in prose use one-based numbering: block 1, head 6. Saved array keys use zero-based block indices and array indices, so that pair is `block.0` and head index 5. Neuron/unit identifiers such as unit 360 or decoder unit 34 remain zero-based. Square indices are zero-based with a1=0 in the current side's orientation.

Policy is normalized across legal candidate moves. Attention is normalized across supplying squares. WDL is normalized across three outcome classes. State and action values use current-side perspective at the neural interface; search handles alternating player perspectives during backup. Reported centipawn-like scores undergo a later conversion. Always name the axis and perspective when interpreting a number.

# Appendix B. Arithmetic ledger and additional bridges
@objectives Reproduce the small attention block; inspect numerical assumptions; understand the difference between a vector magnitude, direction and contribution.

## B.1 The full three-square ledger

All values in this subsection are **ILLUSTRATIVE**. Rows are g7, h7 and h6, in that order. The choice of chess labels does not make these real activations. Chapters 4-7 explain each operation; the ledger makes the intermediate results available together.

@table toy_full

The output for g7 after both residual additions is [1.582284,3.257416]. Values printed to six decimals should be treated as rounded; the companion script performs the calculation before rounding. The normalization in the MLP branch uses epsilon=10^-5. The first projection input Z is supplied directly as a teaching state.

## B.2 A square-pair score is not an attack calculation

To decide whether a bishop attacks h6 from g7 under chess rules, inspect the diagonal displacement and whether an intervening square is occupied. Here the squares are adjacent, so there is no intervening square. A general bishop ray such as c1-to-h6 needs the emptiness of d2, e3, f4 and g5. One raw pair of local piece features cannot encode all those blockers unless some contextual route has already carried the information.

Attention gives routes by which blockers and endpoints can influence each other; MLPs and later layers can transform the gathered information. The adaptive pathway also reads every square's compressed representation. Those operations make relational computations possible. The source does not contain a named bishop-ray detector in the learned trunk, and this book does not claim to have isolated one.

A useful engineered toy detector could sum blocker indicators and apply a nonlinear threshold. That would be an explicit constructive demonstration that neural arithmetic can represent a rule. It would not show that the trained weights use that exact implementation. The distinction between representational capacity and discovered mechanism is central to interpretability.

## B.3 Low rank through a two-pattern example

An outer product uv^T makes every row a scalar multiple of the same vector v. Its rank is at most one. Suppose u=[1,2] and v=[3,-1,0]. The product has rows [3,-1,0] and [6,-2,0]. There are six entries but only one independent row direction.

Add another pattern with a different supplier vector, such as [0,1,1], and the result can have rank two. VEGAM uses sixteen such patterns per head, with a different signed gate on each. The factors remain fixed during a forward pass, but the gates depend on the position. Thus a small vector of coefficients can change a large relation matrix in structured ways.

A gate value zero removes its pattern's contribution for that input. A negative value reverses the pattern. Neither statement says the corresponding concept is absent or negated in a human sense. A pattern is a numerical outer product until tests establish a useful interpretation.

## B.4 Why shared additions can matter after GELU

Consider two pre-activations a=-1 and b=1, and a common offset g. With g=0, GELU(a)-GELU(b) is approximately -0.158655-0.841345=-1. With g=1, the difference is GELU(0)-GELU(2)=0-1.954500=-1.954500. The common offset did not cancel after the nonlinearity.

This is directly relevant to Bg7 and Bf4. They share the same decoder context, source and no-promotion/no-castling embeddings, but have different destinations and interactions. Their combined coordinates enter GELU before the hidden layer. A global signal can therefore change the relative usefulness of destination-specific features without itself being different between the two candidate rows.

## B.5 Magnitudes, directions and cancellations

The L2 norm of v=[3,4] is 5. The norm of -v is also 5, although the directions oppose. If one message is [3,4] and another is [-3,-4], both have large norm but their sum is zero. A message-norm heatmap discards this cancellation information.

An output weight w=[1,0] reads only the first coordinate: w dot [3,4]=3. A message [0,100] has norm 100 yet contributes zero to that particular linear readout. A smaller [1,0] contributes one. Earlier in a nonlinear network, even that fixed-readout argument is incomplete because later transformations depend on the state. This is why intervention results and readout decompositions carry different evidential weight.

## B.6 A numerical summary of the measured anchor

@table anchor_legal

The legal-move table uses the CPU diagnostic run described in M2. Tiny probability differences from GPU recordings are expected and reported. Rows are sorted by policy rather than by the native candidate-slot order. These predictions are not claims that every low-policy move loses or that action WDL is calibrated on this particular endgame.

# Appendix C. Answers to the understanding checks

## C.1 Boards, vectors and attention

**1.** The rook has three legal moves: Rh8, Rxg7 and Rxh6. Each allows White to capture the rook immediately by the listed reply. Calling them losing moves in an absolute game-theoretic sense would require additional analysis; the local capture claim is exact.

**2.** No. The learned square embedding can differ between g7 and f4 even if their 120 input features match. Later pairwise biases and interactions with other squares can differentiate them further.

**3.** A FEN omits the full preceding sequence. The network receives up to eight piece-history frames and repetition information derived from history. The same visible board and halfmove count can therefore have different encoded inputs.

**4.** Increasing the third input by one adds the third column of the stored weight matrix: [+0.5,+1,-1]. Biases do not change. This is a precise local effect before nonlinear processing.

**5.** It subtracts the mean of g7's own 384 coordinates. It does not normalize g7 against other squares or other batch positions. Each square gets a separately normalized row.

**6.** q_h7=[1,1] and k_h6=[1,2], so their dot product is 3. q_h6=[2,1] and k_h7=[0,1] give 1. Different Q/K projections permit directed content compatibility.

**7.** Adding 5 to every logit leaves softmax unchanged because the common exponential factor cancels. Adding 5 only to h6 multiplies its unnormalized weight by exp(5), increasing its share and decreasing the other shares.

**8.** The entire rank-one correction reverses sign. The g7 row changes from [0.1,-0.2,0.3] to [-0.1,0.2,-0.3]. Fixed and content terms remain unchanged in this direct gate substitution, so the final attention does not simply become its negative.

**9.** A zero value vector contributes zero directly to AV. Its logit still competes in softmax: giving it more weight can reduce the contributions from nonzero suppliers.

**10.** GELU is nonlinear, so composing Linear-GELU-Linear generally cannot be reduced to one linear map. Which combinations pass strongly depends on the current input values.

## C.2 Learning and output

**11.** No. The validity mask makes an unlabeled candidate contribute zero to the action-WDL loss. A draw-shaped placeholder is a storage convenience, not evidence that the action draws.

**12.** A soft target spreads mass. Even a perfect prediction assigns probabilities below one to its supported events, so their negative log probabilities remain positive. The minimum cross-entropy is the target entropy.

**13.** The derivative is p-y=-0.5. Gradient descent subtracts the derivative, increasing that logit. Other logits also change, and the probabilities must be recomputed by softmax.

**14.** For the first output coordinate, h6 supplies zero while the current weighted mean is positive. Raising its weight dilutes that positive mean. Importance depends on transmitted values and the chosen output question.

**15.** No. The trained shared weights still reflect the historical action loss. To remove that training influence, train a controlled alternative without the action objective, with otherwise matched data and optimization, then compare appropriately.

**16.** Three passes revisit the same training partition, and repeated openings can occur across many records and games. Accepted exposures count training examples consumed, not distinct positions discovered.

**17.** Its policy weight is negative. Bg7's lower activation multiplied by that negative weight makes a positive contribution to Bg7's logit advantage over Bf4.

**18.** Their logit difference and probability ratio stay fixed. Their individual shares can change relative to the other candidates because the softmax denominator changes.

## C.3 Search and evidence

**19.** No. The code uses action advice for cold edges; after a completed child visit, the searched value takes over. The advice can still have affected which lines received those visits.

**20.** Search budget, retained tree, explored continuations and backed-up values can differ. The display conversion can amplify small changes near extreme values. Identical model weights do not imply identical search statistics.

**21.** It establishes predictive decodability under the stated dataset, split and probe. It does not show that the engine uses that readable direction causally. Leakage, confounds and probe capacity must also be checked.

**22.** It is one exploratory position and a broad removal. The effect could reflect general context, redundant features or multiple mechanisms. A head's direct attention pattern also need not resemble the named chess concept. Generalization and controlled mediation tests are required.

**23.** Otherwise the label "few immediately safe rook moves" can be mistaken for "strategically bad rook." Sacrifice controls expose whether the representation tracks geometry, material consequences or actual usefulness.

**24.** State WDL predicts an outcome distribution, while policy allocates preference among legal moves. Many moves may preserve a winning game, and the policy target need not collapse onto a single move. High state win confidence does not force a 100% policy share.

# Appendix D. A working glossary

**Activation.** A value computed from the current input, as opposed to a parameter retained between positions. A neuron's activation changes even when its weights do not.

**Adaptive bias.** A position-dependent addition to attention logits, formed in VEGAM from fixed learned left/right factors and input-dependent signed gates.

**Affine map.** A linear transformation plus an offset. Linear layers with biases are affine in their input.

**Attention.** Input-dependent weighted mixing of value vectors. Distinguish logits, normalized weights and transmitted messages.

**Batch.** Several independent positions processed together. B is the leading batch dimension in the shape ledger.

**Bias.** An additive learned quantity. Input-layer bias, square embedding and attention pair bias are different objects despite all being added somewhere.

**Bilinear.** Linear in either of two arguments when the other is held fixed. A learned Q/K dot product is bilinear in its two normalized input vectors.

**Calibration.** Here, comparison between instrumented and native outputs on specified inputs. Probability calibration against empirical outcomes is a different evaluation task.

**Causal intervention.** A specified alteration to an internal variable followed by recomputation of downstream behavior. Its conclusion is limited by intervention scope and controls.

**Channel / coordinate.** One axis entry of a representation. An input channel can have an explicit piece meaning; a hidden coordinate need not.

**Checkpoint.** A saved training state that can include weights, optimizer moments, counters and data-order state. An inference export is a different artifact.

**Cold edge.** A search move with no completed child visit. It needs a fallback or action estimate before backed-up search evidence exists.

**Context.** The 128-entry global representation read from the mean of final square states, and its projected 64-entry contribution to move decoding. Specify which one is meant.

**Cross-entropy.** Negative target-weighted log predicted probability. With a soft target, its minimum is generally nonzero.

**DAG.** Directed acyclic graph. In this project, an optional search implementation with transposition sharing; it is not the neural architecture's eight-block structure.

**Decoder.** The learned function that combines move endpoints and context into policy and action-WDL logits. It does not generate legal moves by itself.

**Dot product.** Sum of coordinate-wise products. It combines alignment and vector length, not just angular similarity.

**EMA.** Exponential moving average of parameters. Your exported model uses the selected checkpoint's averaged weights.

**Embedding.** A learned numerical representation. Here, the input projection plus a learned per-square table creates the initial hidden rows.

**Entropy.** A distribution's expected negative log probability under itself. It measures spread in a precise probabilistic sense, not human confusion.

**Epoch / pass.** One traversal of the defined training partition. It need not contain unique board arrangements.

**Feature.** A measurable input property or a learned internal pattern. Always specify whether its meaning is defined by the encoder or inferred by investigation.

**FEN.** A representation of current chess placement, side, rights and counters. It does not contain the complete move history.

**First-play urgency.** Search's fallback value for unvisited moves. Q-on modifies this cold-edge estimate using action predictions.

**GELU.** Smooth activation x Phi(x). It retains signed numerical responses and supplies nonlinearity in the MLPs and decoder.

**Gradient.** A collection of local partial derivatives telling how small changes affect a scalar objective.

**Head.** Ambiguous term: an attention head routes 32-dimensional messages; an output head predicts policy, WDL or duration. They are different modules.

**History frame.** One of eight ordered board snapshots represented by thirteen input channels. History presence and repetition add distinct information.

**Intervention control.** A comparison such as restoration, a matched unrelated component or a donor choice designed to rule out an alternative explanation.

**Key / query / value.** Separate learned projections used for routing comparison and message content. They are vectors, not literal questions or answers.

**LayerNorm.** Per-square normalization over hidden coordinates, followed by learned scales and offsets.

**Logit.** A real-valued pre-softmax score. Differences determine probability ratios; a common additive offset does not.

**Low rank.** A restriction on the number of independent directions needed to express a matrix. Rank sixteen does not mean sixteen chess concepts.

**Mask.** A selector that excludes padding or invalid labels from a computation. A masked missing label is not a negative label.

**MLP.** Multilayer perceptron: learned linear maps separated by nonlinearities. VEGAM's main square MLP is 384-to-576-to-384.

**NPS.** Reported search nodes or visits per second under a particular implementation and measurement interval. It is not automatically unique neural evaluations per second.

**Parameter.** A learned number held fixed during ordinary inference. Activations are recomputed from each input using those parameters.

**Policy.** A distribution over candidate moves. It is neither an action's win probability nor the final searched move by definition.

**Probe.** A separate predictor trained on frozen representations to test whether a label is readable from them.

**PUCT.** A family of tree-selection rules combining value estimates with prior-weighted exploration.

**Residual stream.** The evolving 384-entry state per square, updated by adding attention and MLP outputs.

**Softmax.** Exponentiate scores and divide by their sum on a specified axis. It produces a distribution with coupled entries.

**Superposition.** A possible organization in which several features share representational coordinates. It requires investigation in the model at hand.

**Tensor.** An array with named dimensions. Its shape is essential to knowing what each operation combines.

**WDL.** Win, draw and loss probabilities from a specified perspective. q=W-L and expected score=W+D/2 are derived but different quantities.
