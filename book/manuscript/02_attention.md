# 4. Why a square needs queries, keys and values
@objectives Derive a mechanism for combining squares; calculate Q, K and V; understand learned compatibility as a bilinear function rather than a literal question; connect compatibility to training.

Suppose we stop after the input projection. g7 has no local piece in any of the eight frames. Its row cannot distinguish our anchor board from another board with the same g7 history, same side flag and same clock, but no pawn on h6 and a different rook location. Yet those differences could completely change the desirability of Bg7. Something must allow information from other square rows to influence g7.

## 4.1 Several possible ways to combine the board

We could average all square vectors and copy that average to each square. This is cheap and gives every square global information. But every receiver gets the same mixture: g7 cannot separately emphasize a neighbor while a3 emphasizes a queenside structure.

We could give each ordered square pair a fixed learned coefficient. This allows different routes into different squares, but it cannot change the route when a bishop replaces a knight or a blocking pawn moves. We could instead concatenate the entire board and apply a large unrestricted MLP. That can represent complex relationships, but square-specific processing and sharing become less structured and parameter costs change substantially.

VEGAM uses an input-dependent mixing rule. For every receiving square i, it computes 64 scores, one per supplying square j. These scores depend on both current representations and learned square-pair biases. A normalization turns each score row into weights. The receiver obtains a weighted combination of vectors supplied by the other squares. This is attention, stated as a computational task rather than as a metaphor.

Self-attention includes the square itself: j can equal i. There is no causal mask like a text model's ban on future tokens. All 64 squares are known simultaneously. There is also no rule saying attention must follow legal chess moves. g7 may receive information from a1 even if no piece travels between them. The network is computing features of a position, not moving the pieces along its attention edges.

## 4.2 Three learned projections solve three different problems

From the normalized row z_i, one matrix forms q_i, another forms k_i and a third forms v_i. In a single real head, all three outputs have 32 entries. The query and key participate in routing; the value is the content to be transported. These are different learned transformations of the same source representation. [S1: Block.forward]

@equation qkv

Our row-vector convention writes W_Q, W_K and W_V as 384 x 32 for one head. PyTorch stores corresponding slices with output coordinates first, so its weight arrays are transposed relative to this notation. Q has shape 64 x 32, as do K and V. The code computes all three projections for all twelve heads in one bias-free Linear(384,1152), then reshapes and permutes the result.

What is being "queried"? Nothing linguistic. A query is 32 real numbers. Those numbers specify a direction against which key vectors will be compared. A key is another 32-number projection. A value is a 32-number message. Terms such as "what this square asks for" are shorthand for these learned numerical roles, not internal sentences about bishops or rooks.

Why not use one projection for everything? Requiring routing and transmitted content to share exactly the same coordinates would constrain them unnecessarily. A supplier may need a strong routing signature while sending a small or differently signed message. Different Q and K projections also permit directed relationships: g7's score for h7 need not equal h7's score for g7. If Q=K, the content dot-product matrix would be symmetric before the biases, although row normalization could still make the attention weights asymmetric.

@figure qkv_routes|CODE / SCHEMATIC. One receiver and one supplier shown within a 64-square computation. Q and K influence routing; V supplies the message. The supplier contributes to the softmax denominator alongside all other squares.

## 4.3 A complete small projection example

For the next four chapters, use three square labels: g7, h7 and h6. The labels keep the link to the board visible, but the following two-coordinate vectors are **illustrative teaching values**, not a reduced reading of VEGAM. We begin at the input to the Q/K/V projections, after any normalization. Each row in Z represents one square:

@table toy_qkv

The first row z_g7=[1,0]. Multiplying it by W_Q=[[1,0],[1,1]] gives [1,0]. The second row [0,1] picks out [1,1]; the third row [1,1] adds both rows of W_Q to give [2,1]. Therefore Q's rows are [1,0], [1,1], [2,1].

W_K=[[1,1],[0,1]] gives K rows [1,1], [0,1], [1,2]. W_V=[[1,2],[-1,1]] gives V rows [1,2], [-1,1], [0,3]. Notice that h7's value begins with -1 even though its key begins with 0. Routing signature and transmitted content are different computations. Negative activations are entirely ordinary; they are not negative probabilities.

This tiny Z is a chosen projection input, not the output of LayerNorm with gamma=1 and beta=0 on a two-coordinate row. That distinction avoids an easy teaching inconsistency. We will use a fully specified LayerNorm later when continuing the residual calculation.

## 4.4 What a dot product actually measures

The dot product q_i dot k_j multiplies matching coordinates and adds them. In our example q_g7=[1,0]. Its dot products with the three keys are 1 times 1 + 0 times 1 = 1; 1 times 0 + 0 times 1 = 0; and 1 times 1 + 0 times 2 = 1. The unscaled score row is [1,0,1].

Geometrically, a dot product equals the product of the two vector lengths times the cosine of the angle between them. It grows when vectors align and when their lengths grow. It is **not** pure cosine similarity, because these vectors are not forced to unit length. Nor is it a comparison of the raw piece labels. The transformations that created the vectors matter.

Substitute the projection definitions into the dot product:

@equation bilinear

The 384 x 384 matrix M=W_Q W_K^T expresses a learned bilinear compatibility function on two normalized square representations. "Bilinear" means linear in one input when the other is held fixed. Its entries can make one combination of receiving-square features interact positively or negatively with a combination of supplying-square features. For a 32-wide head, this matrix has rank at most 32; it is not an unrestricted 384 x 384 matrix.

This establishes capacity, not chess meaning. To see what a useful relationship might look like, imagine a deliberately constructed two-feature toy space: one coordinate at the receiver indicates a potential destination, and one at the supplier indicates a friendly supporting pawn. A learned M can give their product a positive coefficient. Repeated training examples can reward that route when it improves downstream move prediction. But the actual model does not receive a hand-labeled "supporting pawn" coordinate, and dot product alone cannot establish whether a bishop's ray is blocked. Other features and layers must provide the necessary context.

## 4.5 How compatibility becomes useful

A compatibility score is useful only insofar as the resulting message improves the prediction loss. If increasing the score from h6 into g7 helps raise the correct move's logit on a training example, the chain rule can send a gradient back through the output head, residual transformations, value mixture and softmax to q_g7 and k_h6, then to W_Q and W_K. Chapter 9 computes this numerically.

There is no separate supervisor saying "make g7 attend to h6." The training target concerns moves and outcomes. Attention routing is adjusted because it lies on a differentiable path to those outputs. A route can become useful in many positions without corresponding neatly to one human relation. Conversely, a visually plausible attention pattern can have little effect if its value messages are redundant or canceled downstream.

@check 6|Using the toy matrices, calculate q_h7 dot k_h6. Is it equal to q_h6 dot k_h7? What does the answer show about directionality?

# 5. Turning compatibility scores into a distribution
@objectives Derive the square-root scale under explicit assumptions; compute softmax and its competition effect; identify the supplying-square axis; distinguish attention probability from move probability.

The content scores just constructed can be any real numbers. We need coefficients for a mixture of supplier values. One simple option would be to choose the largest score and ignore everything else. Another would be to multiply values by raw signed scores. VEGAM uses softmax: it produces positive weights summing to one while retaining a smooth dependence on every score. [S1]

## 5.1 Why divide by sqrt(32)?

In a real head, the dot product adds 32 coordinate products. A useful initialization argument asks what happens as head width changes. Suppose each query and key coordinate has mean zero and variance one; suppose query and key coordinates are independent in the required pairs; and suppose the coordinate products are mutually uncorrelated. Each product then has mean zero and variance one. The variance of their sum is d, and its standard deviation is sqrt(d).

@equation scale_variance

Without scaling, wider heads would tend to produce scores with a larger spread before training has made them useful. Large score differences can push softmax close to a one-hot distribution, where many derivatives become tiny. Dividing by sqrt(d) keeps this idealized variance near one. In VEGAM d=384/12=32, so the denominator is about 5.656854. In our two-dimensional lesson it is sqrt(2), about 1.414214. [R1]

These are assumptions, not a guarantee that trained Q and K actually have independent unit-variance coordinates. They may be correlated and have learned scales. The division is a fixed stabilizing convention. Learned projections and biases can still produce sharper or flatter distributions. It is not a calibration turning the dot product into a chess probability.

For the toy g7 score row [1,0,1], scaling gives [0.707107,0,0.707107]. The full three-row scaled content matrix is shown later alongside the bias-adjusted scores. The number 0.707107 has no centipawn meaning; it is a logit contribution used to route value vectors.

## 5.2 Softmax, one multiplication at a time

For a vector of logits l_j, define p_j=exp(l_j)/sum_k exp(l_k). The exponential is always positive. Dividing by the sum makes all entries add to one. Increasing a logit increases its exponential weight relative to others. A difference of one logit means a probability ratio of e, approximately 2.718, independent of the other logits.

@equation softmax

Take logits [1,0,-1]. Their exponentials are [2.71828,1,0.36788], sum 4.08616. Division produces approximately [0.66524,0.24473,0.09003]. Raising the second logit from 0 to 1 changes the exponentials to [2.71828,2.71828,0.36788], sum 5.80444, and probabilities to [0.46831,0.46831,0.06338]. The first logit did not change, but its probability fell because the denominator changed.

This is competition, not independent confidence. A high policy probability for Bg7 is a share of probability among legal moves. A high attention weight for h6 is a share among 64 supplying squares for one receiver and one head. The two distributions live on different axes and solve different tasks. Neither is a WDL probability.

Subtracting a constant from every logit does not change the distribution, because the common exponential factor cancels. Implementations commonly subtract the maximum first for numerical stability. For [1001,1000,999], we can safely compute from [0,-1,-2] and obtain the same probabilities as [1,0,-1]. No meaning is lost by discarding a common logit offset.

## 5.3 Exactly which dimension gets normalized?

The full per-block attention logits have shape B x 12 x 64 x 64. The axes are batch, head, receiving square and supplying square. The code applies softmax on the last axis. For fixed b, h and i, the 64 entries over j sum to one. Different receivers each get their own independent distribution.

Columns do not have to sum to one. Several receiving squares can all give high weight to h6. This is not a scarce global attention budget that h6 spends once. The distinction is visible when we inspect the actual 64 x 64 matrix: each row represents a different receiver, and g7 is one highlighted row.

@figure matrix64|MEASURED. Block 1, head 6. Columns supply information; rows receive it. The marked g7 row sums to one across its 64 columns. A bright column means many receivers assign weight to that supplier, not that the supplier sends a fixed total amount.

There is no illegality mask on this matrix. Empty squares and occupied squares both participate. Empty squares can encode geometry, history, rule state and context inherited from earlier layers. The legality mask is applied later to the move-policy output, where padded candidate slots must not receive probability.

## 5.4 What softmax gains and what it costs

A convex mixture is easy to control: the attention output before the output projection lies in the convex hull of the value vectors for that head. But a convex mixture does not mean "only positive evidence." Value-vector coordinates can be negative, output weights can be negative and different heads can cancel. The positivity applies only to the routing coefficients.

The denominator also couples all supplier scores. A change to a far-away square can reduce the weight on h6 even if the local h6 score stays fixed. Therefore, one should not read a fall in attention as direct evidence that a pairwise relation disappeared. It may reflect stronger competition from elsewhere.

@check 7|If every attention logit for receiving g7 increases by 5, what happens to that row's attention weights? If only h6's logit increases by 5, what happens?

# 6. Geometry that stays fixed, geometry that changes
@objectives Distinguish square embeddings from pairwise biases; reconstruct VEGAM's rank-16 adaptive correction; understand signed gates, outer products and low rank; inspect real logits without inventing chess labels.

The same piece relationships matter differently on different squares. A bishop can travel along diagonals; a knight jumps in an L; a rook's line can be blocked. The content score can learn some of these dependencies through square embeddings and contextual representations. VEGAM also gives attention two additional sources of information: a fixed learned square-pair table and a position-dependent low-rank correction. [S1]

## 6.1 The fixed table is a prior on ordered square pairs

Each block owns a learned tensor with shape 12 x 64 x 64: one table for each head. Its entry S_hij is added to the content score from supplying square j into receiving square i. It is initialized to zero. Training can increase or decrease individual entries according to their effect on prediction loss.

This is not the same object as the input position embedding. The embedding adds 384 numbers to a square's representation. The pair table adds one number to a particular head's compatibility between two squares. The first affects what a square representation contains; the second directly affects routing between an ordered pair.

A trained table could favor diagonals, files, knight offsets or asymmetric regions. No hand-coded bishop or knight mask forces it to do so. The actual table is arbitrary over the 64 x 64 pairs. Its 49,152 entries per block buy flexibility, but do not impose translation invariance or guarantee a recognizable chess geometry. In a particular head, a seemingly odd fixed preference may interact usefully with content scores and values.

@figure static|MEASURED. The fixed bias row for receiving g7 in block 1, head 6. It is the same table row whenever this model evaluates a position in the same orientation. Color is a signed logit contribution, not a probability or an attack map.

## 6.2 Why a fixed table is not enough

The h7-to-g7 relation changes when h7 holds an opposing rook instead of a friendly pawn. Content scores already respond to the current representations. VEGAM adds a second input-dependent route that can globally adjust a set of learned relation patterns using information compressed from the whole board.

First, each normalized 384-vector is projected to eight numbers, with a learned bias. The resulting 64 x 8 tensor is flattened in square order into 512 entries. A learned 512-to-128 linear map, GELU and a 128-to-192 linear map produce 192 gate pre-activations. Tanh bounds them between -1 and +1. Reshape 192 as 12 heads times 16 rank components. There is **one set of sixteen gates per head per position**, shared across all receiving and supplying pairs in that head.

@equation gates

In this equation z is the normalized 64 x 384 board; Compress maps each row to eight entries; vec concatenates the rows without mixing their order; W_1 and W_2 describe the two gate-network maps, with their biases; GELU supplies the nonlinearity; and tanh bounds the final gates. Each block has its own compressor, gate network and relation factors.

Flattening matters. The gate network does not see an orderless average. A compressed feature at g7 occupies a different input slot from the same feature at a3. It can condition its gates on a particular arrangement. This route can therefore communicate global context even before the block's attention value mixing is finished.

## 6.3 A sum of sixteen signed relation patterns

For each head, the model stores a left matrix L of shape 64 x 16 and a right matrix R of shape 64 x 16. Both start as small random values. The adaptive score matrix is:

@equation adaptive

For one receiver i and supplier j, multiply L_ir by the gate g_r and R_jr, then sum over the sixteen components r. In matrix notation this is L diag(g) R^T. Each column pair contributes an outer product: a 64-entry receiver pattern multiplied by a 64-entry supplier pattern, producing a full 64 x 64 relation pattern. The gate scales that pattern positively, negatively or near zero for the current position.

The adaptive matrix has rank at most 16 before it is added to other terms. Rank describes how many independent row/column directions a matrix needs. It does not mean the model examines sixteen chess lines or discovers sixteen named concepts. The final logits also contain a rank-at-most-32 content matrix and a full fixed bias. Applying softmax need not preserve the rank limit.

For a fixed head, sixteen gates cannot independently set all 4,096 pair entries for every position. That is the useful restriction: the network selects and combines sixteen learned relation patterns instead of generating an unconstrained fresh matrix. L and R together contain 2,048 parameters per head, compared with 4,096 for a single full pair table; the compressor and gate generator add shared costs. Low rank is a tradeoff between flexibility, parameter structure and computation, not free expressiveness.

A negative gate is meaningful. If an outer product increases the score into g7 from h7 when its gate is positive, changing the gate's sign reverses that contribution. Gates are not probabilities and are not softmax-normalized across the sixteen components. All sixteen may be near magnitude one simultaneously.

## 6.4 Continue the small attention calculation

Return to the three-square lesson. Give it a single rank component with receiver factors L=[1,0.5,-0.5], supplier factors R=[0.2,-0.4,0.6] and gate g=0.5. The adaptive row for receiving g7 is 1 times 0.5 times R, giving [0.1,-0.2,0.3]. The h7 row is [0.05,-0.1,0.15], and the h6 row is [-0.05,0.1,-0.15].

Set the fixed g7 bias row to [0,0.2,-0.1]. Adding content [0.707107,0,0.707107], fixed bias and adaptive bias gives logits [0.807107,0,0.907107]. Subtract the maximum: [-0.1,-0.907107,0]. Exponentiate: approximately [0.904837,0.403691,1]. Their sum is approximately 2.308528. Normalize to obtain **[0.391954,0.174869,0.433176]**.

@figure toy_attention|ILLUSTRATIVE. The complete three-square calculation: scaled content, content plus both biases, then row softmax. Every number is generated by the specified matrices; none is a trained VEGAM activation.

Notice a subtle cancellation: for receiving g7 from h7, the +0.2 static bias and -0.2 adaptive bias cancel. That does not mean h7 contributes nothing. Its total logit is zero, exp(0)=1 before any common subtraction, and it still gets about 17.49% of the attention weight in this three-supplier example.

## 6.5 The same decomposition in the real model

For real block 1, head 6, receiving g7 from h7, the 32-dimensional Q/K dot product is approximately 0.626473. Divide by sqrt(32) to get **0.110746**. Add fixed bias **-0.056549** and adaptive bias **-0.079332**. The total is about **-0.025135**. Softmax across all 64 supplying squares gives h7 attention weight about **0.014630**, or 1.463%. [M2]

You cannot obtain that last number by exponentiating this one logit alone: you need the other 63 logits in the denominator. This is why the complete saved trace matters. The final weight is close to the uniform 1/64=1.5625% reference, despite this head later proving influential under removal.

@figure content|MEASURED. Content scores into g7 for block 1, head 6. Each square's color comes from a 32-product sum divided by sqrt(32). High color means a larger content contribution before biases and normalization.

@figure adaptive|MEASURED. The adaptive correction for the same receiver and head. It is a signed addition to the logits, generated from all 64 normalized square rows. It should be read on its own color scale, not compared to the other panels by color alone.

@figure gates|MEASURED. The 12 x 16 signed gates in block 1 for this position. Many are saturated near +1 or -1. Across all eight blocks, 1,161 of 1,536 gates have magnitude at least 0.99. That observation does not show that the gates are constant across positions.

A saturated gate may be implementing a nearly discrete choice on this input. It can also be insensitive to small perturbations because tanh's derivative is 1-g^2. Which interpretation is useful requires comparing many positions and interventions. A picture of gates near the extremes does not by itself show that adaptive attention is broken, essential or semantically interpretable.

@check 8|In the rank-one toy correction, what happens if the gate changes from +0.5 to -0.5? Which g7-to-supplier adaptive entries reverse sign?

# 7. What attention actually delivers
@objectives Multiply attention by values; follow output projection and residual addition; complete a small block including LayerNorm and GELU; distinguish message size from causal effect.

We have routing weights. We do not yet have an updated square representation. An attention map is often the most visually attractive part of a transformer explanation, which makes it easy to stop too early. The next multiplication is what actually transfers numerical content.

## 7.1 A weighted sum of value vectors

In our three-square lesson, the values supplied by g7, h7 and h6 are [1,2], [-1,1] and [0,3]. The receiving g7 row weights them by [0.391954,0.174869,0.433176]. Therefore the first output coordinate is 0.391954 times 1 + 0.174869 times (-1) + 0.433176 times 0 = **0.217085**. The second is 0.391954 times 2 + 0.174869 times 1 + 0.433176 times 3 = **2.258307**.

@equation transport

A has shape 64 x 64 and V has shape 64 x 32 in one real head. Their product Y has shape 64 x 32. Each row is a different mixture. The head does not transmit a square name, an attention probability or a chess sentence. It transmits a learned vector transformed by W_V.

The weights sum to one, but the message can be large or small, helpful or harmful to a particular move. In the first toy coordinate, the h7 message cancels part of g7's positive contribution. In the second coordinate all contributions are positive. The same attention coefficient can therefore matter differently for different coordinates.

## 7.2 Twelve heads, then a learned recombination

VEGAM computes twelve such mixtures independently. At each receiver, twelve 32-entry results are concatenated to restore 384 entries. A learned bias-free 384-to-384 output projection mixes them. Concatenation does not add the heads or average them; it places their coordinates side by side in a fixed order. The output projection can then combine information across heads.

In the two-coordinate lesson, let W_O=[[0.5,0],[0.25,1]]. Multiplying g7's Y=[0.217085,2.258307] gives an attention update [0.673119,2.258307]. The first coordinate is 0.217085 times 0.5 + 2.258307 times 0.25. The second simply keeps the second component under this illustrative matrix.

The residual connection adds the update to the old g7 vector [1,0], giving **[1.673119,2.258307]**. Residual addition preserves an explicit route for the old representation while allowing the block to modify it. It does not preserve every meaning or guarantee that no information is lost: later normalization and nonlinear processing still matter. But the block need not rebuild the entire useful representation from scratch to make a small improvement.

@equation residual

Here X is the old board state; Z=LN_1(X) supplies attention; Update is the projected multihead result; X' is the state after the first addition. A second normalized read, LN_2(X'), feeds the MLP. Its result is added to X' to form X_next. This is a pre-normalized block: normalization is inside each update branch, not applied as the only state carried forward.

## 7.3 The square-wise MLP creates nonlinear combinations

Attention moves and mixes information across squares. The feed-forward network then processes each square's accumulated vector independently with shared weights: Linear(384,576), GELU, Linear(576,384). It is the same MLP on every square, but the inputs differ by square and position. Since attention has already introduced context, "square-wise" does not mean "ignorant of the rest of the board."

A purely linear pair of layers would collapse into one matrix multiplication. The nonlinearity makes it possible for the response to one feature to depend on the levels of others. For example, a learned combination could become large only when a receiver has accumulated several mutually supporting signals. Calling it a conjunction is a possible functional description, not a guaranteed one-neuron chess rule.

GELU is x times Phi(x), where Phi is the cumulative distribution function of a standard normal variable. Equivalently it is x/2 times [1+erf(x/sqrt(2))]. It smoothly reduces negative inputs while retaining most large positive inputs. At -1 it is approximately -0.158655; at 0 it is 0; at 1 it is 0.841345; at 2 it is 1.954500. Negative output is permitted. [S1; R3]

@figure nonlinear|MATHEMATICS. GELU compared with ReLU, and the tanh function used for adaptive gates. GELU shapes hidden-feature responses; tanh supplies bounded signed gates. They play different roles.

## 7.4 Finish the toy block, without skipping arithmetic

Our post-attention g7 vector is [1.673119,2.258307]. Its mean is about 1.965713. The deviations are approximately [-0.292594,+0.292594], with variance about 0.085611. With epsilon=0.00001, gamma=[1,1] and beta=[0,0], LayerNorm yields approximately [-0.999942,+0.999942].

For the illustrative MLP, choose W_1 rows [1,-1,0.5] and [0.5,1,-1], bias [0.1,0,-0.1]. The three pre-activations become **[-0.399971,1.999883,-1.599912]**. For instance, the first is (-0.999942) times 1 + 0.999942 times 0.5 + 0.1 = -0.399971.

GELU gives **[-0.137826,1.954373,-0.087690]**. Let the three rows of W_2 be [0.5,0], [0,0.5], [0.25,-0.25], with zero output bias. The MLP update is [-0.137826 times 0.5 - 0.087690 times 0.25, 1.954373 times 0.5 + 0.087690 times 0.25] = **[-0.090835,0.999109]**.

Add that update to the post-attention residual. The final toy g7 vector is **[1.582284,3.257416]**. We have now followed one receiver through Q/K/V projections, scaled content, fixed and adaptive bias, softmax, value transport, output projection, residual addition, normalization, nonlinear MLP and the second residual addition. Every number is generated by the accompanying arithmetic script. [A1]

## 7.5 What changes across the eight real blocks?

The real vector stays 384-wide throughout the residual stream. Each block owns separate weights; block 8 does not repeat block 1's trained function. Its inputs have also been changed by all earlier blocks. Later compatibility can therefore operate on relations already represented by earlier processing.

@figure g7_layers|MEASURED. The first 32 coordinates of g7's 384-entry residual vector after each block. The colors show signed activations. They do not measure "understanding" or a probability of a chess outcome.

@figure norms|MEASURED. L2 norms of e5, g7, h7 and h6 through the residual stream. The norm is the square root of the sum of squared coordinates. The growing magnitudes show a numerical change, not increasing certainty. A final LayerNorm follows block 8.

Eight blocks are not eight plies of search. A block does not call the legal move generator, apply every move or back up a game tree. It may learn transformations useful for predicting future consequences, but measuring an internal look-ahead mechanism requires evidence beyond counting layers.

## 7.6 Reading message maps honestly

For the real selected head, the largest attention weights into g7 occur at h8, f8, c4, d5, a4 and c2. The largest **projected message norms** instead occur at c4, a4, e4, c6, d5 and h5. Why? A message includes both its weight and its value vector, then the head's slice of the output projection. A moderately weighted large value can produce a larger update than a highly weighted small value.

@figure attention_message|MEASURED. Same receiver and head, two different quantities. Left: attention weight A_g7,j. Right: the norm of A_g7,j V_j after the head's output-projection slice. Message norms are nonnegative magnitudes; they discard direction and cannot reveal cancellation between messages.

Even the message norm is not a causal attribution to Bg7. A large message can point in a direction later removed or canceled; a small message can cross an important nonlinear threshold. To measure an effect on the move preference, intervene and recompute downstream operations. Chapter 14 does exactly that.

@check 9|If a supplier gets attention weight 0.9 but its value vector is zero, what does it directly contribute to the weighted sum? Can its score still affect other suppliers?

@check 10|Why does the second linear layer of an MLP not make the first layer redundant when GELU lies between them?
