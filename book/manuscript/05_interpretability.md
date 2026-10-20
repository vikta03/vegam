# 13. What we can actually see inside VEGAM
@objectives Read activations and heatmaps quantitatively; compare MLP, CNN and transformer inspection; distinguish observation, prediction and intervention; understand the limits of probes and sparse features.

You asked whether we could cut the net open and see what it is thinking. We can open a great deal of the computation: every encoded input, linear weight, activation, attention component and output. The challenge is not access to numbers. It is deciding which statements those numbers justify.

A mechanistic explanation can be numerically exact without translating every hidden vector into human chess vocabulary. Conversely, a compelling chess story can fit a heatmap without explaining the computation. This chapter gives a disciplined way to move from the first toward a tested version of the second.

## 13.1 MLP neurons, CNN kernels and transformer signals

In a simple MLP, a hidden neuron computes an affine function of its inputs followed by a nonlinearity. We can inspect which examples make it large, its incoming weights and its outgoing contribution. A high-activation gallery is descriptive: it tells us where a number is high. It does not automatically isolate the feature that caused that response.

In a CNN, a kernel is shared across spatial locations, and activations form spatial maps. This makes local patterns especially natural to inspect. But a deep CNN activation can also depend on a large receptive field and multiple interacting features. A visually appealing kernel is not a complete explanation of the classifier.

In VEGAM, the main MLP hidden unit is also shared across locations. Block 1 has 576 hidden channels applied to each of 64 square rows. One unit therefore has a 64-square activation map for one board. Across eight blocks there are 4,608 such main-MLP units, with different weights in each block. Separate gate-MLP units, compressed channels, context units and move-decoder units should not all be called the same kind of neuron.

Attention adds a relation map: for each head and receiver, a distribution over suppliers. Adaptive gates add position-level coefficients for learned relation patterns. The residual stream carries a vector per square between sublayers. The decoder adds move-specific activations. These are several distinct things to look at; no one picture replaces the others.

## 13.2 What a neuron map displays

Our selected main-MLP example is block 1, unit 360, using zero-based unit indexing. It was chosen because it had the largest activation at g7 among that layer's 576 units in this recording. Its g7 activation is about **2.4653**, but it is larger at c7, e7, b8 and several other squares. The maximum over the board is about **2.7098** at c7. [M2]

@figure neuron|MEASURED. Main-MLP block 1, unit 360. Each square's value is GELU applied to one learned affine combination of that square's normalized, post-attention 384-vector. High activation appears on several empty squares as well as near the anchor motif.

The number is produced by a specific weight row and bias, then GELU. It is not a count of threats, an attention weight or a probability. Its being high at g7 does not identify it as a bishop-arrival or trapped-rook neuron. Looking only at the target square would hide the broader activation pattern.

To inspect it properly, collect the same unit over many positions, retrieve high and low examples, control for phase and square, and compare with alternative hypotheses. Perhaps it tracks a broad rank-related context, a particular arrangement, a history effect or several unrelated features. The current map does not decide among them.

## 13.3 Four different attention pictures

The content map shows Q_i dot K_j / sqrt(32). The fixed-bias map shows S_ij. The adaptive map shows sum_r L_ir g_r R_jr. Add them, apply row softmax and you obtain the attention map A_ij. These are mathematically related but not interchangeable. Signed logit contributions can be negative; attention weights cannot.

Then multiply A_ij by V_j and project the head's coordinates through its output slice. That produces a message vector into the receiver's residual update. A norm map summarizes its magnitude. A signed projection onto a chosen output-relevant direction can instead summarize alignment with that direction, but the chosen direction must be stated. Norm and alignment answer different questions.

High attention does not imply large message magnitude. Large message magnitude does not imply large final effect. Large final effect under zeroing does not identify a human concept. The sequence of distinctions is essential. Research on attention explanations has demonstrated related pitfalls, but the concrete reason here is visible in the equations themselves: routing, values and later transformations all matter. [R7]

## 13.4 Gradients and sensitivity maps

A gradient of a target scalar with respect to an activation measures infinitesimal local sensitivity. For example, the derivative of the Bg7-minus-Bf4 logit gap with respect to one residual coordinate says how a tiny change at that point would affect that comparison, holding the rest of the upstream state fixed and differentiating downstream.

Multiplying gradient by activation gives a local first-order contribution heuristic. It is exact only in special linear settings. Nonlinear interactions and saturation can make it misleading for large removals. Integrated gradients average sensitivities along a chosen path from a baseline to the actual input, but that baseline and path are part of the question. Interpolating between legal boards can produce feature arrays that represent no legal chess history.

This book's Bg7 causal results come from actual finite interventions, not from uncomputed gradient maps. The toy attention derivatives are independently checked mathematics. We do not label either as a measured full-network gradient attribution.

## 13.5 Linear probes: is a property readable?

A linear probe trains a simple predictor on frozen representations to estimate a label, such as whether a rook has few non-capturable moves under a precisely defined rule. A successful held-out probe shows the label is linearly decodable under that evaluation setup. It does not prove the engine uses that direction to choose moves.

Suppose a probe predicts "rook on h7" with high accuracy from the input. That is unsurprising: piece location is explicitly encoded. If a later-layer probe predicts safe rook mobility better than an input baseline, that can suggest the network has made the relation easier to read. But a weak input baseline, leakage from adjacent positions, material imbalance or square-specific correlations can make the comparison look stronger than it is.

Useful controls include a probe on raw inputs, a probe on random-network features, matched model capacity and regularization, and splits by game with transposition-aware checks. If the trained feature set has thousands of dimensions and the test set is tiny, apparent success can be unstable. Hyperparameters must be selected without peeking at the final test set.

The atlas currently has small saved diagnostic recordings and random-feature support. It does **not** contain a completed broad held-out probe study proving a rook-confinement representation. The larger corpus and probe work were queued, not executed for this textbook.

## 13.6 Sparse autoencoders: a different coordinate system

A sparse autoencoder attempts to reconstruct activation vectors using a larger dictionary of features while encouraging few features to be active at once. A basic form is f=ReLU(x W_enc+b_enc), x_hat=f W_dec+b_dec, trained with reconstruction error plus a sparsity penalty. The hope is that the learned dictionary separates features that overlap in the original coordinates. [R6]

@equation sae

Here x is a recorded neural activation, f is the new feature vector and x_hat its reconstruction. Lambda controls the penalty on active feature magnitudes. A tiny illustrative example: if two dictionary directions are [1,0] and [0,1], then x=[2,0] can be represented by one nonzero coefficient, while x=[2,1] needs both. Real learned dictionaries are overcomplete and non-orthogonal, so interpretation is harder.

A sparse feature with many trapped-rook examples would be a promising candidate, not a certificate. Reconstruction error may lose behaviorally important information; the dictionary may split one concept into several features or merge several concepts. It needs held-out examples and interventions. No SAE has been trained on this VEGAM model for the present book, and no figures here are fabricated SAE discoveries.

## 13.7 Activation patching: transfer a candidate mechanism

Patching replaces an activation in one forward computation with a recorded activation from another input, then recomputes downstream outputs. It asks whether that internal difference mediates an output difference. Unlike setting a signal to zero, a donor patch can use values actually produced by the network. But mixing a donor signal with a recipient's other states can still be unnatural.

A valid comparison must match shape, layer, orientation and semantic role. Patching a White-oriented g7 row into an unrelated Black-oriented square without remapping would not test the intended relation. Whole-board context, history and move candidates also affect what a patch means.

A successful patch that restores part of a lost move preference is stronger evidence of mediation than a pretty heatmap. It still needs controls for donor choice, overall activation scale and unrelated changes. A mechanism is not established by one successful swap selected after trying many possibilities.

## 13.8 Learned look-ahead is an experimental claim

Other chess-network research has found evidence consistent with learned look-ahead using targeted mechanistic experiments. That makes the question worth asking. It does not imply that every transformer chess net implements the same circuits, or that its number of blocks equals a search depth. [R8]

For VEGAM, one could test whether a late representation selectively encodes consequences of candidate continuations and whether patching those signals changes appropriate move comparisons. A convincing study would need rival explanations such as static pattern recognition, material cues and policy memorization. This book gives the instrumentation and vocabulary to ask such questions, not a preannounced positive result.

@check 21|A linear probe predicts a chess property from block 8 with 95% accuracy. What does that establish, and what important causal claim does it leave open?

# 14. The Bg7 interventions: what survived being cut
@objectives Read the actual head and neuron removals; separate additive readout effects from nonlinear upstream effects; design controls for a rook-confinement mechanism; state what remains unproven.

The one-position study made 116 interventions: 96 separate attention-head removals, a mean replacement and restoration for the selected head, eight adaptive-branch removals, decoder-unit removals and restoration, MLP-unit interventions and restoration, and two decoder-term removals. It used the same saved input and weight identity throughout. The numbers in this chapter are measurements, not proposed experiments. [M2]

## 14.1 Start with an output question

We chose two primary readouts: Bg7's policy probability and its logit gap over Bf4. The first measures its share of all legal moves; the second isolates a named competitor. State WDL and action WDL were also recorded. This prevents a vague statement such as "the head mattered" from hiding which behavior changed.

A head removal zeros the selected head's 64 x 32 value-mixture output, after A times V and before the multihead output projection. Other heads and the residual route remain active. The model then recomputes all later layers and heads. This is not the same as zeroing its attention logits, replacing attention by uniform weights or removing only its contribution to g7.

## 14.2 All 96 heads, one at a time

The largest absolute change in Bg7 probability came from removing **block 1, head 6**. Bg7 fell from **72.11% to 44.27%**, a drop of **27.85 percentage points**. Bf4 was about 14.03%, and Bg7 still ranked first. The gap over Bf4 changed from 1.28608 to about 1.14935. [M2]

That pair of observations is instructive. Much of the lost Bg7 share went to the rest of the candidate set, rather than simply to Bf4. Reporting only the probability drop would exaggerate how much this head explains the particular Bg7-versus-Bf4 contrast.

@figure head_ablations|MEASURED. Each cell is a separate full forward pass with one head output zeroed. Values are Bg7 policy changes in percentage points. Bg7 stayed first in all 96 cases. The head highlighted in the text was selected after this screen, so subsequent interpretation is exploratory.

Removing block 8, head 3 lowered Bg7 to about 53.79%. Removing block 7, head 2 instead raised it to about 80.57%. Heads can support or oppose a chosen move relative to the rest of the model. "Important" does not mean "always pushes the correct move," and the remaining model can compensate through other routes.

The fact that Bg7 survives every single-head removal does not prove complete robustness. Two components might be redundant individually but necessary jointly. Larger combinations can produce nonlinear effects. Conversely, a huge single-head effect might reflect broad disruption rather than removal of a specialized concept.

## 14.3 Mean replacement and restoration change the question

Replacing the selected head's output with its mean across receiving squares leaves a board-dependent average message but removes its receiver-specific differences. Bg7 then receives about **69.25%**, much closer to baseline than under complete zeroing. This suggests that keeping the average head signal preserves much of this measured preference under this intervention. It does not prove that spatial distinctions in the head are unimportant for all outputs or positions.

Explicit restoration reproduced the baseline. Restoration is a basic implementation check: a supposed causal effect that persists after restoring the untouched signal may be a state-management or comparison bug. It does not transform an exploratory ablation into a general theory.

@figure interventions|MEASURED. Several interventions with different scopes. A full head removal, mean replacement, adaptive-bias removal, final decoder-unit removal and decoder-context removal are not equivalent experiments. They must not be ranked as if they removed equally sized semantic concepts.

## 14.4 Adaptive bias has effects beyond the chosen move

Zeroing block 1's entire adaptive bias increases Bg7's share to about **75.06%**, yet lowers the raw state win probability from **92.82% to 81.94%**. One scalar looks more favorable to Bg7 while another becomes less optimistic about White. This is a concrete reason to inspect multiple outputs.

It would be wrong to conclude that the adaptive branch is bad because removing it raises the chosen move's policy. The network is a jointly trained system, this is one position, and policy concentration is not a complete quality measure. It would be equally wrong to declare the branch a rook-trap detector because its removal changes the evaluation. A large intervention can disturb broad context and calibration.

## 14.5 A high-firing neuron need not drive the move

Zeroing main-MLP block 1 unit 360 across all squares changes Bg7 only from about **72.11275% to 72.10143%**. Its high g7 activation is therefore not evidence of a large net contribution to Bg7 probability under this global removal. The state win estimate still changes, so "no policy effect" is not "no effect anywhere."

Zeroing the same unit **only at g7** raises Bg7 to about **74.72%**. The global and local effects differ because other square locations matter and later processing is nonlinear. A separately seeded comparison unit, 372, gives Bg7 about 72.04903% when removed globally. The seeded control avoids selecting only dramatic examples, but it is not a full distribution of matched control units.

The exact readout unit 34 is a different case. Its contribution to the Bg7-Bf4 gap is directly additive at the final layer, and removing it changes that gap by the predicted amount. We can be more precise there because the remaining operation is linear. Precision of arithmetic does not automatically supply a chess label for the unit.

## 14.6 What has been established?

**CODE / MEASURED:** The recorded model strongly prefers Bg7 before search. Specified internal components affect that preference when removed or replaced. The final readout gap is exactly decomposable into 64 terms. The native fixed-visit search concentrates heavily on Bg7 and favors a king-march continuation.

**MATHEMATICS:** Attention weights, signed values, output projections and nonlinear downstream processing are distinct factors. Probability changes can occur without a change to the Bg7-Bf4 ratio. A late linear readout permits exact additive decomposition; an early ablation generally does not.

**HYPOTHESIS:** The model's preference partly reflects a reusable representation of rook confinement and the time it buys for king activity. That is a plausible chess interpretation, but the present single-position measurements do not isolate or generalize such a representation.

This is a substantial amount of knowledge. It is also more reliable than naming a colorful head "the trapped-rook head" and treating the name as a discovery.

## 14.7 A rigorous rook-confinement investigation

The next experiment should begin with a definition, not a neuron gallery. Specify a target such as: "the defending rook has no legal move that avoids immediate capture under a stated one-ply reply test." That is computable and fits our three local examples. It is still not identical to the strategic concept of a bad rook: a capture can be tactically favorable, and a rook can be strategically tied down despite having uncapturable moves.

Construct several label families: raw legal rook mobility; immediate non-capture mobility under the reply test; defenders of blocking pieces; material balance; king proximity; and whether a candidate move changes these quantities. Keep the rule-based labels separate from any engine-estimated outcome labels. This prevents the definition from silently changing to fit whatever a probe discovers.

Then assemble legal positions in which the key factors vary. Include positive cases with the supporting pawn and bishop restriction; negative cases with an available rook exit; controls with the same material and similar squares but a different king location; and positions where a rook sacrifice is useful despite immediate recapture. Use real game continuations or validated legal constructions, with appropriately regenerated history. Simply deleting h6 from one input row is useful as an artificial intervention but not sufficient as a chess test set.

Split by game and by closely related position families. Adjacent positions and near-duplicate counterfactuals must not straddle train and test. Match phase, material, side to move and square distribution where possible. Otherwise a probe may learn "late endgame with a rook near h7" instead of the intended relation.

Train small probes on raw inputs, random-network representations and successive trained layers. Evaluate on held-out families, including mirrored color/orientation cases and unrelated rook-confinement examples. Freeze metric choice and hyperparameters before the final evaluation. A gain over the input baseline would indicate easier readout of the defined relation, not yet causal use.

For candidate causal mediation, compare a legal positive/negative pair with different model preferences. Patch a candidate internal signal from the positive into the negative computation and ask whether the relevant logit contrast moves in the predicted direction. Also patch a matched control signal, use several donors and perform the reverse patch. Restore the baseline afterward. Record all attempted candidates, not only successful ones.

Finally test specificity and generalization. Does the intervention alter an unrelated move equally? Does it change state value without the expected action contrast? Does it work when the pieces occupy different files? Does another component compensate? Can a smaller patch recover the effect without broad disruption? A robust answer to those questions could justify a mechanistic claim. A failure would be informative too: the useful relation may be distributed, nonlinear or different from the proposed concept.

## 14.8 What was not run

No large held-out probe study, SAE training, broad donor-patching experiment or causal circuit discovery was performed for this textbook. Those optional jobs are not prerequisites for explaining the actual forward pass. The book uses the existing 64-position calibration and the completed one-position study, plus lightweight arithmetic and figure generation. It does not consume GPU time to manufacture a more dramatic story.

@check 22|Why is the 27.85-percentage-point Bg7 drop insufficient to identify block 1, head 6 as a trapped-rook detector?

@check 23|Why should the rook-confinement test include positions where sacrificing a rook is tactically good?

# 15. Return to the game with a better instrument
@objectives Apply the distinctions to other positions; read the atlas without treating every panel as a verdict; connect the numerical trace to a coherent account of the Catalan game.

The Bg7 study is rich because it separates a visible chess mechanism from an inspectable numerical mechanism. Other moments in the same game show why a single kind of output cannot tell the entire story. The following comparisons use existing saved diagnostic recordings; no new searches were launched for them. [M1, G1]

## 15.1 Before 7.Ne5: a move preference without a winning verdict

@figure early_ne5|RULES / GAME. Before 7.Ne5, White's development and the central pawn tension create a different problem from the late rook restriction. The position is reconstructed from the same PGN, not chosen to resemble the Bg7 motif.

The saved raw policy puts Ne5 first at about **39.98%**, followed by O-O at 18.52% and Nbd2 at 14.16%. State WDL is approximately 20.43% win, 71.88% draw and 7.69% loss. A clearly preferred move can coexist with a draw-heavy evaluation. Preference answers "which action gets probability here?"; value answers a different prediction question.

The subsequent game exchanges bishops and alters the pawn structure. We can describe those actual moves accurately without claiming the raw network explicitly simulated the entire continuation in one pass. Search and future opponent choices remain part of what happened.

## 15.2 Before 44.h6: the played move need not lead raw policy

At this position, the saved raw network prefers **g6+ at 42.11%**, while **h6 receives 29.74%** and Bc3 11.53%. Yet the game move was h6. The state head predicts about 84.12% win. This is a direct counterexample to equating the recorded game move with the network's first unsearched choice.

The h-pawn later becomes the supporting piece in the Bg7 restriction. That retrospective connection is useful chess context. It does not prove that the neural forward pass at move 44 was consciously planning move 81, nor that a present-day fresh search would reproduce the original time-managed decision. The retained tree, clock and search trajectory in the game differed from a fresh raw evaluation.

@figure before_h6|RULES / GAME. Before 44.h6. Gold marks the raw policy's leading g6+; teal marks the played h6, which advances the pawn that survives into the later restriction. This figure illustrates the game sequence, not a claim that the entire sequence was internally planned.

@pagebreak

## 15.3 From maneuvering to a concrete restriction

@figure game_arc|RULES / GAME. Before 68.Kd1 and before 81.Bg7. The game contains a long maneuvering phase. White improves its king and rook activity before the bishop establishes the h7 restriction.

Before 80.Be5, the saved raw policy ranks Rd8, Rf6 and Kc3 ahead of the played bishop move. Before 81.Bg7, after Black's actual reply ...Rh7, Bg7 becomes a dominant raw preference. A move's tactical and strategic meaning depends on the current arrangement and history. We should not assign one permanent preference to the bishop independent of the board.

After the restriction, the game develops into a king march. Before 84.Kc3, the network puts Kc3 first at about 36.83% while state win probability is about 97.01%. Later, before 94.a8=Q, state win probability is about 99.9902%, but queen promotion receives only about 31.94% of raw policy. Several moves can have substantial policy probability in a position the value head considers overwhelmingly favorable.

@figure conversion|RULES / GAME. The king enters on b4 and the a-pawn later promotes. The visual sequence connects the local restriction with the actual conversion, while leaving open which hidden representations mediated the prediction.

## 15.4 A practical route through the atlas

Start with the position view. Confirm the move number, side to move, full-history status and model identity. Read policy and WDL separately. Before interpreting a difference, confirm whether it is a raw diagnostic result, a native result or a searched result, and whether its receipt matches the currently selected position.

Then inspect the inputs. Pick g7 and see the 120 features described in Chapter 2. Move to layers to compare its representation at successive blocks. The neuron view answers where a selected hidden channel is high; the attention view separates content, fixed bias, adaptive bias and normalized weights. The gate view shows signed coefficients, and the move-decoder view compares source/destination combinations and final features.

For an attention inspection, specify a receiving square, block and head before looking at the supplier map. Check the colorbar units. A static-bias map and an attention-probability map have different scales and meanings. For a neuron gallery, remember that the available corpus is small and the displayed extremes are selected examples, not a representative distribution.

Only use the intervention controls when the output question is explicit. Save the baseline; change one named signal; record policy, logit contrast and WDL; restore; compare with controls. The search panel adds native tree evidence, but its chosen settings affect what can be compared with normal gameplay. The PDF is self-contained, so none of these clicks is required to understand the examples already printed here.

## 15.5 The original paragraph, now unpacked

"Content compatibility" now means a learned bilinear score made from normalized square rows through Q and K projections. "Fixed bias" means a trained 64 x 64 table for one head. "Adaptive bias" means sixteen learned outer-product patterns combined by position-dependent signed gates. "Softmax" means a row-wise exponential normalization across supplying squares. "Information transfer" means a weighted sum of V vectors, then a learned output projection and residual addition.

Those updated square representations pass through nonlinear MLPs and further blocks, then source, destination and global routes into a move decoder. Its final policy weight vector gives Bg7 a 1.28608-logit advantage over Bf4 in the recorded diagnostic. Search allocates most of its visits to Bg7 and evaluates continuations that include the king march. Training made the parameters useful by sending prediction errors backward through precisely those operations.

You can now reconstruct the computation. The remaining question is no longer "what are these mysterious words?" It is a concrete scientific question: **which reusable chess relationships do these learned computations implement, and how would we distinguish that explanation from its alternatives?** The arithmetic is open. The semantic investigation has a testable next step.

@check 24|At promotion time, how can state win probability be about 99.99% while the queen-promotion policy share is only about 31.94%?
