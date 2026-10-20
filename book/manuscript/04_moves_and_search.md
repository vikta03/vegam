# 11. From square representations to Bg7
@objectives Reconstruct the legal-action decoder; calculate a real source/destination interaction; distinguish policy and WDL outputs; account exactly for Bg7's logit advantage over Bf4.

After block 8, the model applies a final LayerNorm to each 384-entry square row. Call the resulting 64 x 384 tensor H. It then reads the board in two complementary ways: a global context summarizes all square rows, while separate source and destination projections preserve access to individual squares. [S1: Net.forward]

## 11.1 The global route

The model averages H across the 64-square axis, giving one 384-vector. A learned linear map to 128 coordinates and GELU produce the context vector c. A mean sounds crude, but the rows being averaged are already globally contextualized. Each can contain transformed information about other squares, and learned position information has already influenced the computation.

The mean itself is insensitive to the order in which its 64 already-computed rows are added. That does not make the whole model invariant to swapping pieces on the original board. Such a swap changes square embeddings, pairwise routing, adaptive gates and the preceding contextual rows. A final symmetric reduction can summarize an order-sensitive computation.

Three linear heads read c: state gives three logits, outcome gives three logits and duration gives two parameters. Another linear map transforms c to a 64-entry vector g used in every move's decoder. A global signal can therefore influence all move scores even though it is identical as an added vector before their nonlinear processing.

## 11.2 A candidate is an ordered source/destination pair

The model separately projects each H row from 384 to 64 coordinates through source and destination linear layers. Bg7 gathers source row e5 and destination row g7; Bf4 gathers the same source e5 and destination f4. The source and destination maps have different learned parameters because leaving a square and arriving at it are different roles.

The precise combination is:

@equation decoder

Here s_m and d_m are the gathered 64-vectors for move m; the circle means elementwise multiplication, not a dot product; g is the global context contribution; p_m and k_m are the promotion and castling embeddings selected by the move's integer codes. The product is divided by 8. All six terms produce 64 entries and are added coordinate by coordinate.

The scale 8 equals sqrt(64), but the source code does not establish a trained variance-normalization theorem for these vectors. We can state the computation exactly without claiming an unrecorded design proof. The product allows the influence of one endpoint to depend on the other: for coordinate r, s_r d_r is large and positive when their signs agree and magnitudes are large, negative when signs differ.

The combined vector passes through GELU, then Linear(64,64), then another GELU, then Linear(64,4). The final four numbers are one policy logit and three action-WDL logits. There are **two GELUs in this decoder path**, one before the joint hidden linear layer and one after it. Omitting the first would describe a different network.

## 11.3 A real decoder coordinate

For Bg7, the first source coordinate is -1.3241186 and the first destination coordinate is -1.1348643. Their product divided by 8 is +0.1878369. Add the global context +0.1421157, the no-promotion embedding +0.0302429 and the no-castling embedding -0.3332520. The complete sum is:

-1.3241186 - 1.1348643 + 0.1878369 + 0.1421157 + 0.0302429 - 0.3332520 = **-2.4320393**. [M2]

The "none" embeddings are learned vectors, not mandatory zero vectors. Their contribution can act like a shared offset for ordinary moves. They cancel from a direct pre-activation difference between Bg7 and Bf4, but they can still affect the comparison after nonlinearities.

@figure decoder_terms|MEASURED. Both moves leave e5, so their source vectors match. Their destination and product vectors differ. The bars show term norms, not additive contributions to the final move logit; nonlinear transformations come next.

## 11.4 Why common context does not cancel out

Before a nonlinearity, (s_1+d_1+g)-(s_2+d_2+g) eliminates g. After a nonlinearity f, f(a+g)-f(b+g) generally depends on g. As a simple illustrative example, with ReLU, a=-1 and b=1 give a difference -1 when g=0, but -2 when g=2. GELU changes the comparison smoothly rather than with ReLU's sharp corner.

In the recorded intervention, zeroing the decoder's context contribution lowers Bg7 policy from 72.11% to 39.52%, while Bf4 rises to 32.36%. The Bg7-minus-Bf4 logit gap falls from 1.28608 to about 0.19971. This is direct evidence that the context route affects this comparison. It does not establish which human chess fact the route represents, and zeroing a full vector can create an unusual hidden state. The state-WDL head remains unchanged in this intervention because it reads c before this move-specific branch. [M2]

## 11.5 Raw logits become a legal-move distribution

Bg7's raw policy logit is **3.5559125**; Bf4's is **2.2698314**; Kc3's is **0.6979170**. Applying softmax over all 22 legal candidates yields approximately **72.11275%, 19.92848% and 4.13809%**. These are the CPU diagnostic values. The saved GPU diagnostic and native paths differ slightly, as documented in the reproduction appendix.

@figure outputs|MEASURED. Left: raw policy shares among legal moves. Right: state and action WDL distributions. A move can have a small policy share but a high predicted winning chance; these are different targets and different normalizations.

The logit gap between Bg7 and Bf4 is 1.2860811. Their probability ratio is exp(1.2860811), approximately **3.6186**. This ratio is independent of the other twenty logits. Their individual probability shares are not: increasing Kc3's logit would reduce both shares while preserving their ratio if neither of their logits changed.

That is why the book reports both a probability and a logit difference for interventions. A change in Bg7's percentage can reflect competition elsewhere. The gap isolates one comparison, though it still does not reveal a complete semantic mechanism.

## 11.6 The exact final-readout account

Let h_m be the decoder's 64-entry hidden vector after its second GELU. The policy logit is w dot h_m + b, with a shared learned 64-entry policy weight vector w. Therefore:

@equation readout

The shared bias cancels in the Bg7-minus-Bf4 difference. Every hidden coordinate contributes exactly w_r times (h_Bg7,r-h_Bf4,r). Summing all 64 contributions reconstructs the measured gap to about 2.4 x 10^-7 in the saved float arithmetic. This is an exact algebraic explanation of the final readout, not an approximate heatmap attribution. [M2]

@table readout

For unit 34, the Bg7 activation is 4.500099 and the Bf4 activation is 2.367852. Their difference, 2.132246, multiplied by policy weight 0.123352 gives **+0.263017** of logit advantage. For unit 42, Bg7 has the *lower* activation: 0.824790 versus 1.700437. Its policy weight is negative, -0.224121, so that decrease contributes **+0.196251** in favor of Bg7. High activation is not synonymous with positive evidence.

Unit 26 contributes **-0.087375**, opposing Bg7 relative to Bf4. The network's output is a sum of supportive and opposing terms. There is no requirement that all internal components agree about a move.

@figure readout|MEASURED. The twelve largest absolute contributions to the Bg7-minus-Bf4 policy gap. Teal favors Bg7 in this comparison; red favors Bf4. The complete 64-term list is saved with the reproduction data.

Removing decoder unit 34 lowers Bg7's probability to about 65.80% and reduces the gap to 1.023064. That gap change matches the removed +0.263017 contribution, because this is the final linear readout. Earlier-layer removals do not have such simple additive effects: they change later nonlinear computations.

We now know exactly how the last layer prefers Bg7. We still do not know a clean chess description of unit 34. It is a move-conditioned feature formed from transformed source, destination, interaction and context, not simply "a bishop on g7 neuron." Exact arithmetic and uncertain semantics can coexist.

## 11.7 Three WDL meanings and one expected score

The state head predicts WDL for the current side to move under its teacher-state target. The action head predicts WDL associated with a candidate move, still from the current mover's perspective in this interface. The outcome head predicts the separate game-result target. Their logits undergo separate three-way softmax operations.

For Bg7, action WDL is approximately [0.935444,0.057536,0.007019]. Action q=W-L is **0.928425**. The raw state q is about **0.920847**. If a win scores 1, a draw 0.5 and a loss 0, expected score is W+D/2=(1+q)/2. For the state distribution this is about **0.960423**. Expected score and win probability are therefore not the same number.

Kc3 has a higher action q than Bf4 in the raw network, yet substantially lower policy probability. That is not automatically a bug: policy is trained to match teacher move distribution, and action WDL is a separate sparsely supervised estimate. Search is the place where these signals meet evidence from successor positions.

@check 17|Why can a lower activation on decoder unit 42 favor Bg7? Use the sign of its output weight.

@check 18|If Bg7 and Bf4 both receive the same additive change in their final logits, what stays fixed, and what can change if the other move logits do not change?

# 12. Where neural prediction ends and search begins
@objectives Follow policy and WDL into classic MCTS; explain the exact Q-head blend; distinguish retained trees from DAG sharing; reconstruct the reported score's conversion and its limits.

The network evaluates one encoded position in a forward pass. The engine repeatedly chooses positions to evaluate, applies exact moves, recognizes terminal states and combines results in a search tree. Those are different computations. A strong unsearched Bg7 preference helps guide search; search can confirm it, overturn it or fail to examine a critical alternative within its budget. [S6, S7]

## 12.1 A leaf evaluation supplies priors and value

At a newly evaluated nonterminal node, policy probabilities become priors P on legal outgoing moves. State WDL supplies a value prediction; the native adapter normalizes WDL and exposes q=W-L and d=D. With Q enabled it also supplies per-move action q and d. The actual daily configuration disables moves-left utility and outcome blending.

A selection rule balances accumulated value with an exploration term. In a simplified view of this classic path, the score is Q + c P sqrt(N_parent)/(1+N_child), with implementation details for completed and in-flight visits, first-play urgency, dynamic c, root restrictions and proven results. [S7: GetFpu, ComputeCpuct; S8: GetU]

@equation puct

P is a prior probability, not a visit count. N records search effort; Q reflects backed-up value once a move has completed visits. The exploration bonus favors a promising move that has received relatively little work. c can grow with parent visits in the implemented formula. Virtual/in-flight accounting helps threads avoid doing identical work simultaneously.

An illustrative comparison: with c=2, sqrt(N_parent)=10, P=0.6 and N_child=9, the bonus is 2 times 0.6 times 10 / 10 = 1.2. With the same prior but 99 child visits it is 0.12. This large toy bonus is an arithmetic demonstration, not a realistic assertion about the anchor search's exact selection values.

The selected move is applied to a copied or advanced rule state. At the next player-to-move node, value perspective changes. Backup handles that sign convention so a move good for the opponent is bad for us. Terminal checkmates and draws come from chess rules, not from a neural probability threshold. The network may be extremely confident and still be wrong; a recognized terminal result has a different status.

## 12.2 What your Q option actually changes

An unvisited edge has no completed search value. Ordinary first-play urgency supplies a fallback derived from the parent and search settings. When a finite action-head estimate is present, the code uses **75% fallback plus 25% action estimate**, including the configured draw-score term. Once a child has a completed visit, the backed-up child value takes over. [S8: Edge::ColdQ, EdgeAndNode::GetQ]

@equation coldq

If fallback is 0.6 and Bg7's action q is 0.928425 with draw score zero, ColdQ is 0.75 times 0.6 + 0.25 times 0.928425 = **0.682106**. With Bf4 action q=0.872066, it becomes **0.668017**. The action head modestly differentiates cold edges; it does not permanently add 25% action prediction to every searched Q.

This happens throughout the tree, not just at the root. The prior P still drives exploration, and accumulated search evidence can disagree with action advice. Root policy probabilities do not change merely because the option is switched off. The neural graph can produce the same outputs while search chooses a different allocation of visits.

The earlier corrected-engine Q screen ended at 5.5-5.5 over eleven scored games. It did demonstrate changed exploration in some diagnostics, including different root visit distributions, but it did not establish a general playing-strength advantage or equivalence. Your choice to daily-drive Q-on is a configuration choice under that uncertainty, not a measured proof that the option can only help. [T3]

## 12.3 The actual 1,024-visit Bg7 search

The saved native diagnostic used a fresh tree, one thread, batch size one, Q on and a fixed 1,024-root-visit budget. It chose Bg7. Of the 1,023 child visits, **977 went to Bg7, 43 to Bf4 and 3 to Kc3**. The extra root visit accounts for the difference between root and child totals. [M3]

@figure search|MEASURED. Native priors versus child-visit shares in the saved bounded search. Bg7's share rises to about 95.50% of child visits. This is search allocation on one position, not an estimated probability that Bg7 is optimal.

The final principal variation was **Bg7 Kg4 Rf6 Kg5 Kc3 Kh5 Kb4 Kg5 Kxa4**. It connects the kingside restriction with queenside king activity, which is consistent with the human chess account. A PV is the engine's current favored continuation through its tree. It is not a proof that the opponent has no better defense, and a short PV does not summarize every line examined.

The visited Bg7 edge reported WL about 0.96339 and D about 0.025. Bf4 reported WL about 0.73885 and D about 0.234. These searched values differ from raw action predictions and can justify a more concentrated visit distribution. Rounded verbose output is sufficient to interpret the trend, not to reproduce every floating-point selection decision.

The final diagnostic NPS was 720. It is deliberately a one-thread, batch-one interpretability search, not a daily-play speed benchmark. Earlier ordinary two-thread/batch-eight match searches had median native NPS around 6,230 for Q-on in their recorded conditions. Do not infer a speed regression by comparing these differently configured measurements. NPS itself counts reported search visits per measured interval, not necessarily unique GPU evaluations. [M3, T3]

## 12.4 A reusable tree is not the same as DAG search

Classic search can keep the useful part of a tree after played moves and continue from that retained subtree. Previously accumulated visits can then inform the next decision. A transposition DAG attempts additional sharing when different move orders reach equivalent reusable states. This can avoid repeated work, but history, repetition, rule-50 state and backup semantics complicate what may safely be shared.

Your daily launcher explicitly selects **classic**. The separate **dag-preview** candidate includes the ordering correction and was tested in the earlier bounded Fixed 3 work. That record found higher reported NPS on five positions, with a 4-win, 12-draw, 4-loss twenty-game screen against classic. These results do not establish a general strength advantage, and they do not turn the current launcher into DAG search. [S6, T4]

The exact-input backend cache and retained search tree are yet other objects. A neural-output cache reuses a forward result for matching encoded input; a tree retains search statistics and structure; a DAG shares eligible transposition information. They save different work. A cache key that ignores relevant history can be wrong even if the visible board matches.

## 12.5 Why the reported +7.38 is not a raw head output

The native search reported `score cp 738` at the end of the diagnostic, conventionally displayed as +7.38. The model has no single scalar centipawn head that directly emitted 738. Its state and action heads emitted WDL logits; search backed values up; a score-display conversion then transformed the chosen edge's statistics. [S7: Search::GetUciInfo]

The default ScoreType is WDL_mu. The code first attempts a random-walk-style WDL conversion. From w=(1+WL-D)/2 and l=(1-WL-D)/2 it forms a=log(1/l-1), b=log(1/w-1), and mu=(a-b)/(a+b), with additional rescaling safeguards. It then chooses between 100mu and a centipawn-like fallback according to explicit conditions.

@equation score_fallback

For the rounded final Bg7 statistics, w is about 0.969195 and l about 0.005805. The approximate 100mu is about 507. The fallback 45 tan(1.56728071628 times 0.96339) is about **738.08**. The implementation's condition selects the fallback here, matching the printed 738 after integer conversion. This is a demonstrable numerical bridge from searched WDL-like statistics to the GUI score, not a pawn-count interpretation.

Near |WL|=1, the tangent curve becomes steep. Small value changes can create large displayed score swings while move ranking remains stable. The original game recorded +5.44 at Bg7 under its own time budget and retained-tree history; our separate fixed-visit search reported +7.38. Those are different search circumstances, not two inconsistent raw network heads.

If search recognizes a terminal mate, it reports a mate score through another branch. Neither a huge centipawn-like score nor a duration prediction is a proof of mate. This distinction explains why your neural engine can show very large numeric advantages before switching to a mate announcement.

## 12.6 Correct search mechanics still matter with a good net

The current source lineage includes corrections involving move selection ordering, repetition identity for unusable en-passant markers and FEN counters, along with earlier search and rule repairs. These affect whether the engine searches and reports the intended state correctly. They are not new learned concepts in the network. [T4]

The root mate-safety logic also uses exact rules to constrain immediate mating outcomes. It can catch a one-move disaster regardless of policy preference, but it is not a complete tactical solver. The broader lesson is that a trained net, an input adapter, a rule engine and a search algorithm must all agree. A bug in one can make a correct neural computation look strategically inexplicable.

@check 19|After an edge has a completed visit, does the deployed Q mechanism keep blending 25% raw action q into its searched value?

@check 20|Why can two searches of the same board show different centipawn-like scores even when the model weights are unchanged?
