# 1. A bishop move worth opening the network for
@objectives Read the anchor position as chess; distinguish a legal fact from an evaluation; identify the exact question the numerical trace will answer.

White has no material advantage. Both sides have a rook, a bishop and four pawns. Yet White's next move, **81.Bg7**, makes Black's position feel dramatically smaller. This is an excellent place to begin understanding your network: the important thing is a relationship among pieces, squares and future possibilities, not a material count.

The position comes from your 9 October 2026 game, VEGAM-TF3-Q against forge_mini_mpv50, at 60 seconds plus two seconds per move. It is White to move. White has Kc2, Rd6, Be5 and pawns a3, b2, e3, h6. Black has Kh5, Rh7, Bd5 and pawns a4, c4, c6, e4. No castling or en-passant right remains. Seventeen halfmoves have elapsed since the last pawn move or capture. [G1]

@figure position|RULES / GAME. The actual position before and after 81.Bg7. Arrows on the second board mark White's control of h8 and h6 from g7, and the h6-pawn's protection of g7. These are chess-rule relationships, not measured attention.

## 1.1 What changes when the bishop reaches g7?

Before the move, Black's rook on h7 has access along the seventh rank. After the bishop occupies g7, the nearest square to its left is an enemy piece protected by the h6-pawn. Below the rook sits that pawn, protected by the bishop. Above the rook is h8, also controlled by the bishop. The two White pieces support a compact restriction.

A precise statement is better than saying the rook has "no moves." It has **three legal rook moves** immediately after Bg7: Rh8, Rxg7 and Rxh6. Each permits an immediate capture of the rook. The distinction between legal mobility and safe mobility is central to this position and to interpreting a neural feature. A feature that predicts the number of legal rook moves has not necessarily learned the strategic fact we care about.

@figure exit_h8|RULES / GAME. After 81...Rh8, 82.Bxh8 captures the rook. The bishop travels one diagonal step from g7 to h8.

@figure exit_g7|RULES / GAME. After 81...Rxg7, 82.hxg7 exchanges White's bishop for Black's rook and creates a pawn on g7. A move can be legal while failing to solve the defender's problem.

@figure exit_h6|RULES / GAME. After 81...Rxh6, 82.Bxh6 captures the rook. Black can choose a king or bishop move instead; these diagrams are local answers to rook moves, not a proof of a forced win against every defense.

White's rook on d6 also attacks the bishop on d5, while Black's c6-pawn protects that bishop. Black's king on h5 cannot simply walk onto the protected h6-pawn. White's king is far away on c2. That last fact matters: a restriction on one wing can buy time for a king to enter on the other. In the game, White eventually plays Kc3, Kb4 and Kxa4, and promotes the a-pawn. The beautiful connection is between immobilizing a rook on the kingside and winning the queenside race.

That is a human chess interpretation supported by the legal sequence, not yet an explanation of what VEGAM computed. We will keep these two statements separate: **Bg7 creates the restriction just described**; **VEGAM's raw network strongly prefers Bg7**. Establishing the first does not establish that the second arose from an internal "rook confinement" detector.

## 1.2 The question we can actually answer

Before any new search in our diagnostic recording, the network assigns Bg7 about 72.11% of the legal-move policy probability and Bf4 about 19.93%. It predicts a state WDL distribution of approximately 92.82% win, 6.44% draw and 0.74% loss, from White's perspective. These are outputs of a learned predictor. They are not tablebase facts and do not mean the position was played thousands of times to measure those percentages. [M2]

The main question of this book is: **what exact numerical path takes this board and its history to those preferences?** We can reconstruct the inputs, projections, attention, nonlinear transformations and final readout. We can change a component and measure what happens. We cannot infer the human meaning of every hidden coordinate merely by printing it. That is a boundary of evidence, not a gap to fill with imaginative narration.

## 1.3 How to read the evidence

Throughout the book, three labels distinguish different claims. **CODE / MEASURED** means the implementation directly specifies the operation or a named receipt records the result. **MATHEMATICS** means the statement follows from an equation under stated assumptions. **HYPOTHESIS** means an interpretation of learned behavior that needs additional evidence. Numerical teaching examples are labeled **ILLUSTRATIVE** and never passed off as trained VEGAM activations.

The baseline used for the detailed arithmetic is the saved CPU diagnostic run: float32 arithmetic with the trained EMA weights first rounded to float16. It was checked against the saved native TensorRT output and the original PyTorch forward function. The instrumented and original CPU heads agreed exactly in that run. The largest probability discrepancy from the saved native reference on this position was 0.1476 percentage points, in an action-WDL component. Real GPU execution is close, not bit-identical. [M1, M2]

@check 1|After Bg7, is "Black's rook has zero legal moves" true? What narrower, useful statement is true?

# 2. Turning this board into 7,680 input numbers
@objectives Construct the 120-channel input; track orientation and history; explain why empty squares still have information; distinguish rule state from time control.

A computer does not start with the idea of a bishop. It starts with a representation chosen by the programmer. In VEGAM, the board is arranged as **64 square rows**, each with **120 feature columns**. One number can say "our bishop is on this square now"; another can say "our bishop was here one halfmove ago." Those meanings are explicit because the encoder writes them that way. [S2, S3]

For one position, the array has shape 64 x 120, or 7,680 numbers. A batch of B positions has shape B x 64 x 120. A shape tells you which questions an index answers. In features[b,s,c], b chooses a position, s chooses a square and c chooses a feature. Changing s moves across the board. Changing c changes which fact you ask about the same square. A tensor is simply an array with several such axes; the term does not add mysterious substance to the numbers.

## 2.1 Squares, coordinates and perspective

Square index is file + 8 times rank, with zero-based file and rank in the network's orientation: a1=0, b1=1, a2=8 and h8=63. Here White is to move, so the usual diagram matches the network coordinates. The bishop on e5 is row 36, the empty g7 square is row 54, the Black rook on h7 is row 55 and the White pawn on h6 is row 47.

When Black is to move, the board's ranks are reflected. A physical square such as h7 is represented at the oriented h2 index; files do not swap. "Our" then means Black and "their" means White. This makes a pawn moving toward promotion advance in the same network direction for either side. It is a representational convenience, not a statement that both positions are always strategically identical: channel 108 still records the absolute side to move, 0 for White and 1 for Black. [S2]

All eight history frames are expressed from the current side's perspective. The encoder alternates its mirror handling as it walks backward through positions whose internal side-to-move orientation alternates. Consequently, "our pawn two plies ago" keeps referring to the current player's pawn. Do not interpret the history frames as eight unrelated boards each using its own color convention. [S3]

## 2.2 The first 104 channels: eight snapshots

Each frame uses thirteen channels in this order: our pawn, knight, bishop, rook, queen, king; their pawn, knight, bishop, rook, queen, king; repetition. Frame h begins at channel 13h. h=0 is the current board, h=1 the previous halfmove, and h=7 the oldest supplied frame. A piece plane contains ones at squares occupied by that piece type and zeros elsewhere. The repetition plane is broadcast across all squares when that historical position has at least one earlier occurrence according to the engine's repetition count.

Consider h6. The White pawn has remained there throughout these eight frames. Its row therefore has ones at channels 0, 13, 26, 39, 52, 65, 78 and 91. This is not eight pawns. It is one pawn observed at eight times. On e5, the bishop is present in the current and immediately preceding frames, so channels 2 and 15 are one. Its earlier locations differ. On h7, the current Black rook gives channel 9 a one; the rook only just arrived there with 80...Rh7, so earlier h7 rook channels are zero. [M2]

History lets identical current piece arrangements produce different inputs. That is useful for reversible moves, repetition and the preceding pawn move. It also creates a reproduction trap: a FEN specifies a current position and counters, not the entire played history. Supplying only this FEN does not reproduce the same neural input as replaying the game's 160 preceding plies.

## 2.3 The remaining sixteen channels

@table channels

Castling channels are square-specific rights markers at the relevant rook's starting square, not merely four broadcast yes/no bits. This preserves the rook identity required by Chess960. In our endgame, all four are zero.

Channel 109 is 17/100=0.17 everywhere. It describes the reversible halfmove clock used by draw rules. It is not White's remaining thinking time, the increment, or the seconds spent searching. Those clocks affect search outside this network input.

Channel 110 is constructed by comparing the previous opponent-pawn plane with the current one to detect a double push and mark its intermediate square. It is not a universal "this en-passant capture is legal" flag: a king-safety restriction can make an apparent capture illegal. The move generator remains responsible for legality. In this position there was no double push and the plane is zero.

The eight history-presence channels distinguish a missing snapshot from a supplied board. With ordinary fill disabled, absent frames are not simply invented by copying the starting board. There is a specific encoder exception when an en-passant marker makes the preceding pawn move reconstructible. Our complete game prefix requires no such inference. Finally, channel 119 is always one: a constant feature available to the learned projection.

@figure feature_rows|MEASURED. Four selected square rows from the actual 64 x 120 input. Isolated vertical marks encode pieces at specific history offsets. The block near the right edge contains shared history-presence and constant features. All values come from the native input dump.

## 2.4 The exact vector for the empty g7 square

Here is the entire nonzero specification of g7's 120-vector: channel 109 is 0.17; channels 111 through 118 are each 1; channel 119 is 1. Every other channel is 0. This is a complete exact description at the human decimal level; float32 stores 0.17 as approximately 0.1700000018.

It may seem surprising that the destination of the beautiful move contains no local bishop signal. That is correct. We are evaluating the board **before** the move. g7 is empty. The current bishop signal belongs to e5. The move decoder will later combine the e5 source representation with the g7 destination representation. Attention and the adaptive pathway can make g7 depend on the rest of the current position before the decoder ever scores that pair.

@figure board_input|MEASURED. The chessboard and the complete list of nonzero g7 input entries. Empty does not mean a zero row: rule state, eight supplied-history flags and the constant channel remain present.

What happens if we actually play Bg7? In the next position the side to move becomes Black. The White bishop becomes an opposing bishop at oriented g2, not an own bishop still written at row g7. The halfmove clock increases, history shifts and legal moves become Black's. A legitimate successor encoding changes all these related pieces of state. Editing one feature bit is an artificial neural intervention, not the same thing as playing a legal move.

## 2.5 Rules supply the candidate moves

Alongside the board tensor, the network receives a 256 x 4 integer array and a 256-element legal mask. Each active row specifies source square, destination square, promotion code and castling code. Unused rows are padding. The promotion codes are 0 for none, 1 knight, 2 queen, 3 rook and 4 bishop. Castling uses 0 for none, 1 kingside and 2 queenside; the internal castling move points the king toward the relevant rook square. [S2, S4]

Before Bg7 there are 22 legal moves. In the recorded order, Bg7 occupies slot 13, with [36,54,0,0]. Bf4 occupies slot 11, with [36,29,0,0]. Slot numbers are administrative positions in a candidate array, not learned move identities. The same move can appear in a different slot in a different call. The decoder gathers source and destination representations using the coordinates, so it is the move's content that matters.

This is an important division of labor. The network does not have to discover the bishop's movement law in order to avoid illegal moves. A rule-based generator supplies legal candidates, and padding logits are masked out. But understanding which *legal* move is good still requires learned relationships and search. Knowing that Bg7 is legal is much easier than understanding what its restriction achieves.

@check 2|If g7 and f4 are both empty and have the same local history, must their initial hidden representations be identical? Name the part of the next stage that can distinguish them.

@check 3|Why would analyzing only the anchor FEN fail to reproduce the exact recorded neural input, even if the pieces and halfmove clock match?

# 3. One square acquires 384 coordinates
@objectives Calculate a linear projection by hand; reproduce a real g7 coordinate; distinguish named input features from learned hidden coordinates; understand what the square embedding contributes.

We now have reliable numbers describing the board. The next problem is flexibility. A 120-entry row has meanings chosen in advance: piece types at particular times, rights, clock and presence. The network needs combinations that can be useful for many downstream tasks. A learned linear projection provides the first set of combinations. [S1: Net.__init__, Net.forward]

## 3.1 Matrix multiplication is many weighted sums

Take a teaching vector x=[1,0,2,1]. Let its coordinates stand for four invented input facts; the particular values are illustrative. Suppose three output coordinates use these weight rows: [2,-1,0.5,0], [-1,1,1,2] and [0,0.5,-1,1]. Let the three biases be [0.1,-0.2,0.3].

Output 0 is 2 times 1 + (-1) times 0 + 0.5 times 2 + 0 times 1 + 0.1 = 3.1. Output 1 is (-1) times 1 + 1 times 0 + 1 times 2 + 2 times 1 - 0.2 = 2.8. Output 2 is 0 times 1 + 0.5 times 0 - 1 times 2 + 1 times 1 + 0.3 = -0.7. The answer is [3.1,2.8,-0.7]. Nothing was classified by name. Each output is a different learned mixing of the inputs.

@equation input_projection

In the real equation, X is 64 x 120; W_in is stored by PyTorch as 384 x 120; b_in has 384 entries. The superscript T means transpose: exchange the two matrix axes. Thus X times W_in^T is 64 x 384. For each square s and hidden coordinate j, we multiply its 120 features by weight row j and sum. We then add the learned bias b_j and the learned position entry E_sj.

The same W_in is used at every square. A bishop feature receives the same initial transformation whether it occurs on e5 or b2, subject to different other features. The separate position table E has shape 1 x 64 x 384: each square has its own learned vector. Broadcasting the leading dimension applies the same table to every position in a batch. Shared feature processing and square-specific geometry are both available.

## 3.2 A real g7 output coordinate, fully accounted for

For coordinate 0 of the actual g7 embedding, only ten input channels contribute because all other features are zero. The following numbers are from the deployed-weight rounding and the saved diagnostic input. [M2]

@table projection

The ten products sum to 0.02997978. Add the projection bias, +0.03479004, and g7's coordinate-0 position entry, -0.06829834. The result is **-0.00352852**, exactly the saved embedding to the shown precision. Coordinate 1 similarly becomes 0.16779648; coordinate 2 becomes 0.31786954. Across all 64 x 384 entries, direct reconstruction differs from the saved tensor by at most about 6 x 10^-8 due to accumulation rounding in this audit.

This is the requested bridge from physical emptiness to a hidden number. g7 does not contain a mysterious evaluation value waiting to be read. Its explicit clock and presence features are multiplied by learned weights, summed with a learned bias, then combined with a learned square vector. That is where the number comes from.

What would adding an own bishop to a row do at this immediate linear stage, if everything else were held fixed? It would add the entire 384-entry weight column for channel 2. That is a precise local sensitivity. It would not yet establish the meaning of that added direction. And a legal move changes source, destination, history, side and possibly rights, so that one-column change is not a complete simulation of moving the bishop.

## 3.3 More coordinates do not mean more raw information

The map expands 120 coordinates to 384, but it cannot conjure missing board facts at a single square. Before nonlinearities or inter-square communication, the square row remains an affine function of those 120 inputs plus the square identity. Why expand it? To provide room for multiple useful combinations and separate downstream projections. One combination can help a gate; another can help a value channel; several can jointly support a relationship after attention.

A scalar is one number. A vector is an ordered list. A matrix is a grid of numbers. Calling the 384-vector a "representation of g7" means it is stored at g7's token slot and currently summarizes the inputs computed there. It does not mean coordinate 0 is "g7 safety" or coordinate 1 is "bishop arrival potential." Input axes have explicit labels; hidden axes acquire their usefulness through optimization. We must measure before assigning a chess name.

Nor is the learned position vector a built-in diagonal detector. It is a table of trainable numbers initialized with small random entries. Repeated useful gradients can make its distinctions helpful for geometry. Architecture permits that; the current table must be examined to see which distinctions actually developed. Even then, hidden-coordinate basis choices can distribute one useful property across many entries.

## 3.4 A representation is allowed to change its job

@figure g7_vectors|MEASURED. The 384 coordinates at g7 after input projection, after block 1 and after block 8, plotted on the same vertical scale. The row stays attached to the same square, but its values become dependent on other squares. Coordinates are not labeled chess concepts.

The input projection knows only the local 120 features and a fixed square vector. After one transformer block, g7 can depend on every supplied square. After later blocks, it can depend on information that those squares have already combined. This changing context is why an empty destination can acquire a rich representation before any move is played.

There is a useful discipline here: always ask both **where an activation is stored** and **what inputs can influence it**. A value stored at g7 after block 8 is not necessarily caused by a piece on g7. Its receptive field is already the whole encoded board and history. In this network the adaptive pathway also reads a flattened summary of all 64 squares, adding another global route.

## 3.5 Normalization prepares the next computation

Before a block forms queries, keys and values, LayerNorm normalizes each 384-entry row independently. It computes that row's mean, subtracts it, divides by its standard deviation with a small stabilizer, then applies a learned per-coordinate scale and offset. The normalization axis is the 384 coordinates, not the 64 squares and not the batch. [S1; R2]

@equation layernorm

Here d=384, mean m is the average of the row's entries, variance v is the average squared deviation with divisor d, epsilon is 10^-5, gamma is the learned scale and beta the learned offset. The normalized intermediate has controlled relative scale. With learned gamma and beta, the final row need not have mean zero or variance one.

For x=[1,3], m=2 and v=1. Ignoring only the tiny epsilon for a first mental calculation, normalization gives [-1,1]. If gamma=[2,0.5] and beta=[0.1,-0.2], the output becomes [-1.9,0.3]. Including epsilon changes the values slightly to approximately [-1.899990,0.299998]. Scale stabilization and learned flexibility coexist.

Normalization does not move a bishop or decide what matters in chess. It provides numerically manageable coordinates for subsequent learned transformations. The original row remains on the residual route. The block reads a normalized copy to compute its update, then adds that update to the unnormalized residual state. We will follow that addition explicitly in Chapter 7.

@check 4|In the four-input example, increase x's third coordinate from 2 to 3. Before any nonlinearity, how much does each output change?

@check 5|LayerNorm is applied to g7. Does it subtract the mean of all 64 squares, or the mean of g7's 384 coordinates? Why does that distinction matter?
