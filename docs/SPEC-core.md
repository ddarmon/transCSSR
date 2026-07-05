# transCSSR Core Algorithm Specification

This document specifies the exact algorithmic semantics of the core inference
path in `transCSSR.py` (branch `transcssr-int`, as read from disk). It is the
contract a fresh, token-first reimplementation must satisfy. All line numbers
refer to `transCSSR.py` on this branch.

The core path is:

```
estimate_predictive_distributions   ->  (word_lookup_marg, word_lookup_fut)
run_transCSSR                        ->  (epsilon, invepsilon, morph_by_state)
    (internally: chisquared_test, remove_transients, remove_state,
     get_connected_component, get_transitions, filter_and_predict)
```

CSSR here infers an **epsilon-transducer**: it models
`P(Y_t | X_t, X_past, Y_past)` (memoryful in both input and output). An
epsilon-machine is the degenerate case with a single input symbol
(`axs = ['0']`), so all input conditioning collapses.

---

## 0. Data model, alphabets, and the morph indexing convention

- `axs`: list of input symbols (strings), e.g. `['0','1']`.
- `ays`: list of output symbols (strings), e.g. `['0','1']`.
- `e_symbols`: **always** constructed by callers as
  `list(itertools.product(axs, ays))` (see `demo_transCSSR.py:111`,
  `demo_CSSR.py:70`). This ordering is load-bearing:

  ```
  e_symbols[k] = (axs[x_ind], ays[y_ind])   where   k = len(ays)*x_ind + y_ind
  ```

  i.e. the input index is the *outer* loop, the output index the *inner* loop.

- A **history** (a.k.a. "word", "past") is a tuple `(xhist, yhist)` where
  `xhist` and `yhist` are **equal-length strings** (one character per time step
  in the base case). `('', '')` is the length-0 null history.

- A **morph** (`morph_by_state[state]`, or a per-history `morph_by_history`) is
  a flat list of counts of length `len(e_symbols) == len(axs)*len(ays)`, indexed
  the same way as `e_symbols`:

  ```
  morph[k] = count of emission (axs[x_ind], ays[y_ind])   with   k = len(ays)*x_ind + y_ind
  ```

  So the morph is a **row-stacked stochastic matrix**: row `x_ind` occupies the
  contiguous slice `morph[x_ind*len(ays) : x_ind*len(ays) + len(ays)]` and, once
  normalized within the row, gives `P(Y_t = ays[y_ind] | X_t = axs[x_ind], state)`.
  This convention is used correctly in `filter_and_predict` (line 1993),
  `print_morph_by_states` (line 933), and `draw_dot*`.

  **WARNING:** `chisquared_test` slices this array *inconsistently* (see §2).

State labels are **non-negative integers** assigned from a monotonically
increasing counter `num_states`. `epsilon` maps `(xhist, yhist) -> int state`;
`invepsilon` maps `int state -> dict{ (xhist,yhist): True }` (a set implemented
as a dict); `morph_by_state` maps `int state -> list[int] counts`.

---

## 1. `estimate_predictive_distributions` (line 1073)

Signature:
`estimate_predictive_distributions(stringX, stringY, L_max, counting_method=0, axs=None, ays=None, is_multiline=False, verbose=False)`.

Returns `(word_lookup_marg, word_lookup_fut)`, both `collections.Counter`.

### 1.1 What the two tables mean

- `word_lookup_fut[(xword, yword)]`: count of joint word occurrences where
  `xword` and `yword` have **equal length** ℓ (1 ≤ ℓ ≤ L_max+1). This is the
  count of `(X_{t-ℓ+1..t}, Y_{t-ℓ+1..t})`, i.e. a past **including the current
  output symbol**. Used as the numerator for predictive distributions and to
  build morphs.

- `word_lookup_marg[(xword, yword)]`: count where `xword` is **one character
  longer** than `yword`. `xword` has length ℓ+1, `yword` has length ℓ
  (0 ≤ ℓ ≤ L_max). This is the count of `(X_{t-ℓ..t}, Y_{t-ℓ..t-1})`, i.e. the
  output past plus the input **including the current input** `X_t`. It is the
  denominator for `P(Y_t | X_t, past)` and is used in `run_transCSSR` only as an
  occurrence test (`!= 0`) to decide whether a candidate child history ever
  occurs (lines 1433-1435).

The estimator `P(y_fut | x_fut, xpast, ypast)` is
`#(xpast·x_fut, ypast·y_fut) / #(xpast·x_fut, ypast)` =
`word_lookup_fut / word_lookup_marg`.

### 1.2 `counting_method == 0` (default, lines 1217-1231 single / 1165-1190 multiline)

For a single realization of length `T` (assert `len(stringX)==len(stringY)`):

```
for t_ind in range(T - L_max):                 # start positions 0 .. T-L_max-1
    cur_stringX = stringX[t_ind : t_ind+L_max+1]   # length L_max+1
    cur_stringY = stringY[t_ind : t_ind+L_max+1]   # length L_max+1
    word_lookup_marg[(cur_stringX, cur_stringY[:-1])] += 1   # x len L_max+1, y len L_max
    word_lookup_fut [(cur_stringX, cur_stringY)]      += 1   # both len L_max+1
    for remove_inds in range(1, L_max+1):                    # 1 .. L_max
        trunc_stringX = cur_stringX[:-remove_inds]           # a PREFIX (drops most-recent)
        trunc_stringY = cur_stringY[:-remove_inds]
        word_lookup_marg[(trunc_stringX, trunc_stringY[:-1])] += 1
        word_lookup_fut [(trunc_stringX, trunc_stringY)]      += 1
```

Semantics: each length-`(L_max+1)` window contributes counts for **all of its
prefixes** of every length from 1 to L_max+1 (truncation `[:-remove_inds]`
removes the *most recent* symbols, so the retained sub-word is anchored at the
window start `t_ind`).

**Edge effect (critical):** because sub-words are only counted as prefixes of
windows starting at `t_ind ∈ [0, T-L_max-1]`, a joint word of length ℓ ≤ L_max
is counted exactly `#occurrences starting at positions 0 .. T-L_max-1`.
Occurrences that start in the final `L_max` positions of the string are **never
counted**. Length-`(L_max+1)` words are counted once per occurrence at their
unique start. This right-boundary truncation is deliberate and must be
reproduced exactly.

`axs`/`ays` are **ignored** by method 0.

### 1.3 `counting_method == 1` (lines 1232-1234 single / 1191-1204 multiline, then 1236-1277)

Pass 1 counts **only** the maximal-length windows:

```
for t_ind in range(T - L_max):
    word_lookup_fut[(stringX[t_ind:t_ind+L_max+1], stringY[t_ind:t_ind+L_max+1])] += 1
```

Pass 2 (lines 1236-1277) **requires `axs` and `ays`** (asserted at 1237) and
derives all shorter counts by bottom-up marginalization:

```
histories_by_L = [ list(word_lookup_fut.keys()) ]      # level 0 = length-(L_max+1) words
for L_cur in range(L_max, -1, -1):                      # L_max down to 0
    for (wordX, wordY) in histories_by_L[prev_level]:
        subwordX, subwordY = wordX[:L_cur], wordY[:L_cur]   # PREFIX of length L_cur
        if already-seen(subwordX, subwordY): continue
        mark seen
        c_xy = 0
        for ax in axs:
            c_x = 0
            for ay in ays:
                c = word_lookup_fut.get((subwordX+ax, subwordY+ay), 0)
                c_xy += c ;  c_x += c
            if c_x != 0:
                word_lookup_marg[(subwordX+ax, subwordY)] = c_x       # NOTE: assignment, not +=
        if c_xy != 0:
            word_lookup_fut[(subwordX, subwordY)] = c_xy
del word_lookup_fut[('', '')]      # line 1277: remove the length-0 entry
```

**Equivalence to method 0:** for every key of length ≥ 1 the two methods produce
**identical counts** (the marginalization sums exactly the windows that have the
sub-word as a prefix, which equals method 0's prefix count). They differ only in:

1. Method 1 requires `axs`/`ays`; method 0 does not.
2. Method 1 transiently creates the length-0 `word_lookup_fut[('','')]` (total
   count) and then **deletes it**; method 0 never creates any length-0
   `word_lookup_fut` key (its smallest is length 1). Method 0 *does* create
   length-0-`y` marg keys like `(x, '')` (from `cur_stringY[:-1]` when the
   truncated y has length 1); method 1 creates the same via `(subwordX+ax, '')`
   at `L_cur==0`. After the `del`, both tables have the same key sets.
3. Method 1 uses `Counter.get(...,0)` and single-character concatenation
   `subwordX+ax`, `subwordY+ay` (single-char assumption — see §6).

### 1.4 `is_multiline == True` (lines 1159-1204)

`stringX`, `stringY` are **lists of strings** (one realization per element). The
same per-line loop is run for each line, accumulating into the **shared**
Counters. Method 1's marginalization pass (lines 1236-1277) runs once at the end
over the pooled `word_lookup_fut`. Consequences:

- Counts are pooled across realizations.
- No window ever spans a line boundary (realizations are independent).
- A line with `len < L_max+1` yields `range(T-L_max)` empty (or effectively
  empty) and contributes nothing.
- The right-boundary edge effect of §1.2 applies **per line**.

---

## 2. `chisquared_test` (line 28)

Signature: `chisquared_test(morph1, morph2, axs, ays, alpha=0.001, test_type='chi2')`.
Returns `(test, pvalue)` where `test` is a **bool: True means REJECT** the null
hypothesis that the two morphs come from the same distribution (i.e. the
histories are *different* / should be split), and `pvalue` is `1 - CDF`.

Both morphs are length-`len(axs)*len(ays)` count vectors (row-stacked as in §0).
The test is computed **row-wise per input symbol x, then summed**, because the
object being compared is the conditional stochastic matrix `P(Y|X, past)`:

```
statistic = 0
df_x = 0                                  # number of x-rows that contribute
for row_ind in range(len(axs)):
    tmp1 = array_morph1[row_ind*len(axs) : row_ind*len(axs) + len(ays)]   # SEE BUG BELOW
    tmp2 = array_morph2[row_ind*len(axs) : row_ind*len(axs) + len(ays)]
    n1, n2 = sum(tmp1), sum(tmp2)
    if n1 == 0 or n2 == 0:                 # a row unseen in EITHER morph is skipped entirely
        continue                           #   ("don't hold an unseen (X_t, past) against a state")
    df_x += 1
    theta0 = (tmp1 + tmp2) / (n1 + n2)     # pooled MLE of the row distribution
    for array_ind in range(len(tmp1)):     # over output symbols
        for sample_ind in (0, 1):          # the two morphs
            denom = ns[sample_ind]*theta0[array_ind]
            if denom == 0: continue        # skip cells with zero expected count
            if test_type == 'chi2':
                statistic += (obs - denom)**2 / denom             # Pearson chi-square
            elif test_type == 'G':
                if obs != 0:
                    statistic += obs*log(obs/denom)               # (times 2 at the end)
if test_type == 'G': statistic *= 2
df = df_x * (len(ays) - 1)
test   = statistic > scipy.stats.chi2.ppf(1 - alpha, df)         # reject iff above quantile
pvalue = 1 - scipy.stats.chi2.cdf(statistic, df)
```

- `test_type == 'chi2'` (lines 70-130): Pearson chi-square against pooled
  expected counts `n_i * theta0`, summed over both samples and all cells.
- `test_type == 'G'` (lines 131-185): log-likelihood-ratio / G-statistic,
  `2 * Σ obs·log(obs/expected)` over cells with `obs > 0`.
- **Degrees of freedom are data-dependent:** `df = df_x*(len(ays)-1)`, where
  `df_x` counts only the input rows that are non-empty in **both** morphs. Rows
  where either morph has zero total are skipped (both from the statistic and
  from `df_x`). If `df_x == 0` (no shared non-empty row), `df == 0`; then
  `chi2.ppf(1-alpha, 0)` is `0` and any positive-or-zero statistic behaves
  degenerately — in practice statistic is also 0 so `test = (0 > 0) = False`
  (no rejection). Reimplementers should reproduce the `df_x`-based `df`.
- `alpha` is the significance level; the null is rejected when the statistic
  exceeds the `1-alpha` chi-square quantile at `df` degrees of freedom.

### 2.1 SUSPECTED BUG — row slice stride uses `len(axs)` instead of `len(ays)`

Lines 95-96 and 145-146:

```
tmp_morph1 = array_morph1[row_ind*len(axs) : row_ind*len(axs) + len(ays)]
```

The morph is stacked in blocks of `len(ays)` (one block per input symbol), so
row `row_ind` should start at `row_ind*len(ays)`. The code starts at
`row_ind*len(axs)`. The **block length** `len(ays)` is correct, but the **stride
between blocks** is `len(axs)`.

- When `len(axs) == len(ays)` (e.g. `demo_transCSSR.py`: both `['0','1']`), the
  bug is invisible.
- When `len(axs) == 1` (epsilon-machine, `demo_CSSR.py`), `row_ind` only takes
  value 0, offset 0, so the bug is invisible.
- When `len(axs) > 1` and `len(axs) != len(ays)`, rows are mis-aligned: the test
  compares wrong slices and may read out of the intended block. **A faithful
  reimplementation should use `row_ind*len(ays)`** (the mathematically correct
  indexing) — flag this divergence explicitly, since matching the legacy bug
  byte-for-byte would require replicating `len(axs)`. Recommend implementing the
  correct `len(ays)` stride and documenting the deviation.

---

## 3. `run_transCSSR` (line 1281)

Signature:
`run_transCSSR(word_lookup_marg, word_lookup_fut, L_max, axs, ays, e_symbols, Xt_name, Yt_name, alpha=0.001, test_type='chi2', fname=None, verbose=False, all_digits=False, stringX=None, stringY=None)`.

Returns `(epsilon, invepsilon, morph_by_state)`.

### Stage 0 — setup (lines 1355-1391)

1. Ensure `transCSSR_results/` exists (side effect: `os.mkdir`).
2. `epsilon = {}`, `invepsilon = defaultdict(dict)`, `morph_by_state = {}`.
3. `df = len(axs)*len(ays) - 1` (computed but **unused** — the real df comes from
   `chisquared_test`).
4. `num_states = 0`. Create state 0 holding the null history:
   - `epsilon[('','')] = 0`; `invepsilon[0][('','')] = True`.
   - `morph_by_state[0] = [ word_lookup_fut[(e_symbol[0], e_symbol[1])] for e_symbol in e_symbols ]`
     — the length-1 word counts, i.e. the marginal emission distribution.
   - `num_states = 1`.

### Stage 1 — Homogenize / split (lines 1401-1531)

Grow histories one step **into the past** (prepend), for target lengths
`L_cur+1` where `L_cur` runs `0 .. L_max-1`. So children reach length at most
`L_max`.

Pseudocode (faithful to control flow and mutation order):

```
for L_cur in range(0, L_max):                       # 0 .. L_max-1
    states = list(invepsilon.keys())                # SNAPSHOT (deepcopy) — order = dict insertion order
    for state in states:
        histories = list(invepsilon[state])         # SNAPSHOT (deepcopy)
        for history in histories:
            if len(history[0]) != L_cur: continue    # only grow length-L_cur histories this pass
            all_distinct = True
            for ay in ays:                           # OUTER loop = output symbol
                for ax in axs:                       # INNER loop = input symbol
                    new_history = (ax + history[0], ay + history[1])   # PREPEND one symbol

                    # occurrence test: does this child ever appear (any next input)?
                    does_not_occur = True
                    for ax in axs:                   # <-- REUSES/shadows loop var `ax` (see §6)
                        if word_lookup_marg[(new_history[0]+ax, new_history[1])] != 0:
                            does_not_occur = False
                    if does_not_occur: continue

                    morph_by_history = [ word_lookup_fut[(new_history[0]+e[0], new_history[1]+e[1])]
                                         for e in e_symbols ]
                    test, _ = chisquared_test(morph_by_history, morph_by_state[state], axs, ays,
                                              alpha=alpha, test_type=test_type)
                    if test:                          # REJECT: child differs from its parent state
                        is_matching_state = False
                        if num_states != 1:
                            cur_best_pvalue = -inf ; cur_best_state = -1
                            for altstate in list(invepsilon.keys()):   # SNAPSHOT; order-dependent
                                if altstate == state: continue
                                _, pvalue = chisquared_test(morph_by_history, morph_by_state[altstate],
                                                            axs, ays, alpha=alpha, test_type=test_type)
                                if pvalue > alpha and pvalue > cur_best_pvalue:   # strict >: first wins ties
                                    cur_best_pvalue = pvalue ; cur_best_state = altstate
                            if cur_best_state != -1:                  # matched an existing state
                                is_matching_state = True
                                epsilon[new_history] = cur_best_state
                                invepsilon[cur_best_state][new_history] = True
                                morph_by_state[cur_best_state] += morph_by_history   # elementwise
                        if not is_matching_state:                      # create a brand-new state
                            epsilon[new_history] = num_states
                            invepsilon[num_states][new_history] = True
                            morph_by_state[num_states] = morph_by_history
                            num_states += 1
                    else:                             # DO NOT reject: child belongs to parent
                        all_distinct = False
                        epsilon[new_history] = state
                        invepsilon[state][new_history] = True
                        morph_by_state[state] += morph_by_history      # elementwise
            if all_distinct:
                # every child of `history` landed in a different state; drop `history` itself
                invepsilon[state].pop(history)
                epsilon.pop(history)
                if len(invepsilon[state]) == 0:
                    remove_state(state, epsilon, invepsilon, morph_by_state)
                else:
                    old_morph = [ word_lookup_fut[(history[0]+e[0], history[1]+e[1])] for e in e_symbols ]
                    morph_by_state[state] -= old_morph                 # elementwise
```

Key points:

- **Child generation order:** for each parent history, children are generated in
  order `for ay in ays: for ax in axs:` — output-outer, input-inner. This is the
  order in which new states get created and numbered.
- **Parent morph mutates during the pass:** when a child is assigned to `state`
  (no-reject branch) its counts are added to `morph_by_state[state]`
  immediately, so later children in the same pass are tested against the
  *updated* parent morph. Order matters.
- **`all_distinct` bookkeeping:** a parent history is removed only if *all* its
  children were split off into other states. When removed and the state is not
  emptied, the parent's own morph (`old_morph` = the length-`(len+1)` child
  counts of `history`) is subtracted back out of the state morph (lines
  1520-1525). This subtraction is fragile; the author's comments (lines 1506,
  1543) flag possible over/under-counting.
- **Matching to existing states** uses argmax over p-value among states with
  `pvalue > alpha`, excluding the current `state`. Strict `>` means the
  first-encountered state wins ties — **order-dependent** on
  `list(invepsilon.keys())`.

### Stage 1b — prune short histories (lines 1536-1562)

```
for hist in list(epsilon.keys()):
    if len(hist[0]) < L_max - 1:
        state = epsilon[hist]
        # subtract this history's child-morph out of the state (lines 1545-1547)
        hist_morph = [ word_lookup_fut[(hist[0]+e[0], hist[1]+e[1])] for e in e_symbols ]
        morph_by_state[state] -= hist_morph
        epsilon.pop(hist)
        invepsilon[state].pop(hist)
# then remove any now-empty states
for state in list(invepsilon.keys()):
    if len(invepsilon[state]) == 0:
        remove_state(state, epsilon, invepsilon, morph_by_state)
```

After this, **only histories of length `L_max-1` and `L_max` remain**. (For
`L_max == 1`, `L_max-1 == 0`, so length-0 and length-1 histories remain.)

### Stage 2 — first transient removal (line 1583)

`remove_transients(epsilon, invepsilon, morph_by_state, e_symbols, L_max)` — see
§4. Then `get_connected_component(...)` is called (line 1588) but its return is
**not used** afterward (dead call kept for parity/debug).

### Stage 3 — Determinize (lines 1597-1774)

Repeatedly split states until the transition structure is deterministic
(unifilar): from a given state, every emission symbol that occurs must lead to a
single next state. The loop performs **one split per outer iteration** and
rescans from scratch.

```
recursive = False ; determinize_count = 0
while not recursive:
    recursive = True
    for state in list(invepsilon.keys()):          # order-dependent
        has_split = False
        histories = list(invepsilon[state].keys()) # order-dependent
        for e_symbol in e_symbols:                  # order-dependent
            # 1) find pilot x0 = first history with a non-null transition on e_symbol
            for hist_ind, x0 in enumerate(histories):
                x0_to_state = TRANS(x0, e_symbol)   # see TRANS() below; -1 if none
                if x0_to_state != -1: break
            if hist_ind == len(histories) - 1:      # (see BUG in §6) treated as "none transition"
                continue
            # 2) find first xa (after x0) that transitions to a DIFFERENT non-null state
            to_split = []
            for pilot_ind in range(hist_ind+1, len(histories)):
                xa = histories[pilot_ind]
                xa_to_state = TRANS(xa, e_symbol)
                if xa_to_state != x0_to_state and xa_to_state != -1:
                    to_split.append(xa) ; break
            if pilot_ind == len(histories) - 1:     # (see BUG in §6) treated as "all agree"
                continue
            # 3) collect every later history that transitions like xa
            for new_ind in range(pilot_ind+1, len(histories)):
                xn = histories[new_ind]
                if TRANS(xn, e_symbol) == xa_to_state:
                    to_split.append(xn)
            # 4) if any to split, move them into a fresh state
            if len(to_split) > 0:
                recursive = False ; has_split = True
                morph_by_state[num_states] = [0]*len(e_symbols)
                candidates = dict.fromkeys(to_split, True)     # dedup, preserves order
                for candidate in candidates:
                    epsilon[candidate] = num_states
                    invepsilon[num_states][candidate] = True
                    invepsilon[state].pop(candidate)
                    m = [ word_lookup_fut[(candidate[0]+e[0], candidate[1]+e[1])] for e in e_symbols ]
                    morph_by_state[num_states] += m            # elementwise
                    morph_by_state[state]      -= m            # elementwise
                num_states += 1
                break                                          # restart e_symbol scan
        if has_split: break                                    # restart state scan
    determinize_count += 1
```

`TRANS(hist, e_symbol)` (lines 1642-1645, 1658-1661, 1674-1677) is:

```
if len(hist[0]) == L_max - 1:
    return epsilon.get((hist[0] + e_symbol[0],       hist[1] + e_symbol[1]),       -1)   # grow (len -> L_max)
elif len(hist[0]) == L_max:
    return epsilon.get((hist[0][1:] + e_symbol[0],   hist[1][1:] + e_symbol[1]),   -1)   # shift+append
# (else: variable left undefined -> NameError; unreachable after Stage 1b pruning)
```

Notes:

- Nondeterminism is resolved by choosing the first history (`x0`) as the
  reference; histories whose transition on `e_symbol` matches the *first
  divergent* target `xa_to_state` are moved out together into one new state. A
  third distinct target is handled in a subsequent pass (the whole scan
  restarts after each split).
- `candidates` is deduplicated via dict keys (order preserved).
- Histories that make **no** transition (`-1`) on `e_symbol` are ignored (never
  cause a split and are never moved for that symbol).
- The morph of the new state is *rebuilt* from the moved histories' own child
  counts, and those counts are subtracted from the source state's morph.

### Stage 3b — cleanup (lines 1767-1785)

- Remove any now-empty states (`remove_state`).
- `remove_transients(...)` again, to drop transients introduced by splitting.

### Stage 4 — optional morph re-estimation by filtering (lines 1791-1830)

Only runs if **both** `stringX` and `stringY` are non-`None`. The demos and tests
in this repo **do not** pass them (`demo_transCSSR.py:134`, `demo_CSSR.py:91`,
`test_*`), so by default the count-based morphs from Stages 1-3 are returned
unchanged.

If run:

```
morph_by_state_old = morph_by_state.copy()
morph_by_state = { state: [0]*len(e_symbols) for state in invepsilon }
ays_to_ind = { ay: i for i, ay in enumerate(ays) }
# single realization (str) or list of realizations
filtered_states, _, _ = filter_and_predict(sX, sY, epsilon, invepsilon,
                                            morph_by_state_old, axs, ays, e_symbols, L_max)
for t, state in enumerate(filtered_states):
    if state != -1:
        morph_by_state[state][ ays_to_ind[ sY[t] ] ] += 1
```

**SUSPECTED BUG (§6):** the increment uses index `ays_to_ind[y]` ∈
`[0, len(ays)-1]` only — it does **not** add the `len(ays)*x_ind` offset. So
re-estimated morphs deposit *all* counts into the `x_ind == 0` row and leave
every other input row zero. This is correct only for epsilon-machines
(`len(axs) == 1`); for genuine transducers with `len(axs) > 1` the re-estimated
morph ignores the input entirely. (Not exercised by the shipped demos.)

### Stage 5 — output (lines 1905-1912)

Writes `transCSSR_results/{Xt_name}+{Yt_name}.dot` and `.dat_results` (or
`{fname}` if given), then returns `(epsilon, invepsilon, morph_by_state)`.

### What the return values contain

- `epsilon`: dict `(xhist, yhist) -> int state`, keys limited to histories of
  length `L_max-1` and `L_max` that survived pruning, transient removal, and
  determinization.
- `invepsilon`: dict `int state -> {history: True}` (the inverse partition).
- `morph_by_state`: dict `int state -> list[int]` of length `len(axs)*len(ays)`,
  indexed `len(ays)*x_ind + y_ind`. Unless Stage 4 ran, these are summed child
  counts (integers), **not** normalized probabilities. State labels are integers
  but are **not** guaranteed contiguous (splits and removals leave gaps).

---

## 4. `remove_transients`, `remove_state`, `get_connected_component`, `get_transitions`

### 4.1 `get_connected_component` (line 189)

Builds a directed state-transition graph and returns its **strongly connected
components** plus adjacency structures. Steps:

1. `states = list(invepsilon.keys())`; assign `states_to_index` /
   `index_to_states` by enumeration order (**order-dependent** node indexing).
2. Build `state_matrix` (dense `N×N` 0/1 adjacency) and `trans_dict`
   (`{from_index: {to_index: True}}`). For each state, decide the transition
   source histories:
   - `need_Lmax = True`; if the state has **any** history of length `L_max-1`,
     set `need_Lmax = False`.
   - For each history in the state:
     - if `len == L_max`: only use it (with **left-shift** `hist[0][1:] +
       e_symbol[0]`) when `need_Lmax` is True; otherwise skip it.
     - else (length `L_max-1`): use it with **grow** `hist[0] + e_symbol[0]`.
   - Look up `epsilon.get((...), -1)`; if a real state `s`, set
     `state_matrix[i, states_to_index[s]] = 1` and record in `trans_dict`.

   I.e. transitions are read preferentially off the length-`(L_max-1)` histories
   (append), falling back to length-`L_max` histories (shift+append) only for
   states that have no length-`(L_max-1)` history.
3. `g = Graph.Adjacency(state_matrix.tolist())` builds a **directed** igraph
   graph; `clusters = g.clusters()` returns its components.

**igraph replacement note (critical):** `Graph.clusters()` (the legacy alias for
`connected_components`) defaults to **STRONG** components, and the
`remove_transients` logic below is *only correct for strongly connected
components* — with weak components the cross-component-edge test (below) would
never fire and transient removal would be vacuous. A plain-Python
reimplementation must compute **strongly connected components** (Tarjan or
Kosaraju) over the same directed edge set. `clusters.membership` is a list
mapping node index -> component id; `clusters` is iterable over components. Only
`.membership` (and, indirectly, component sizes) is used downstream.

### 4.2 `remove_transients` (line 768)

```
clusters, _, trans_dict, states_to_index, index_to_states =
    get_connected_component(epsilon, invepsilon, e_symbols, L_max)   # STRONG comps
# Pass 1: any state with an edge leaving its own SCC is transient -> remove it.
remove = {}
for from_idx in trans_dict:
    for to_idx in trans_dict[from_idx]:
        if membership[from_idx] != membership[to_idx]:
            remove[index_to_states[from_idx]] = True
for s in remove: remove_state(s, epsilon, invepsilon, morph_by_state)
# Pass 2: recompute SCCs; keep ONLY the component with the most states.
clusters, ... = get_connected_component(...)
keep_component, _ = map(int, scipy.stats.mode(clusters.membership))   # most frequent membership id
for state_idx, membership_id in enumerate(clusters.membership):
    if membership_id != keep_component:
        remove_state(index_to_states[state_idx], ...)
```

- Pass 1 removes SCC-source states of any cross-component edge (states that can
  leave their recurrent class = transients).
- Pass 2 keeps the **largest recurrent component by node count** and deletes all
  others. `scipy.stats.mode` returns the **smallest** label among size ties, so
  tie-breaking depends on component id numbering (which depends on node/enumeration
  order — **order-dependent**, see §6).
- `memoryless=True` calls `get_connected_component_memoryless`, which **does not
  exist in this file** (latent NameError). The core path always calls with
  `memoryless=False`, so this branch is dead here.
- `scipy.stats.mode` return-shape has changed across SciPy versions
  (array vs scalar, `keepdims`); `list(map(int, mode(...)))` assumes the old
  two-element-iterable form. A reimplementation should just compute the argmax of
  component sizes (smallest id on ties).

### 4.3 `remove_state` (line 859)

```
for hist in invepsilon[state]: epsilon.pop(hist)
invepsilon.pop(state)
morph_by_state.pop(state)
```

Deletes a state and all its histories from all three structures, in place. No
renumbering of remaining states (labels stay sparse).

### 4.4 `get_transitions` (line 975)

Returns `trans_dict` mapping `(from_state, e_symbol) -> to_state` (state labels,
not indices). Same source-history selection rule as `get_connected_component`
(the `need_Lmax` logic, §4.1), but keyed by state label and emission symbol.
`memoryless=True` substitutes `'n'` for the output symbol in both the key and
the lookup (`hist[1][1:] + 'n'`), used by the memoryless filtering path. Used by
`filter_and_predict` to drive state transitions once synchronized.

---

## 5. `filter_and_predict` (line 1914) — ORIGINAL un-adapted version

Signature:
`filter_and_predict(stringX, stringY, epsilon, invepsilon, morph_by_state, axs, ays, e_symbols, L, memoryless=False)`.
Returns `(filtered_states, filtered_probs, stringY_pred)`.

### 5.1 Precompute row-normalized probabilities (lines 1987-1996)

For each state, `prob_by_state[state]` is the length-`len(axs)*len(ays)` vector
whose each input-row slice `[x*len(ays) : x*len(ays)+len(ays)]` is the morph row
normalized to sum 1 (`counts_by_x / sum(counts_by_x)`). Correct
`len(ays)`-strided indexing here (contrast §2). Then
`trans_dict = get_transitions(...)`.

### 5.2 Synchronization and filtering (lines 2003-2088)

- `filtered_states` is prefixed with `L-1` placeholder `-1` values (and
  `filtered_probs` with `0.5`) — the first `L-1` time steps cannot be
  synchronized.
- Initial state at index `L-1`:
  `s0 = epsilon.get((stringX[:L-1], stringY[:L-1]), -1)` (or `'n'*(L-1)` for y
  when `memoryless`). **Special case:** if the machine has exactly one state
  (`len(invepsilon)==1`), `s0` is forced to that single state
  (lines 2018-2019).
- If not yet synchronized (`s0 == -1`), a while-loop (lines 2039-2058) slides the
  window forward one step at a time (`stringX[num_ahead:L+num_ahead]`) trying to
  match a length-`L` history, until synchronized or the string is exhausted
  (`len(filtered_states) < len(stringX)-1` guards against infinite loop). Each
  step appends a state (possibly `-1`) and a prob.
- Main loop (lines 2060-2087) over `ind` from `L-1+num_ahead` to
  `len(stringX)-2`:
  - if currently unsynchronized (`s0 == -1`): try to **re-synchronize** by
    direct lookup of the length-`L` window `epsilon.get((stringX[ind-L+1:ind+1],
    stringY[ind-L+1:ind+1]), -1)`.
  - else: follow the transition `s1 = trans_dict.get((s0, e_symbol), -1)` where
    `e_symbol = (stringX[ind], stringY[ind])`.
  - append `s1`; `filtered_probs` gets `prob_by_state[s1][len(ays)*cur_x_ord + 1]`
    (the probability of output index **1** given the *next* input `stringX[ind+1]`)
    or `0.5` if `s1 == -1`. **Note:** `filtered_probs` is hardwired to output
    index `1` (binary-`Y` assumption; see §6).
  - `s0 = s1`.

### 5.3 Prediction and the `'N'` placeholder (lines 2089-2106)

```
for t_ind in range(len(filtered_states)):
    state = filtered_states[t_ind]
    if state == -1:
        y = 'N'                                            # cannot predict: not synchronized
    else:
        cur_x = stringX[t_ind] ; cur_x_ord = input_lookup[cur_x]
        y = str(ays[ argmax( prob_by_state[state][cur_x_ord*len(ays) : cur_x_ord*len(ays)+len(ays)] ) ])
    Y_pred.append(y)
stringY_pred = ''.join(Y_pred)
```

The MAP output for each time step is the argmax of the state's normalized morph
row for the observed input `stringX[t_ind]`. When the state is `-1` (never
synchronized / desync), the predicted symbol is the literal character `'N'`.

### 5.4 Return values

1. `filtered_states`: list (length `len(stringX)-1`) of integer state labels per
   time step, with `-1` for unsynchronized steps.
2. `filtered_probs`: list of `P(Y_t = ays[1] | X_t, state)` per step (0.5 when
   unsynchronized). **Binary-output convention** (index 1).
3. `stringY_pred`: string of MAP-predicted outputs, `'N'` where unsynchronized.

- `memoryless=True` replaces the output part of every history lookup with `'n'`
  characters and uses the memoryless `trans_dict` keys.
- **`argmax` tie-break:** `numpy.argmax` returns the **first** maximal index, so
  ties are resolved toward the earlier `ays` symbol (order-dependent).

---

## 6. CRITICAL: order-dependence, single-character assumptions, dead branches, and bugs

### 6.1 Order-dependence / nondeterminism (two correct impls can diverge)

These make the concrete **state numbering** and even **which histories land in
which state** depend on iteration order. A reimplementation must replicate
CPython dict/`Counter` **insertion order** (stable since 3.7) to reproduce the
exact `epsilon`/`invepsilon`/`morph_by_state`:

1. **State creation order** in Stage 1 follows `for ay in ays: for ax in axs:`
   (output-outer, input-inner) child generation (lines 1427-1428), across states
   iterated in `list(invepsilon.keys())` insertion order (line 1408) and
   histories in insertion order (line 1413).
2. **Existing-state matching** (lines 1465-1474) is an argmax over p-value using
   strict `>`, so the **first** state encountered wins ties; iteration order is
   `list(invepsilon.keys())`.
3. **Parent morph mutates mid-pass** (lines 1486, 1509, 1525), so test outcomes
   for later children depend on earlier ones within the same pass.
4. **Determinization** (Stage 3) iterates states, histories, and `e_symbols` in
   insertion/list order; the pilot `x0` and first-divergent `xa` are whichever
   come first (lines 1625-1680). Different orders yield different splits and
   numbering.
5. **`get_connected_component` node indexing** is enumeration order of
   `invepsilon.keys()` (line 254), which sets igraph node ids and hence SCC ids.
6. **`remove_transients` "largest component"** tie-break via `scipy.stats.mode`
   keeps the **smallest** SCC id among equal-size components (line 849) —
   depends on 5.
7. **`numpy.argmax`** in `filter_and_predict` (line 2102) breaks ties toward the
   earliest `ays` symbol.

### 6.2 Single-character-symbol assumptions (break for multi-char tokens)

The code assumes every input/output symbol is exactly one character. A
token-first reimplementation replacing strings with sequences of arbitrary-width
tokens must revisit every one of these:

- `estimate_predictive_distributions`: window slicing `stringX[t_ind:t_ind+L_max+1]`,
  prefix truncation `[:-remove_inds]`, `cur_stringY[:-1]` (lines 1177-1234); method 1
  concatenation `subwordX + ax`, `subwordY + ay` (lines 1264-1275) and `del
  word_lookup_fut['','']`.
- `run_transCSSR`: `ax + history[0]`, `ay + history[1]` prepend (line 1429);
  `new_history[0] + ax` (line 1434); morph key building `'{}{}'.format(hist, e)`
  everywhere; `len(history[0]) == L_cur` length tests (lines 1417, 1539, 1642…);
  `hist[0][1:]` left-shift (lines 1645, 1661, 1677).
- `get_connected_component` / `get_transitions`: `hist[0][1:] + e_symbol[0]`,
  `hist[0] + e_symbol[0]` (lines 272-284, 1044-1062); `len(hist[0]) == L_max`
  tests.
- `filter_and_predict`: `stringX[:L-1]`, window slices, `'n'*(L-1)`
  (lines 2006-2068).

All history-length logic uses `len(<string>)` as a **time-step count**, valid
only when 1 char = 1 token.

### 6.3 Suspected bugs / dead code (flagged, not fixed)

1. **`chisquared_test` wrong row stride** (`row_ind*len(axs)`; lines 95-96,
   145-146). Should be `row_ind*len(ays)`. Latent whenever
   `len(axs) not in {1, len(ays)}`. (§2.1)
2. **Stage 4 morph re-estimation ignores input** (line 1811/1823):
   `morph_by_state[state][ays_to_ind[y]] += 1` omits the `len(ays)*x_ind`
   offset, collapsing all inputs into row 0. Wrong for `len(axs) > 1`. Not
   triggered by shipped demos (they pass no `stringX`/`stringY`). (§3 Stage 4)
3. **Determinization off-by-one "found at last index" conflation**
   (lines 1650, 1668): `if hist_ind == len(histories)-1: pass` cannot
   distinguish "a transition was found only at the last history" from "no
   history transitions". If **only the last** history transitions on a symbol it
   is silently skipped as a pilot. Similarly `pilot_ind == len(histories)-1`.
   Edge effect on small states.
4. **`get_connected_component_memoryless` is undefined** in this file, so
   `remove_transients(..., memoryless=True)` and `get_transitions(...,
   memoryless=True)`-adjacent paths would `NameError`. Dead in the core path
   (always `memoryless=False`).
5. **Shadowed loop variable `ax`** in Stage 1 occurrence test (lines 1428 vs
   1433): the inner `for ax in axs` clobbers the outer `ax`. Functionally
   harmless (the outer `for` rebinds `ax` each iteration and `new_history` is
   already computed), but a genuine code smell to preserve carefully.
6. **`filter_and_probs` binary-output assumption** (line 2032/2056): filtered
   probability is hardwired to output index `1`
   (`prob_by_state[...][len(ays)*cur_x_ord + 1]`), only meaningful for binary
   `Y`. (`stringY_pred` via argmax is general; `filtered_probs` is not.)
7. **`df` in `run_transCSSR`** (line 1378) is computed but never used.
8. **Dead `get_connected_component` call** at line 1588 (return value unused).
9. **Fragile morph subtraction bookkeeping** in Stage 1/1b (lines 1520-1525,
   1545-1547), flagged by the author's own comments (lines 1506, 1543) as a
   suspected over/under-count source when histories are pruned.
10. **`scipy.stats.mode` API drift** (line 849): assumes the legacy
    two-value-iterable return; newer SciPy returns a scalar `ModeResult` and may
    break `list(map(int, ...))`.

---

## 7. Minimal reimplementation checklist

1. Represent histories as `(x_tokens, y_tokens)` sequences; do not rely on
   1-char = 1-token.
2. Reproduce `estimate_predictive_distributions` counts exactly, including the
   right-boundary edge effect and the `word_lookup_marg` = (x-including-current,
   y-past) shape. Methods 0 and 1 must agree on all length-≥1 keys.
3. Implement `chisquared_test` row-wise per input symbol with **`len(ays)`
   stride** (the correct indexing), data-dependent `df = df_x*(len(ays)-1)`,
   skipping rows unseen in either morph, supporting `chi2` and `G`.
4. Follow the exact Stage 1 → 1b → transient-removal → determinize → cleanup →
   transient-removal order, preserving insertion-order iteration and the
   argmax/first-wins tie-breaks, to match state numbering.
5. Replace igraph with **strongly connected components** (Tarjan/Kosaraju) over
   the directed transition graph built by the `need_Lmax` source-history rule;
   keep the largest recurrent SCC (smallest id on ties) after removing
   cross-SCC-source (transient) states.
6. Return integer-labeled (possibly non-contiguous) states, with
   `morph_by_state` indexed `len(ays)*x_ind + y_ind`.
