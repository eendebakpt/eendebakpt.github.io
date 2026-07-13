<sub>[← home](../../)</sub>

# A small conjecture, an AI proof, and what Lean did for it

*Pieter Eendebak — July 2026*

In 2023, with Eric Schoen, Alan Vazquez, and Peter Goos, I published a
paper [*Systematic enumeration of two-level even-odd designs of strength 3, CSDA, 2023*](https://doi.org/10.1016/j.csda.2022.107678)
on a family of statistical designs we called *even–odd*. The setting is
two-level orthogonal arrays of strength 3 — basically arrays of $\pm 1$ where
every column is balanced and every triple of columns has all eight sign
patterns equally represented. Each such array has a list of numbers attached
to it, called $J$-characteristics, one per subset of columns; an "even–odd"
design is one where some four-column subset gives a nonzero number, and at
least one odd-size subset does too. The point of the paper was an algorithm
that enumerates these.

While we were running the algorithm, we started classifying even–odd designs
by the smallest odd subset size where the $J$-characteristic is nonzero. We called this
the *type* of the design. A type-5 design has some 5-column subset with
nonzero $J$. A type-7 design has all 5-column $J$'s equal to zero but some
7-column one nonzero. A design of type 9 or higher has both 5- and 7-column
$J$'s vanishing, with the first non-zero odd one at order 9 or higher. What our
enumeration showed was striking: **type-5 designs are everywhere, and type-7
and type-9 designs were essentially absent**.

The numbers below come from Tables 2, 4, and 6 of the 2023 paper. All counts are of
*even-odd* designs: strength-3 $\pm 1$ arrays with $J_4\not\equiv 0$ and at
least one nonzero odd-order $J$. A "—" means not possible, and "**?**" marks cells which are unknown.

**Number of even-odd designs with $J_5\not\equiv 0$ (type 5, abundant):**

| $N$ | $n=5$ | $n=7$ | $n=9$ | $n=11$ | $n=13$ | $n=15$ | $n=17$ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 40 | 1 | 2 | 2 | 0 | 0 | — | — |
| 48 | 3 | 123 | 485 | 717 | 237 | — | — |
| 56 | 3 | 1,243 | 47,868 | 157 | 6 | — | — |
| 64 | 10 | 52,194 | 1.2·10⁸ | 6.0·10⁹ | 2.0·10¹⁰ | ≥3.8·10¹⁰ | ≥2.2·10⁹ |

**Number of even-odd designs with $J_5\equiv 0$ but $J_7\not\equiv 0$ (type 7):**

| $N$ | $n=5$ | $n=7$ | $n=9$ | $n=11$ | $n=13$ | $n=15$ | $n=17$ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 40 | — | 0 | 0 | 0 | 0 | — | — |
| 48 | — | 0 | 0 | 0 | 0 | — | — |
| 56 | — | 0 | 0 | 0 | 0 | — | — |
| 64 | — | 0 | 0 | 0 | 0 | 0 | 0 |

The type-7 counts are all zero, and this is established in the 2023 paper
itself: using the complete strength-3 catalogs of Schoen et al. (2010) and an
extension argument, the only 56- or 64-run strength-3 design with $J_5\equiv 0$
and a nonzero $J_7$-characteristic is the regular $2^{7-1}$ design of
resolution VII at $N=64$ — and that design is not even–odd.

**Number of even-odd designs with $J_5\equiv J_7\equiv 0$ but some odd $J\not\equiv 0$ (type 9 or higher):**

| $N$ | $n=5$ | $n=7$ | $n=9$ | $n=11$ | $n=13$ | $n=15$ | $n=17$ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 40 | — | — | 0 | 0 | 0 | — | — |
| 48 | — | — | 0 | 0 | 0 | — | — |
| 56 | — | — | **?** | **?** | **?** | — | — |
| 64 | — | — | **?** | **?** | **?** | **?** | **?** |

For the remaining cells — the designs of type 9 or higher — we believed that no designs exist:

> **Conjecture (Eendebak, Schoen, Vazquez, Goos, 2023, §5 and §6).** No
> strength-3 even–odd design exists with $N\in\{56,64\}$ runs, at least
> nine factors, $J_5\equiv 0$ and $J_7\equiv 0$. Equivalently,
> the cells marked `?` in the type-9 table above are all zero.

Note that type-7 and type-9 even–odd designs do exist by construction at larger run
sizes — the smallest we know of are $N=80$ (type 7, the union of the even-weight
half-fraction $H_7$ with a 16-run strength-3 fold-over block) and $N=280$
(type 9, $H_9$ union a 24-run block). Those constructions are not part of
the 2023 paper; but at the time we were aware of them.

## Asking Claude

I decided to hand the problem to Claude (Opus 4.7). I gave it the
relevant chunks of the 2023 paper, stated the conjecture, and asked it to either prove or disprove the thing.

After a few back-and-forths Claude came back with
a proposed proof. The core idea was a divisibility bound: a symmetry argument
on the Walsh–Hadamard inversion formula shows that if all lower-order odd
characteristics vanish, the top one must be divisible by a large power of two. To
close the conjecture from this divisibility bound, the first version
relied on an integer linear programming argument that bounded the
row-multiplicity vector and ruled out the remaining cases by numerical
search.

The suggested proof contained novel ideas and the broad picture looked ok, but
it did need reviewing — and reviewing mathematical proofs can be a hard and
tedious task.

## Lean as a stress test

So I asked Claude to translate the argument into [Lean 4](https://lean-lang.org/). Lean is a proof
assistant: you write the proof in a formal language and the computer checks
every inference step against a library of accepted mathematics. If anything
is missing or incorrect, the build fails.

This turned out to be a productive step. The
formalisation surfaced two kinds of issue. Some were small gaps — places
where the prose proof had skipped a step that, when you tried to write it
out for Lean, you discovered actually needed an argument. Those got fixed.
But the more interesting thing was structural. When we had to nail down
what each piece was actually doing, the ILP-based closure collapsed: a
short projection argument turned out to do the same work without any
numerical search. The divisibility statement also got cleaner.

The Lean build of the new proof finishes in about a thousand checked steps. This means that instead of reviewing the full proof, we now only have to verify the conjecture statement was translated properly to Lean.
The complete formalization — from the Walsh–Hadamard identities down to the $N \ge 256$ run-size bound — is available as a single self-contained file, [EvenOddDivisibility.lean](EvenOddDivisibility.lean).

## Human verification

I wrote up the result and sent it to Eric Schoen, co-author on the 2023
paper, for a sanity check from the design-of-experiments side. He confirmed
the proof looked correct, but found the AI-generated paper poorly written.

His review pointed out a list of things: the opening paragraph defined the $J$-characteristics
incorrectly (it asserted equalities with related quantities that do not
hold), the symbol $t$ was used for two unrelated things in the
same lemma, and there was an entire subsection called "Why integrality"
that was just the paper defending itself against approaches no one had
asked about. The proof was also stated more than once, in slightly different
forms, in different sections. None of this affected correctness, but it made
the paper harder to read.

I went back to Claude with Eric's comments and we worked through them. The
final revised version is shorter, the notation more consistent, and overall flow
is better. The revised paper is on arXiv as
[arXiv:2607.09478](http://arxiv.org/abs/2607.09478).

## Final thoughts

The conjecture turned out to have a short and elementary proof. Even though we believed there was such a proof, we were not able to find it and Claude did just that with a few prompts (and a couple of hours of computation).

Prompting Claude to formalize the conjecture and proof in Lean was a good step. It led to both new insights, and more confidence in the validity of the proof.

The AI-written paper was definitely not at the standards of a peer-reviewed journal.
I think here the Claude model can be improved, in particular when trained on published articles
or teaching material. One example: journals discourage numbered equations in the abstract (inline formulas are discouraged too). This is a well-known convention, but Claude used them in the abstract anyway. For reference, the original unrevised AI-written paper is available [here](divisibility_ai_only_paper.pdf).


(disclaimer: this text was written with assistance from Claude)
