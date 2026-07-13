/-
================================================================================
  A machine-checked proof of the main result of

    P. T. Eendebak,
    "A divisibility theorem for odd J-characteristics of two-level designs",
    arXiv:XXXX.XXXXX (2026).

  The development is self-contained (Lean 4 + Mathlib only): it compiles with
  no `sorry` and no axioms beyond Lean's standard three (propext,
  Classical.choice, Quot.sound).

  MODEL. The design space is `Point n := Fin n → ZMod 2`, where coordinate
  value 0 encodes the level +1 and value 1 encodes -1:

    sign x i  := if x i = 0 then 1 else -1     (the ±1 value of coordinate i)
    chi s x   := ∏ i ∈ s, sign x i             (interaction monomial χ_s)
    jc f s    := ∑ x, f x * chi s x            (signed J-characteristic j(s))
    flipAll x := fun i => x i + 1              (global sign reversal x ↦ -x)

  A design with N runs is a nonnegative integer function f with N = ∑ f.

  ROADMAP (theorem names ↔ paper statements):
    chi_orth            — Lemma 1  (orthogonality of the characters)
    inversion           — Lemma 2  (Walsh–Hadamard inversion)
    foldover            — the fold-over identity behind the key equation
    two_pow_dvd_top     — Theorem 3 for the full coordinate set s = [n]
    jc_proj             — the projection lemma (sub-characteristics survive
                          marginalization onto a subset of the factors)
    two_pow_dvd_subset  — Theorem 3, per-subset form: 2^(|s|-1) ∣ j(s)
    run_size_bound      — Corollary 4: j(s) ≠ 0 forces N ≥ 2^(|s|-1)
    min_odd_bound       — the minimal-odd-order argument
    esvg_bound          — the main corollary: a strength-3 design with
                          J₅ ≡ J₇ ≡ 0 and any nonzero odd-order
                          J-characteristic has N ≥ 2^8 = 256 runs; this
                          settles the Eendebak–Schoen–Vazquez–Goos (2023)
                          conjecture uniformly in the number of factors.

  READING GUIDE. Each declaration is written so that its *statement* is
  everything up to and including `:= by`, and its *proof* is the indented
  tactic block that follows, whose first line is the marker `-- Proof:`.
  To trust the result one only needs to check that the definitions above and
  the statement of `esvg_bound` faithfully translate the paper; the proofs
  are checked by the Lean kernel.
================================================================================
-/

import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Finset.Sort
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Piecewise
import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Order.GroupWithZero.Basic
import Mathlib.Algebra.Ring.Parity
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Common

open Finset BigOperators

namespace EoDivisibility

/-! ### Definitions -/

/-- A point of the two-level design space: a sign pattern on `n` coordinates,
encoded with `ZMod 2` (0 ↦ +1, 1 ↦ -1). -/
abbrev Point (n : ℕ) := Fin n → ZMod 2

variable {n : ℕ}

/-- The ±1 sign of coordinate `i` of a point. -/
def sign (x : Point n) (i : Fin n) : ℤ := if x i = 0 then 1 else -1

/-- The interaction monomial (character) for a subset `s` of coordinates. -/
def chi (s : Finset (Fin n)) (x : Point n) : ℤ := ∏ i ∈ s, sign x i

/-- The signed J-characteristic of a function `f` on the design space. -/
def jc (f : Point n → ℤ) (s : Finset (Fin n)) : ℤ := ∑ x, f x * chi s x

/-- Global sign reversal: flip every coordinate. -/
def flipAll (x : Point n) : Point n := fun i => x i + 1

/-! ### Elementary sign lemmas -/

/-- Statement: every element of `ZMod 2` is either `0` or `1`. -/
theorem zmod_two_cases (a : ZMod 2) : a = 0 ∨ a = 1 := by
  -- Proof:
  revert a; decide

/-- Statement: the sign of a coordinate squares to `1` (it is `±1`). -/
theorem sign_sq (x : Point n) (i : Fin n) : sign x i * sign x i = 1 := by
  -- Proof:
  unfold sign
  rcases zmod_two_cases (x i) with h | h <;> simp [h]

/-- Statement: flipping a coordinate negates its sign. -/
theorem sign_flipAll (x : Point n) (i : Fin n) :
    sign (flipAll x) i = - sign x i := by
  -- Proof:
  unfold sign flipAll
  rcases zmod_two_cases (x i) with h | h <;> rw [h] <;> decide

/-! ### Behaviour of the character under a global flip -/

/-- Statement: under a global sign reversal the character picks up
`(-1)^|s|`. -/
theorem chi_flipAll (s : Finset (Fin n)) (x : Point n) :
    chi s (flipAll x) = (-1) ^ s.card * chi s x := by
  -- Proof:
  unfold chi
  rw [← Finset.prod_const, ← Finset.prod_mul_distrib]
  apply Finset.prod_congr rfl
  intro i _
  rw [sign_flipAll]
  ring

/-! ### Orthogonality of the characters (Lemma 1 of the paper) -/

/-- Statement: a single orthogonality factor `sign x i * sign y i + 1` equals
`2` when the coordinates agree and `0` when they differ. -/
theorem factor_eq (x y : Point n) (i : Fin n) :
    sign x i * sign y i + (1 : ℤ) = if x i = y i then 2 else 0 := by
  -- Proof:
  unfold sign
  rcases zmod_two_cases (x i) with hx | hx <;>
    rcases zmod_two_cases (y i) with hy | hy <;>
      simp [hx, hy]

/-- Statement (Lemma 1): orthogonality of characters,
`∑ s, chi s x * chi s y = if x = y then 2^n else 0`. -/
theorem chi_orth (x y : Point n) :
    (∑ s : Finset (Fin n), chi s x * chi s y) = if x = y then (2 : ℤ) ^ n else 0 := by
  -- Proof:
  -- Each character product is a product over the subset.
  have hprod : ∀ s : Finset (Fin n),
      chi s x * chi s y = ∏ i ∈ s, (sign x i * sign y i) := by
    intro s; unfold chi; rw [← Finset.prod_mul_distrib]
  simp_rw [hprod]
  -- Convert the sum over all subsets to a single product via `Finset.prod_add`.
  have hpow :
      (∑ s : Finset (Fin n), ∏ i ∈ s, (sign x i * sign y i))
        = ∏ i : Fin n, (sign x i * sign y i + 1) := by
    rw [Finset.prod_add]
    rw [Finset.powerset_univ]
    apply Finset.sum_congr rfl
    intro s _
    -- the constant-`1` product over `univ \ s` is `1`
    have : (∏ _i ∈ (univ \ s), (1 : ℤ)) = 1 := by simp
    rw [this, mul_one]
  rw [hpow]
  -- Now rewrite each factor and split on `x = y`.
  simp_rw [factor_eq]
  by_cases hxy : x = y
  · subst hxy
    simp only [if_true]
    rw [Finset.prod_const, Finset.card_univ, Fintype.card_fin]
  · rw [if_neg hxy]
    -- there is a coordinate where they differ; that factor is 0
    obtain ⟨i, hi⟩ := Function.ne_iff.mp hxy
    apply Finset.prod_eq_zero (Finset.mem_univ i)
    rw [if_neg hi]

/-! ### Fourier (Walsh–Hadamard) inversion (Lemma 2 of the paper) -/

/-- Statement (Lemma 2): Fourier inversion, `2^n * f x = ∑ s, jc f s * chi s x`. -/
theorem inversion (f : Point n → ℤ) (x : Point n) :
    (2 : ℤ) ^ n * f x = ∑ s : Finset (Fin n), jc f s * chi s x := by
  -- Proof:
  -- expand jc and swap the order of summation
  unfold jc
  have hexpand :
      (∑ s : Finset (Fin n), (∑ y, f y * chi s y) * chi s x)
        = ∑ s : Finset (Fin n), ∑ y : Point n, (f y * chi s y * chi s x) := by
    apply Finset.sum_congr rfl; intro s _; rw [Finset.sum_mul]
  rw [hexpand, Finset.sum_comm]
  have hfactor :
      (∑ y : Point n, ∑ s : Finset (Fin n), f y * chi s y * chi s x)
        = ∑ y : Point n, f y * (∑ s : Finset (Fin n), chi s y * chi s x) := by
    apply Finset.sum_congr rfl; intro y _
    rw [Finset.mul_sum]; apply Finset.sum_congr rfl; intro s _; ring
  rw [hfactor]
  have horth : ∀ y : Point n,
      (∑ s : Finset (Fin n), chi s y * chi s x)
        = if y = x then (2 : ℤ) ^ n else 0 := fun y => chi_orth y x
  simp_rw [horth]
  rw [Finset.sum_congr rfl (g := fun y => if y = x then f y * (2:ℤ)^n else 0)
        (by intro y _; rw [mul_ite, mul_zero])]
  rw [Finset.sum_ite_eq' Finset.univ x (fun y => f y * (2:ℤ)^n)]
  simp [mul_comm]

/-! ### The fold-over identity and the top-set divisibility theorem -/

/-- The all-`+1` design point (every coordinate `0`). -/
def x0 : Point n := fun _ => 0

/-- Statement: the character of the full coordinate set at `x0` is `1`. -/
theorem chi_univ_x0 : chi (Finset.univ) (x0 : Point n) = 1 := by
  -- Proof:
  unfold chi x0 sign
  apply Finset.prod_eq_one
  intro i _
  simp

/-- Statement (fold-over identity): combining Fourier inversion at `x` and at
`flipAll x` (via `chi_flipAll`), every coordinate-set contributes
`jc f s * (1 - (-1)^|s|) * chi s x`. -/
theorem foldover (f : Point n → ℤ) (x : Point n) :
    (2 : ℤ) ^ n * (f x - f (flipAll x))
      = ∑ s : Finset (Fin n), jc f s * (1 - (-1) ^ s.card) * chi s x := by
  -- Proof:
  have hx := inversion f x
  have hfx := inversion f (flipAll x)
  have hfx' : (2 : ℤ) ^ n * f (flipAll x)
      = ∑ s : Finset (Fin n), jc f s * ((-1) ^ s.card * chi s x) := by
    rw [hfx]; apply Finset.sum_congr rfl; intro s _; rw [chi_flipAll]
  rw [mul_sub, hx, hfx', ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro s _
  ring

/-- **Divisibility theorem, top-set form** (Theorem 3 of the paper for
`s = [n]`).

Statement: let `n` be odd. If every odd-cardinality *proper* subset of
coordinates has vanishing J-characteristic, then `2^(n-1)` divides the top
J-characteristic `jc f univ`.

NOTE.  The hypothesis `Odd n` is genuinely necessary: for even `n` the statement
is false.  Explicit counterexample at `n = 4`: there is an integer function `f`
with every odd-cardinality (proper) J-characteristic equal to `0` for which
`jc f univ = -10`, which is not divisible by `2^(4-1) = 8`.  (This is visible
from the foldover identity, whose `univ`-coefficient `1 - (-1)^n` equals `2`
exactly when `n` is odd and `0` when `n` is even.) -/
theorem two_pow_dvd_top (f : Point n → ℤ) (hodd_n : Odd n)
    (hodd : ∀ s : Finset (Fin n), Odd s.card → s ≠ Finset.univ → jc f s = 0) :
    (2 : ℤ) ^ (n - 1) ∣ jc f Finset.univ := by
  -- Proof:
  obtain ⟨k, hk⟩ := id hodd_n
  have hn : 1 ≤ n := by omega
  -- In the foldover sum, every term except `s = univ` vanishes.
  have hterm : ∀ s : Finset (Fin n), s ∈ (Finset.univ : Finset (Finset (Fin n))) →
      s ≠ Finset.univ →
      jc f s * (1 - (-1) ^ s.card) * chi s (x0 : Point n) = 0 := by
    intro s _ hsne
    rcases Nat.even_or_odd s.card with hev | hod
    · have : ((-1 : ℤ)) ^ s.card = 1 := hev.neg_one_pow
      rw [this]; ring
    · rw [hodd s hod hsne]; ring
  have hsum :
      (∑ s : Finset (Fin n), jc f s * (1 - (-1) ^ s.card) * chi s (x0 : Point n))
        = jc f Finset.univ * (1 - (-1) ^ (Finset.univ : Finset (Fin n)).card)
            * chi Finset.univ (x0 : Point n) := by
    rw [Finset.sum_eq_single (Finset.univ : Finset (Fin n))]
    · intro b _ hb; exact hterm b (Finset.mem_univ b) hb
    · intro h; exact absurd (Finset.mem_univ _) h
  have hcard : (Finset.univ : Finset (Fin n)).card = n := by
    rw [Finset.card_univ, Fintype.card_fin]
  have hpow : ((-1 : ℤ)) ^ ((Finset.univ : Finset (Fin n)).card) = -1 := by
    rw [hcard]; exact Odd.neg_one_pow hodd_n
  have hkey := foldover f (x0 : Point n)
  rw [hsum, hpow, chi_univ_x0] at hkey
  set D : ℤ := f (x0 : Point n) - f (flipAll (x0 : Point n)) with hD
  have h2 : (2 : ℤ) ^ n * D = jc f Finset.univ * 2 := by
    rw [hkey]; ring
  have hsplit : (2 : ℤ) ^ n = 2 * 2 ^ (n - 1) := by
    conv_lhs => rw [show n = (n - 1) + 1 by omega]
    rw [pow_succ]; ring
  rw [hsplit] at h2
  have hfinal : jc f Finset.univ = 2 ^ (n - 1) * D := by
    have heq : (2 : ℤ) * jc f Finset.univ = 2 * (2 ^ (n - 1) * D) := by
      have : (2 : ℤ) * (2 ^ (n - 1) * D) = jc f Finset.univ * 2 := by
        rw [← h2]; ring
      rw [this]; ring
    have h2ne : (2 : ℤ) ≠ 0 := by norm_num
    exact mul_left_cancel₀ h2ne heq
  exact ⟨D, hfinal⟩

/-! ### Projection onto a subset of the coordinates -/

/-- Restriction of a point of `{-1,1}^n` to the coordinates in `s` (with
`s.card = q`), reindexed to `Fin q` through the order embedding
`Finset.orderEmbOfFin` that enumerates `s` in increasing order. -/
def restrict (s : Finset (Fin n)) {q : ℕ} (hq : s.card = q) (x : Point n) :
    Point q :=
  fun j => x (s.orderEmbOfFin hq j)

/-- The projection of `f` onto the coordinates in `s`: the value at a point
`y` of the small cube is the sum of `f` over the fiber of `restrict` above
`y` (the "marginal" of the design on the factors in `s`). -/
def proj (f : Point n → ℤ) (s : Finset (Fin n)) {q : ℕ} (hq : s.card = q) :
    Point q → ℤ :=
  fun y => ∑ x : Point n, if restrict s hq x = y then f x else 0

/-- The character of a reindexed subset, evaluated at `x`, is the character of
the original subset evaluated at the restriction of `x`. -/
theorem chi_restrict (s : Finset (Fin n)) {q : ℕ} (hq : s.card = q)
    (u : Finset (Fin q)) (x : Point n) :
    chi (u.map (s.orderEmbOfFin hq).toEmbedding) x = chi u (restrict s hq x) := by
  -- Proof:
  unfold chi
  rw [Finset.prod_map]
  rfl

/-- **Projection lemma** (Lemma 5 of the paper). The J-characteristics of the
projected function are J-characteristics of the original function: for
`u ⊆ Fin q`, `jc (proj f s hq) u = jc f (image of u in s)`. -/
theorem jc_proj (f : Point n → ℤ) (s : Finset (Fin n)) {q : ℕ} (hq : s.card = q)
    (u : Finset (Fin q)) :
    jc (proj f s hq) u = jc f (u.map (s.orderEmbOfFin hq).toEmbedding) := by
  -- Proof:
  unfold jc proj
  -- expand the fiber sums and exchange the order of summation
  simp_rw [Finset.sum_mul, ite_mul, zero_mul]
  rw [Finset.sum_comm]
  -- collapse the inner sum over `y` to the single term `y = restrict s hq x`
  simp_rw [Finset.sum_ite_eq, Finset.mem_univ, if_true]
  -- identify the characters through `chi_restrict`
  exact Finset.sum_congr rfl fun x _ => by rw [chi_restrict]

/-! ### The divisibility theorem, per-subset form (Theorem 3 of the paper) -/

/-- **Divisibility of odd sub-characteristics.** Let `s` be a subset of the
coordinates of odd cardinality. If every odd-cardinality subset `u ⊆ s` with
`u ≠ s` has vanishing J-characteristic, then `2^(|s|-1)` divides `jc f s`.

This is obtained by applying the top-set divisibility theorem
`two_pow_dvd_top` to the projection of `f` onto `s`. -/
theorem two_pow_dvd_subset (f : Point n → ℤ) (s : Finset (Fin n))
    (hodd : Odd s.card)
    (hvan : ∀ u : Finset (Fin n), u ⊆ s → Odd u.card → u ≠ s → jc f u = 0) :
    (2 : ℤ) ^ (s.card - 1) ∣ jc f s := by
  -- Proof:
  have hq : s.card = s.card := rfl
  -- every odd proper subset of the projected cube has vanishing jc
  have hvan' : ∀ u : Finset (Fin s.card), Odd u.card → u ≠ Finset.univ →
      jc (proj f s hq) u = 0 := by
    intro u huodd hune
    rw [jc_proj]
    apply hvan
    · -- the image of `u` is a subset of `s`
      have hsub : u.map (s.orderEmbOfFin hq).toEmbedding ⊆
          Finset.univ.map (s.orderEmbOfFin hq).toEmbedding :=
        Finset.map_subset_map.mpr (Finset.subset_univ u)
      rwa [Finset.map_orderEmbOfFin_univ] at hsub
    · -- the image of `u` has the same (odd) cardinality
      rwa [Finset.card_map]
    · -- the image of `u` is not all of `s` (else `u` would be `univ`)
      intro hcontra
      apply hune
      have hcard : u.card = s.card := by
        have h := congrArg Finset.card hcontra
        rwa [Finset.card_map] at h
      apply (Finset.card_eq_iff_eq_univ u).mp
      rw [hcard, Fintype.card_fin]
  -- apply the top-set divisibility theorem on the projected cube
  have key := two_pow_dvd_top (proj f s hq) hodd hvan'
  rwa [jc_proj, Finset.map_orderEmbOfFin_univ] at key

/-! ### The run-size bound (Corollary 4 of the paper) -/

/-- The character takes values of absolute value `1`. -/
theorem abs_chi (s : Finset (Fin n)) (x : Point n) : |chi s x| = 1 := by
  -- Proof:
  unfold chi
  rw [Finset.abs_prod]
  apply Finset.prod_eq_one
  intro i _
  unfold sign
  rcases zmod_two_cases (x i) with h | h <;> simp [h]

/-- For a nonnegative `f` (a design with `N = ∑ f` runs), every
J-characteristic is bounded by `N` in absolute value. -/
theorem abs_jc_le_sum (f : Point n → ℤ) (hpos : ∀ x, 0 ≤ f x)
    (s : Finset (Fin n)) :
    |jc f s| ≤ ∑ x : Point n, f x := by
  -- Proof:
  unfold jc
  calc |∑ x : Point n, f x * chi s x|
      ≤ ∑ x : Point n, |f x * chi s x| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ x : Point n, f x := by
        apply Finset.sum_congr rfl
        intro x _
        rw [abs_mul, abs_chi, mul_one, abs_of_nonneg (hpos x)]

/-- **Run-size bound** (Corollary 4 of the paper). Under the hypotheses of
`two_pow_dvd_subset`, if moreover `f ≥ 0` and `jc f s ≠ 0`, then the design
has at least `2^(|s|-1)` runs. -/
theorem run_size_bound (f : Point n → ℤ) (hpos : ∀ x, 0 ≤ f x)
    (s : Finset (Fin n)) (hodd : Odd s.card)
    (hvan : ∀ u : Finset (Fin n), u ⊆ s → Odd u.card → u ≠ s → jc f u = 0)
    (hne : jc f s ≠ 0) :
    (2 : ℤ) ^ (s.card - 1) ≤ ∑ x : Point n, f x := by
  -- Proof:
  have hdvd := two_pow_dvd_subset f s hodd hvan
  have h1 : (2 : ℤ) ^ (s.card - 1) ≤ |jc f s| :=
    Int.le_of_dvd (abs_pos.mpr hne) ((dvd_abs _ _).mpr hdvd)
  exact h1.trans (abs_jc_le_sum f hpos s)

/-! ### The main corollary (Corollary 6 of the paper) -/

/-- **Main theorem (minimal-odd-order form).** Let `f ≥ 0` be the
row-multiplicity function of an `N`-run design on `n` two-level factors,
`N = ∑ f`. Suppose every odd-order J-characteristic of order `< 9` vanishes
(strength 3 gives orders 1 and 3; `J₅ ≡ 0` and `J₇ ≡ 0` give orders 5 and 7).
If the design has ANY nonzero odd-order J-characteristic, then `N ≥ 2^8 = 256`. -/
theorem min_odd_bound (f : Point n → ℤ) (hpos : ∀ x, 0 ≤ f x)
    (hvan : ∀ u : Finset (Fin n), Odd u.card → u.card < 9 → jc f u = 0)
    (hex : ∃ s : Finset (Fin n), Odd s.card ∧ jc f s ≠ 0) :
    (2 : ℤ) ^ 8 ≤ ∑ x : Point n, f x := by
  -- Proof:
  classical
  -- take a subset `s` of MINIMAL odd cardinality with `jc f s ≠ 0`
  have hP : ∃ q : ℕ, ∃ s : Finset (Fin n), s.card = q ∧ Odd s.card ∧
      jc f s ≠ 0 := by
    obtain ⟨s, h1, h2⟩ := hex
    exact ⟨s.card, s, rfl, h1, h2⟩
  obtain ⟨s, hcard, hodd, hne⟩ := Nat.find_spec hP
  -- minimality: every odd subset of strictly smaller cardinality vanishes
  have hmin : ∀ u : Finset (Fin n), Odd u.card → u.card < Nat.find hP →
      jc f u = 0 := by
    intro u huodd hult
    by_contra hcontra
    exact Nat.find_min hP hult ⟨u, rfl, huodd, hcontra⟩
  -- the minimal odd order is at least 9
  have h9 : 9 ≤ Nat.find hP := by
    by_contra h
    exact hne (hvan s hodd (by omega))
  -- every odd proper subset of `s` vanishes (its cardinality is smaller)
  have hvs : ∀ u : Finset (Fin n), u ⊆ s → Odd u.card → u ≠ s →
      jc f u = 0 := by
    intro u hsub huodd hune
    have hlt : u.card < s.card := Finset.card_lt_card (hsub.ssubset_of_ne hune)
    exact hmin u huodd (by omega)
  -- run-size bound at `s`, then monotonicity of the power
  have hbound := run_size_bound f hpos s hodd hvs hne
  have hpow : (2 : ℤ) ^ 8 ≤ (2 : ℤ) ^ (s.card - 1) :=
    pow_le_pow_right₀ (by norm_num) (by omega)
  linarith

/-- **Resolution of the ESVG conjecture (uniform in `n` and `N`).**
Any two-level design (`f ≥ 0`, `N = ∑ f` runs) of strength 3
(`J₁ ≡ J₃ ≡ 0`) with `J₅ ≡ 0` and `J₇ ≡ 0` that has any nonzero odd-order
J-characteristic satisfies `N ≥ 2^8 = 256`. In particular no such design
exists with `N < 256` — e.g. with `N ∈ {56, 64}` — for any number of
factors. -/
theorem esvg_bound (f : Point n → ℤ) (hpos : ∀ x, 0 ≤ f x)
    (h1 : ∀ u : Finset (Fin n), u.card = 1 → jc f u = 0)
    (h3 : ∀ u : Finset (Fin n), u.card = 3 → jc f u = 0)
    (h5 : ∀ u : Finset (Fin n), u.card = 5 → jc f u = 0)
    (h7 : ∀ u : Finset (Fin n), u.card = 7 → jc f u = 0)
    (hex : ∃ s : Finset (Fin n), Odd s.card ∧ jc f s ≠ 0) :
    (2 : ℤ) ^ 8 ≤ ∑ x : Point n, f x := by
  -- Proof:
  apply min_odd_bound f hpos _ hex
  intro u huodd hult
  obtain ⟨k, hk⟩ := huodd
  have hcases : u.card = 1 ∨ u.card = 3 ∨ u.card = 5 ∨ u.card = 7 := by omega
  rcases hcases with h | h | h | h
  · exact h1 u h
  · exact h3 u h
  · exact h5 u h
  · exact h7 u h

end EoDivisibility
