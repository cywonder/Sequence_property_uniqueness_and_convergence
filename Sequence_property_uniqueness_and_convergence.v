(** MA_CH1 *)

(* 读入库文件 *)
Require Export Real_Axioms.

Parameter R_stru : Real_struct.
Parameter R_axio : Real_axioms R_stru. 

Global Hint Resolve R_axio : core.

Declare Scope MA_R_scope.
Delimit Scope MA_R_scope with ma.
Open Scope MA_R_scope.

Definition ℝ := @R R_stru.
Definition ℕ := @N R_stru.
Definition ℤ := @Z R_stru.
Definition ℚ := @Q R_stru.
Definition fp := @fp R_stru.
Definition zeroR := @zeroR R_stru.
Definition fm := @fm R_stru.
Definition oneR := @oneR R_stru.
Definition Leq := @Leq R_stru.
Definition Abs := @Abs R_stru.

Notation "x + y" := fp[[x, y]] : MA_R_scope.
Notation "0" := zeroR : MA_R_scope.
Notation "x · y" := fm[[x, y]](at level 40) : MA_R_scope.
Notation "1" := oneR : MA_R_scope.
Notation "x ≤ y" := ([x, y] ∈ Leq)(at level 77) : MA_R_scope.
Notation "- a" := (∩(\{ λ u, u ∈ ℝ /\ u + a = 0 \})) : MA_R_scope.
Notation "x - y" := (x + (-y)) : MA_R_scope.
Notation "a ⁻" := (∩(\{ λ u, u ∈ (ℝ ~ [0]) /\ u · a = 1 \}))
  (at level 5) : MA_R_scope.
Notation "m / n" := (m · (n⁻)) : MA_R_scope.
Notation "｜ x ｜" := (Abs[x])(at level 5, x at level 0) : MA_R_scope.

Definition LT x y := x ≤ y /\ x <> y.
Notation "x < y" := (LT x y) : MA_R_scope.
Definition Ensemble_R := @Ensemble_R R_stru R_axio.
Definition PlusR := @PlusR R_stru R_axio.
Definition zero_in_R := @zero_in_R R_stru R_axio.
Definition Plus_P1 := @Plus_P1 R_stru R_axio.
Definition Plus_P2 := @Plus_P2 R_stru R_axio.
Definition Plus_P3 := @Plus_P3 R_stru R_axio.
Definition Plus_P4 := @Plus_P4 R_stru R_axio.
Definition MultR := @MultR R_stru R_axio.
Definition one_in_R := @one_in_R R_stru R_axio.
Definition Mult_P1 := @Mult_P1 R_stru R_axio.
Definition Mult_P2 := @Mult_P2 R_stru R_axio.
Definition Mult_P3 := @Mult_P3 R_stru R_axio.
Definition Mult_P4 := @Mult_P4 R_stru R_axio.
Definition Mult_P5 := @Mult_P5 R_stru R_axio.
Definition LeqR := @LeqR R_stru R_axio.
Definition Leq_P1 := @Leq_P1 R_stru R_axio.
Definition Leq_P2 := @Leq_P2 R_stru R_axio.
Definition Leq_P3 := @Leq_P3 R_stru R_axio.
Definition Leq_P4 := @Leq_P4 R_stru R_axio.
Definition Plus_Leq := @Plus_Leq R_stru R_axio.
Definition Mult_Leq := @Mult_Leq R_stru R_axio.
Definition Completeness := @Completeness R_stru R_axio.

Definition Plus_close := @Plus_close R_stru R_axio.
Definition Mult_close := @Mult_close R_stru R_axio.
Definition one_in_R_Co := @one_in_R_Co R_stru R_axio.
Definition Plus_Co1 := @Plus_Co1 R_stru R_axio.
Definition Plus_Co2 := @Plus_Co2 R_stru R_axio.
Definition Plus_neg1a := @Plus_neg1a R_stru R_axio.
Definition Plus_neg1b := @Plus_neg1b R_stru R_axio.
Definition Plus_neg2 := @Plus_neg2 R_stru R_axio.
Definition Minus_P1 := @Minus_P1 R_stru R_axio.
Definition Minus_P2 := @Minus_P2 R_stru R_axio.
Definition Plus_Co3 := @Plus_Co3 R_stru R_axio.
Definition Mult_Co1 := @Mult_Co1 R_stru R_axio.
Definition Mult_Co2 := @Mult_Co2 R_stru R_axio.
Definition Mult_inv1 := @Mult_inv1 R_stru R_axio.
Definition Mult_inv2 := @Mult_inv2 R_stru R_axio.
Definition Divide_P1 := @Divide_P1 R_stru R_axio.
Definition Divide_P2 := @Divide_P2 R_stru R_axio.
Definition Mult_Co3 := @Mult_Co3 R_stru R_axio.
Definition PlusMult_Co1 := @PlusMult_Co1 R_stru R_axio.
Definition PlusMult_Co2 := @PlusMult_Co2 R_stru R_axio.
Definition PlusMult_Co3 := @PlusMult_Co3 R_stru R_axio.
Definition PlusMult_Co4 := @PlusMult_Co4 R_stru R_axio.
Definition PlusMult_Co5 := @PlusMult_Co5 R_stru R_axio.
Definition PlusMult_Co6 := @PlusMult_Co6 R_stru R_axio.
Definition Order_Co1 := @Order_Co1 R_stru R_axio.
Definition Order_Co2 := @Order_Co2 R_stru R_axio.
Definition OrderPM_Co1 := @OrderPM_Co1 R_stru R_axio.
Definition OrderPM_Co2a := @OrderPM_Co2a R_stru R_axio.
Definition OrderPM_Co2b := @OrderPM_Co2b R_stru R_axio.
Definition OrderPM_Co3 := @OrderPM_Co3 R_stru R_axio.
Definition OrderPM_Co4 := @OrderPM_Co4 R_stru R_axio.
Definition OrderPM_Co5 := @OrderPM_Co5 R_stru R_axio.
Definition OrderPM_Co6 := @OrderPM_Co6 R_stru R_axio.
Definition OrderPM_Co7a := @OrderPM_Co7a R_stru R_axio.
Definition OrderPM_Co7b := @OrderPM_Co7b R_stru R_axio.
Definition OrderPM_Co8a := @OrderPM_Co8a R_stru R_axio.
Definition OrderPM_Co8b := @OrderPM_Co8b R_stru R_axio.
Definition OrderPM_Co9 := @OrderPM_Co9 R_stru R_axio.
Definition OrderPM_Co10 := @OrderPM_Co10 R_stru R_axio.
Definition OrderPM_Co11 := @OrderPM_Co11 R_stru R_axio.
Definition IndSet := @IndSet R_stru.
Definition IndSet_P1 := @IndSet_P1 R_stru.
Definition N_Subset_R := @N_Subset_R R_stru R_axio.
Definition one_in_N := @one_in_N R_stru R_axio.
Definition zero_not_in_N := @zero_not_in_N R_stru R_axio.
Definition IndSet_N := @IndSet_N R_stru R_axio.
Definition MathInd := @MathInd R_stru R_axio.
Definition Nat_P1a := @Nat_P1a R_stru R_axio.
Definition Nat_P2 := @Nat_P2 R_stru R_axio.
Definition Nat_P3 := @Nat_P3 R_stru R_axio.
Definition Nat_P4 := @Nat_P4 R_stru R_axio.
Definition Nat_P5 := @Nat_P5 R_stru R_axio.
Definition Nat_P6 := @Nat_P6 R_stru R_axio.
Definition one_is_min_in_N := @one_is_min_in_N R_stru R_axio.
Definition N_Subset_Z := @N_Subset_Z R_stru.
Definition Z_Subset_R := @Z_Subset_R R_stru R_axio.
Definition Int_P1_Lemma := @Int_P1_Lemma R_stru R_axio.
Definition Int_P1a := @Int_P1a R_stru R_axio.
Definition Int_P1b := @Int_P1b R_stru R_axio.
Definition Int_P2 := @Int_P2 R_stru R_axio.
Definition Int_P3 := @Int_P3 R_stru R_axio.
Definition Int_P4 := @Int_P4 R_stru R_axio.
Definition Int_P5 := @Int_P5 R_stru R_axio.
Definition Z_Subset_Q := @Z_Subset_Q R_stru R_axio.
Definition Q_Subset_R := @Q_Subset_R R_stru R_axio.
Definition Frac_P1 := @Frac_P1 R_stru R_axio.
Definition Frac_P2 := @Frac_P2 R_stru R_axio.
Definition Rat_P1a := @Rat_P1a R_stru R_axio.
Definition Rat_P1b := @Rat_P1b R_stru R_axio.
Definition Rat_P2 := @Rat_P2 R_stru R_axio.
Definition Rat_P3 := @Rat_P3 R_stru R_axio.
Definition Rat_P4 := @Rat_P4 R_stru R_axio.
Definition Rat_P5 := @Rat_P5 R_stru R_axio.
Definition Rat_P6 := @Rat_P6 R_stru R_axio.
Definition Rat_P7 := @Rat_P7 R_stru R_axio.
Definition Rat_P8 := @Rat_P8 R_stru R_axio.
Definition Rat_P9 := @Rat_P9 R_stru R_axio.
Definition Rat_P10 := @Rat_P10 R_stru R_axio.
Definition Even := @Even R_stru.
Definition Odd := @Odd R_stru.
Definition Even_and_Odd_P1 := @Even_and_Odd_P1 R_stru R_axio.
Definition Even_and_Odd_P2_Lemma := @Even_and_Odd_P2_Lemma R_stru R_axio.
Definition Even_and_Odd_P2 := @Even_and_Odd_P2 R_stru R_axio.
Definition Even_and_Odd_P3 := @Even_and_Odd_P3 R_stru R_axio.
Definition Existence_of_irRational_Number :=
  @Existence_of_irRational_Number R_stru R_axio.
Definition Arch_P1 := @Arch_P1 R_stru R_axio.
Definition Arch_P2 := @Arch_P2 R_stru R_axio.
Definition Arch_P3_Lemma := @Arch_P3_Lemma R_stru R_axio.
Definition Arch_P3a := @Arch_P3a R_stru R_axio.
Definition Arch_P3b := @Arch_P3b R_stru R_axio.
Definition Arch_P4 := @Arch_P4 R_stru R_axio.
Definition Arch_P5 := @Arch_P5 R_stru R_axio.
Definition Arch_P6 := @Arch_P6 R_stru R_axio.
Definition Arch_P7 := @Arch_P7 R_stru R_axio.
Definition Arch_P8 := @Arch_P8 R_stru R_axio.
Definition Arch_P9 := @Arch_P9 R_stru R_axio.
Definition Arch_P10 := @Arch_P10 R_stru R_axio.
Definition Abs_is_Function := @Abs_is_Function R_stru R_axio.
Definition Abs_in_R := @Abs_in_R R_stru R_axio.
Definition Distance := @Distance R_stru.
Definition me_zero_Abs := @me_zero_Abs R_stru R_axio.
Definition le_zero_Abs := @le_zero_Abs R_stru R_axio.
Definition Abs_P1 := @Abs_P1 R_stru R_axio.
Definition Abs_P2 := @Abs_P2 R_stru R_axio.
Definition Abs_P3 := @Abs_P3 R_stru R_axio.
Definition Abs_P4 := @Abs_P4 R_stru R_axio.
Definition Abs_P5 := @Abs_P5 R_stru R_axio.
Definition Abs_P6 := @Abs_P6 R_stru R_axio.
Definition Abs_P7 := @Abs_P7 R_stru R_axio.
Definition Abs_P8 := @Abs_P8 R_stru R_axio.

Global Hint Resolve Plus_close zero_in_R Mult_close one_in_R one_in_R_Co
  Plus_neg1a Plus_neg1b Plus_neg2 Minus_P1 Minus_P2
  Mult_inv1 Mult_inv2 Divide_P1 Divide_P2 OrderPM_Co9
  N_Subset_R one_in_N Nat_P1a Nat_P1b
  N_Subset_Z Z_Subset_R Int_P1a Int_P1b
  Z_Subset_Q Q_Subset_R Rat_P1a Rat_P1b Abs_in_R: real.
  
(* 1.2 数集 确界原理 *)

(* 1.2.1 区间与邻域 *)

(* 有限区间 *) 
(* 开区间 *)
Notation "］ a , b ［" := (\{ λ x, x ∈ ℝ /\ a < x /\ x < b \})
  (at level 5, a at level 0, b at level 0) : MA_R_scope.

(* 闭区间 *)
Notation "［ a , b ］" := (\{ λ x, x ∈ ℝ /\ a ≤ x /\ x ≤ b \})
  (at level 5, a at level 0, b at level 0) : MA_R_scope.

(* 左开右闭 *)
Notation "］ a , b ］" := (\{ λ x, x ∈ ℝ /\ a < x /\ x ≤ b \})
  (at level 5, a at level 0, b at level 0) : MA_R_scope.

(* 左闭右开 *)
Notation "［ a , b ［" := (\{ λ x, x ∈ ℝ /\ a ≤ x /\ x < b \})
  (at level 5, a at level 0, b at level 0) : MA_R_scope.

(* 无限区间 *)
Notation "］ a , +∞［" := (\{ λ x, x ∈ ℝ /\ a < x \})
  (at level 5, a at level 0) : MA_R_scope.

Notation "［ a , +∞［" := (\{ λ x, x ∈ ℝ /\ a ≤ x \})
  (at level 5, a at level 0) : MA_R_scope.

Notation "］-∞ , b ］" := (\{ λ x, x ∈ ℝ /\ x ≤ b \})
  (at level 5, b at level 0) : MA_R_scope.

Notation "］-∞ , b ［" := (\{ λ x, x ∈ ℝ /\ x < b \})
  (at level 5, b at level 0) : MA_R_scope.

Notation "]-∞ , +∞[" := ℝ (at level 0) : MA_R_scope.

(* 邻域 *)
Definition Neighbourhood x δ := x ∈ ℝ /\ δ ∈ ℝ /\ x ∈ ］(x - δ),(x + δ)［.

(* 邻域 *)
Definition Neighbor a δ := \{ λ x, x ∈ ℝ /\ ｜ (x - a) ｜ < δ \}.
Notation "U( a ; δ )" := (Neighbor a δ)
  (a at level 0, δ at level 0, at level 4) : MA_R_scope.

(* 左邻域 *)
Definition leftNeighbor a δ := ］a-δ, a］.

(* 右邻域 *)
Definition rightNeighbor a δ := ［a, (a+δ)［.

(* 去心邻域 *)
Definition Neighbor0 a δ := \{ λ x, x ∈ ℝ 
  /\ 0 < ｜(x-a)｜ /\ ｜(x-a)｜ < δ \}.
Notation "Uº( a ; δ )" := (Neighbor0 a δ)
  (a at level 0, δ at level 0, at level 4) : MA_R_scope.

(* 左去心邻域 *)
Definition leftNeighbor0 a δ := ］a-δ, a［.
Notation "U-º( a ; δ )" := (leftNeighbor0 a δ)
  (a at level 0, δ at level 0, at level 4) : MA_R_scope.

(* 右去心邻域 *)
Definition rightNeighbor0 a δ := ］a, (a+δ)［.
Notation "U+º( a ; δ )" := (rightNeighbor0 a δ)
  (a at level 0, δ at level 0, at level 4) : MA_R_scope.

(* 无穷邻域 *)
Definition Neighbor_infinity M := \{ λ x, x ∈ ℝ /\ M ∈ ℝ
  /\ 0 < M /\ M < ｜x｜ \}.
Notation "U(∞) M" := (Neighbor_infinity M) (at level 5) : MA_R_scope.

(* 正无穷邻域 *)
Definition PNeighbor_infinity M := \{ λ x, x ∈ ℝ /\ M ∈ ℝ /\ M < x \}.
Notation "U(+∞) M" := (［ M , +∞［) (at level 5) : MA_R_scope.

(* 负无穷邻域 *)
Definition NNeighbor_infinity M := \{ λ x, x ∈ ℝ /\ M ∈ ℝ /\ x < M \}.
Notation "U(-∞) M" := (］-∞ , M ［) (at level 5) : MA_R_scope.

(* 1.2.2 有界集 确界原理 *)

(* 上界 *)
Definition UpperBound S M := S ⊂ ℝ /\ M ∈ ℝ /\ (∀ x, x ∈ S -> x ≤ M).

(* 下界 *)
Definition LowerBound S L := S ⊂ ℝ /\ L ∈ ℝ /\ (∀ x, x ∈ S -> L ≤ x).

(* 有界集 *)
Definition Bounded S := ∃ M L, UpperBound S M /\ LowerBound S L.

(* 无界集 *)
Definition Unbounded S := ~ (Bounded S).

(* 上确界 *)
Definition Sup S η := UpperBound S η /\ (∀ α, α ∈ ℝ -> α < η
  -> (∃ x0, x0 ∈ S /\ α < x0)).

(* 下确界 *)
Definition Inf S ξ := LowerBound S ξ /\ (∀ β, β ∈ ℝ -> ξ < β
  -> (∃ x0, x0 ∈ S /\ x0 < β)).
  
Definition Max S c := S ⊂ ℝ /\ c ∈ S /\ (∀ x, x ∈ S -> x ≤ c).

Definition Min S c := S ⊂ ℝ /\ c ∈ S /\ (∀ x, x ∈ S -> c ≤ x).

Corollary Max_Corollary : ∀ S c1 c2, Max S c1 -> Max S c2 -> c1 = c2.
Proof.
  intros. unfold Max in *. destruct H as [H []], H0 as [H0 []].
  pose proof H1. pose proof H3. apply H2 in H6. apply H4 in H5.
  apply Leq_P2; auto.
Qed.
  
Corollary Min_Corollary : ∀ S c1 c2, Min S c1 -> Min S c2 -> c1 = c2.
Proof.
  intros. unfold Min in *. destruct H as [H []], H0 as [H0 []].
  pose proof H1. pose proof H3. apply H2 in H6. apply H4 in H5.
  apply Leq_P2; auto.
Qed.

Definition Sup_Equal S η := Min \{ λ u, UpperBound S u \} η.

Corollary Sup_Corollary : ∀ S η, Sup S η <-> Sup_Equal S η.
Proof.
  intros. split; intro.
  - red in H; red. destruct H, H as [H []]. repeat split. 
    unfold Included; intros. apply AxiomII in H3 as [_[_[]]]; auto.
    apply AxiomII; split. unfold Ensemble; exists ℝ; auto. split; auto.
    intros. apply AxiomII in H3 as [H3 [H4 []]]. pose proof H5.
    apply (Order_Co1 x η) in H7 as [H7 | [|]]; auto.
    + apply H0 in H7 as [x0 []]; auto. pose proof H7. apply H6 in H9.
      destruct H8. elim H10. apply Leq_P2; auto.
    + destruct H7. auto.
    + rewrite H7. apply Leq_P1; auto.
  - red in H; red. destruct H as [H []]. apply AxiomII in H0 as [_[H0 []]].
    repeat split; auto. intros. apply NNPP; intro.
    assert (∀ x1, x1 ∈ S -> x1 ≤ α).
    { intros. apply NNPP; intro. elim H6. exists x1. split; auto.  
      pose proof H4. apply (@ Order_Co1 x1 α) in H9
      as [H9 | [|]]; auto. elim H8. destruct H9. auto. 
      rewrite H9 in H8. elim H8. apply Leq_P1; auto. }
    assert (α ∈ \{ λ u, UpperBound S u \}).
    { apply AxiomII. split. exists ℝ; auto. split; auto. }
    apply H1 in H8. destruct H5. elim H9. 
    apply Leq_P2; auto.
Qed.

Definition Inf_Equal S ξ := Max \{ λ u, LowerBound S u \} ξ.

Corollary Inf_Corollary : ∀ S ξ, Inf S ξ <-> Inf_Equal S ξ.
Proof.
  intros. split; intro.
  - red in H; red. destruct H, H as [H []]. repeat split.
    unfold Included; intros. apply AxiomII in H3 as [_[_[]]]; auto.
    apply AxiomII; split. exists ℝ; auto. repeat split; auto.
    intros. apply AxiomII in H3 as [H3 [H4 []]]. pose proof H5.
    apply (@ Order_Co1 x ξ) in H7 as [H7 | [|]]; auto.
    + destruct H7. auto.
    + apply H0 in H7 as [x0 []]; auto. pose proof H7. apply H6 in H9.
      destruct H8. elim H10. apply Leq_P2; auto.
    + rewrite H7. apply Leq_P1; auto.
  - red in H; red. destruct H as [H []]. apply AxiomII in H0 as [_[H2 []]].
    repeat split; auto. intros. apply NNPP; intro.
    assert (∀x1, x1 ∈ S -> β ≤ x1).
    { intros. apply NNPP; intro. elim H6. exists x1. split; auto.
      pose proof H4. apply (Order_Co1 x1 β) in H9 
      as [H9 | [|]]; auto. elim H8. destruct H9. auto.
      rewrite H9 in H8. elim H8. apply Leq_P1; auto. }
    assert (β ∈ \{ λ u, LowerBound S u \}).
    { apply AxiomII; split. exists ℝ; auto. repeat split; auto. }
    apply H1 in H8. destruct H5. elim H9.
    apply Leq_P2; auto.
Qed.

(* 确界原理 *)

(* Theorem SupL' : ∀ S, S ⊂ ℝ -> S <> Φ -> (∃ M, UpperBound S M)
  -> exists η, Sup S η.
Proof.
  intros. destruct H1 as [M].
  set (Y := \{ λ u, UpperBound S u \}).
  assert (Y <> Φ).
  { apply NEexE. exists M. apply AxiomII; split; auto. 
    red in H1. destruct H1 as [_[]]. eauto. }
  assert (Y ⊂ ℝ).
  { unfold Included. intros. apply AxiomII in H3. 
    destruct H3 as [_[_[]]]. auto. }
  assert (∃ c, c ∈ ℝ /\ (∀ x y, x ∈ S -> y ∈ Y -> (x ≤ c /\ c ≤ y))).
  { apply (@ Completeness R_stru); auto. intros. apply AxiomII in H5
    as [_[_[]]]. auto. }
  destruct H4 as [c[]]. unfold Sup. exists c. split.
  - unfold UpperBound. repeat split; auto; intros.
    apply NEexE in H2 as [y]. apply (H5 x y); auto.
  - intros. apply NNPP. intro.
    assert (∀ x0, x0 ∈ S -> x0 ≤ α).
    { intros. apply NNPP; intro. apply H8. exists x0.
      split; auto. apply (@ Order_Co1 R_stru R_axio α x0) in H6 
      as [H6|[]]; auto. destruct H6. contradiction. rewrite H6 in H10.
      elim H10. apply (@ Leq_P1 R_stru); auto. }
    assert (α ∈ Y).
    { apply AxiomII; split; eauto. repeat split; intros; auto. }
    apply NEexE in H0 as []. apply (H5 x α) in H10 as []; auto.
    destruct H7. apply H12. apply (@ Leq_P2 R_stru); auto.
Qed. *)

(* 上确界引理 *)
Lemma SupLemma : ∀ X, X ⊂ ℝ -> X <> Φ -> (∃ c, UpperBound X c) 
  -> exists ! η, Sup X η.
Proof.
  intros. set (Y:=\{ λ u, UpperBound X u \}).
  assert (Y <> Φ).
  { apply NEexE. destruct H1 as [x]. exists x. apply AxiomII;
    split; auto. destruct H1 as [_[]]. unfold Ensemble. exists ℝ; auto. }
  assert (Y ⊂ ℝ).
  { unfold Included; intros. apply AxiomII in H3 as [_].
    destruct H3 as [_[]]. auto. }
  assert (∃ c, c ∈ ℝ /\ (∀ x y, x ∈ X -> y ∈ Y 
    -> (x ≤ c /\ c ≤ y))) as [c[]].
  { apply Completeness; auto. intros. apply AxiomII in H5 as [_].
    destruct H5 as [_[]]. apply H6 in H4; auto. }
  assert (c ∈ Y).
  { apply AxiomII; repeat split; eauto. intros.
    apply NEexE in H2 as [y]. pose proof H5 _ _ H6 H2; tauto. }
  assert (Min \{ λ u, UpperBound X u \} c).
  { red. repeat split; auto. intros y H7. apply NEexE in H0 as [x].
    pose proof H5 _ _ H0 H7. tauto. }
  exists c. split. apply Sup_Corollary. auto. intros.
  apply Sup_Corollary in H8. apply (Min_Corollary Y); auto.
Qed.

(* 下确界引理 *)
Lemma InfLemma : ∀ X, X ⊂ ℝ -> X <> Φ -> (∃ c, LowerBound X c)
  -> exists ! ξ, Inf X ξ.
Proof.
  intros. set(Y:=\{ λ u, LowerBound X u \}).
  assert (Y <> Φ).
  { apply NEexE. destruct H1 as [x]. exists x. apply AxiomII;
    split; auto. destruct H1 as [_[]]. exists ℝ; auto. }
  assert (Y ⊂ ℝ).
  { unfold Included. intros. apply AxiomII in H3 as [].
    destruct H4 as [_[]]; auto. }
  assert (∃ c, c ∈ ℝ /\ (∀ y x, y ∈ Y -> x ∈ X 
    -> y ≤ c /\ c ≤ x)) as [c[]].
  { apply Completeness; auto. intros.
    apply AxiomII in H4 as [_[_[]]]. apply H6 in H5; auto. }
  assert (c ∈ Y).
  { apply AxiomII. repeat split; eauto. intros. 
    apply NEexE in H2 as [y]. pose proof H5 _ _ H2 H6; tauto. }
  assert (Max \{ λ u, LowerBound X u \} c).
  { red. repeat split; auto. intros y H7. apply NEexE in H0 as [x].
    pose proof H5 _ _ H7 H0; tauto. }
  exists c. split. apply Inf_Corollary; auto. intros.
  apply Inf_Corollary in H8. apply (Max_Corollary Y); auto.
Qed.

(* 确界原理 *)
Theorem Sup_Inf_Principle : ∀ X, X ⊂ ℝ -> X <> Φ
  -> ((∃ c, UpperBound X c) -> exists ! η, Sup X η)
  /\ ((∃ c, LowerBound X c) -> exists ! ξ, Inf X ξ).
Proof.
  intros. split; intros.
  - apply SupLemma; auto.
  - apply InfLemma; auto.
Qed.

(* 1.3 函数概念 *)

(* 1.3.1 函数的定义 *)
(* Note ：MK中已经给出 *)

(* 有序数对 *)
(* Definition Ordered x y := [ [x] | [x|y] ].
Notation "[ x , y ]" := (Ordered x y) (at level 0) : MA_R_scope. *)

(* 以有序数对的第一个元为第一坐标 *)
(* Definition First z := ∩∩z. *)

(* 以有序数对的第二个元为第二坐标 *)
(* Definition Second z := (∩∪z)∪(∪∪z) ~ (∪∩z). *)

(* 有序数对相等，对应坐标相等 *)
(* Theorem ProdEqual : ∀ x y u v, Ensemble x -> Ensemble y
  -> ([x, y] = [u, v] <-> x = u /\ y = v).
Proof.
  split; intros; [|destruct H1; subst; auto].
  assert (Ensemble ([x,y])); auto. rewrite H1 in H2.
  apply MKT49b in H2 as [].
  rewrite <-(MKT54a x y), H1,
  <-(MKT54b x y),H1,MKT54a,MKT54b; auto.
Qed. *)

(* 关系 *)
(* Definition Relation r := ∀ z, z ∈ r -> (∃ x y, z = [x, y]). *)

(* 关系的复合及关系的逆 *)
(* Definition Composition r s := \{\ λ x z, ∃ y, [x,y] ∈ s /\ [y,z] ∈ r \}\.
Notation "r ∘ s" := (Composition r s) (at level 50).

Definition Inverse r := \{\ λ x y, [y, x] ∈ r \}\.
Notation "r ⁻¹" := (Inverse r) (at level 5). *)

(* 满足性质P的有序数对构成的集合: { (x,y) : ... } *)
(* Notation "\{\ P \}\" :=
  (\{ λ z, ∃ x y, z = [x,y] /\ P x y \}) (at level 0). *)

(* 分类公理图示II关于有序数对的适应性事实 *)
(* Fact AxiomII' : ∀ a b P,
  [a,b] ∈ \{\ P \}\ <-> Ensemble ([a,b]) /\ (P a b).
Proof.
  split; intros.
  - PP H x y. apply MKT55 in H0 as []; subst; auto; ope.
  - destruct H. appA2G.
Qed. *)

(* 函数 *)
(* Definition Function f := 
  Relation f /\ (∀ x y z, [x,y] ∈ f -> [x,z] ∈ f -> y = z). *)

(* 定义域 *)
(* Definition Domain f := \{ λ x, ∃ y, [x,y] ∈ f \}.
Notation "dom( f )" := (Domain f) (at level 5). *)

(* 值域 *)
(* Definition Range f := \{ λ y, ∃ x, [x,y] ∈ f \}.
Notation "ran( f )" := (Range f) (at level 5). *)

(* f在点x的函数值 *)
(* Definition Value f x := ∩\{ λ y, [x,y] ∈ f \}.
Notation "f [ x ]" := (Value f x) (at level 5). *)

(* 声明函数的定义域和值域都是实数ℝ类型 *)
(* Definition Function f : Prop := dom(f) = ℝ /\ ran(f) ⊂ ℝ. *)

(* 1.3.2 函数的四则运算 *)

Definition Plus_Fun f g :=
  \{\ λ x y, x ∈ dom(f) /\ x ∈ dom(g) /\ y = f[x] + g[x] \}\.
Definition Sub_Fun f g :=
  \{\ λ x y, x ∈ dom(f) /\ x ∈ dom(g) /\ y = f[x] - g[x] \}\.
Definition Mult_Fun f g :=
  \{\ λ x y, x ∈ dom(f) /\ x ∈ dom(g) /\ y = f[x] · g[x] \}\.
Definition Div_Fun f g :=
  \{\ λ x y, x ∈ dom(f) /\ x ∈ dom(g) /\ g[x] <> 0 /\ y = f[x] / g[x] \}\.
  
Notation "f \+ g" := (Plus_Fun f g) (at level 45, left associativity).
Notation "f \- g" := (Sub_Fun f g) (at level 45, left associativity).
Notation "f \· g" := (Mult_Fun f g) (at level 40, left associativity).
Notation "f // g" := (Div_Fun f g) (at level 40, left associativity).

(* 1.3.3 复合函数 *)

Definition Comp : ∀ f g, Function f -> Function g -> Function (f ∘ g).
Proof.
  split; intros; unfold Composition; auto.
  appoA2H H1. appoA2H H2. rdeHex. destruct H0.
  apply H with x0; auto. rewrite (H7 x x0 x1); auto.
Qed.

(* 复合函数完整条件定义 *)
(* Definition Comp f g := ∀ f g x u, Function f -> Function g
  -> x ∈ dom(g) /\ g[x] ∈ dom(f)
  /\ u = g[x] /\ (f ∘ g)[x] = f[u] /\ dom(f ∘ g) = 
  (\{ λ x, g[x] ∈ dom(f) \} ∩ dom(g)) /\ dom(f ∘ g) <> Φ. *)

(* Definition Comp (f g : Fun) := ∀(f g : Fun) (x u : R), x ∈ dom[g] /\
g[x] ∈ dom[f] /\ u = g[x] /\ (f ∘ g) [x] = f[u]/\ dom[(f ∘ g)] =
(\{ λ x : R, (g [x] ∈ dom[f]) \} ∩ dom[g]) /\ NotEmpty dom[(f ∘ g)]. *)

(* 1.3.4 反函数 *)

Definition Inverse_Fun f g := Function1_1 f /\ g = f⁻¹.

Corollary Inverse_Co1 : ∀ f u, Function f -> Function f⁻¹ -> u ∈ dom(f)
  -> (f⁻¹)[f[u]] = u.
Proof.
  intros. apply Property_Value,invp1,Property_Fun in H1; auto.
Qed.

Corollary Inverse_Co2: ∀ f u, Function f -> Function f⁻¹ -> u ∈ ran(f)
  -> f[(f⁻¹)[u]] = u.
Proof.
  intros. rewrite reqdi in H1. apply Property_Value in H1; auto.
  apply ->invp1 in H1; auto. apply Property_Fun in H1; auto.
Qed.

(* 1.4 具有某些特性的函数 *)

(* 1.4.1 有界函数 *)

Definition UpBoundedFun f D : Prop :=
  Function f /\ D = dom(f) /\ (∃ M, M ∈ ℝ /\ ∀ x, x ∈ D -> f[x] ≤ M).

Definition LowBoundedFun f D : Prop :=
  Function f /\ D = dom(f) /\ (∃ L, L ∈ ℝ /\ ∀ x, x ∈ D -> L ≤ f[x]).

Definition BoundedFun f D : Prop := Function f /\ D = dom(f) 
  /\ (∃ M, M ∈ ℝ -> 0 < M /\ ∀ x, x ∈ D -> ｜[f[x]]｜ ≤ M).

(* 1.4.2 单调函数 *)

Definition IncreaseFun f := Function f
  /\ (∀ x1 x2, x1 ∈ dom(f) -> x2 ∈ dom(f) -> x1 < x2 -> f[x1] ≤ f[x2]).
Definition StrictIncreaseFun f := Function f
  /\ (∀ x1 x2, x1 ∈ dom(f) -> x2 ∈ dom(f) -> x1 < x2 -> f[x1] < f[x2]).
Definition DecreaseFun f := Function f
  /\ (∀ x1 x2, x1 ∈ dom(f) -> x2 ∈ dom(f) -> x1 < x2 -> f[x2] ≤ f[x2]).
Definition StrictDecreaseFun f := Function f
  /\ (∀ x1 x2, x1 ∈ dom(f) -> x2 ∈ dom(f) -> x1 < x2 -> f[x2] < f[x1]).

Theorem Theorem1_2_1 : ∀ f, Function f -> dom(f) ⊂ ℝ -> ran(f) ⊂ ℝ
  -> StrictIncreaseFun f -> StrictIncreaseFun f⁻¹.
Proof.
  intros; unfold StrictIncreaseFun in *. destruct H2 as [H2 H3].
  assert (Function f⁻¹).
  { unfold Function in *. unfold Relation in *. destruct H as []; split.
    - intros. apply AxiomII in H5 as [H5 [x[y]]]. exists x, y; tauto.
    - intros. apply AxiomII' in H5 as []; apply AxiomII' in H6 as [].
      assert (y ∈ ℝ).
      { unfold Included in H0. apply H0. apply AxiomII; split.
        apply MKT49c2 in H5; auto. exists x; auto. }
      assert (z ∈ ℝ).
      { apply H0. apply Property_dom in H8; auto. }
      New H7. New H8. apply Property_dom in H11, H12; auto.
      destruct (Order_Co1 y z) as [H13 | [|]]; auto.
      + apply H3 in H13; auto.
        assert (x = f[y] /\ x = f[z]) as [].
        { split; [apply (H4 y)|apply (H4 z)]; auto;
          apply Property_Value; auto. }
        rewrite <-H14, <-H15 in H13. destruct H13. elim H16. auto.
      + apply H3 in H13; auto.
        assert (x = f[y] /\ x = f[z]) as [].
        { split; [apply (H4 y)|apply (H4 z)]; auto;
          apply Property_Value; auto. }
        rewrite <-H14, <-H15 in H13. destruct H13. elim H16. auto. }
  split; auto; intros.
  - New H5; New H6. apply Property_Value,Property_ran in H8, H9; auto.
    rewrite <-deqri in H8, H9.
    destruct (Order_Co1 (f⁻¹)[x1] (f⁻¹)[x2])
    as [H10 | [|]]; auto.
    + apply H3 in H10; auto. rewrite f11vi in H10, H10; auto;
      try rewrite reqdi; auto. destruct H7, H10. elim H12.
      rewrite <-reqdi in H5, H6. apply Leq_P2; auto.
    + assert (f[(f ⁻¹)[x1]] = f[(f ⁻¹)[x2]]). { rewrite H10. auto. }
      rewrite <-reqdi in H5, H6. rewrite f11vi in H11, H11; auto.
      destruct H7. elim H12; auto.
Qed.

Theorem Theorem1_2_2 : ∀ f, Function f -> dom(f) ⊂ ℝ -> ran(f) ⊂ ℝ 
  -> StrictDecreaseFun f -> StrictDecreaseFun f⁻¹.
Proof.
  intros. unfold StrictDecreaseFun in *. destruct H2 as [].
  assert (Function f⁻¹).
  { unfold Function in *. unfold Relation in *. destruct H as []; split.
    + intros. apply AxiomII in H5 as [H5 [x[y]]]. exists x, y; tauto.
    + intros. apply AxiomII' in H5 as [], H6 as [].
      New H7; New H8. apply Property_dom in H9, H10.
      destruct (Order_Co1 y z) as [H11 | [|]]; auto.
      * apply H3 in H11; auto.
        assert (x = f[z] /\ x = f[y]) as [].
        { split; [apply (H4 z)|apply (H4 y)]; auto;
          apply Property_Value; auto. }
        rewrite <-H12, <-H13 in H11. destruct H11. elim H14. auto.
      * apply H3 in H11; auto.
        assert (x = f[z] /\ x = f[y]) as [].
        { split; [apply (H4 z)|apply (H4 y)]; auto;
          apply Property_Value; auto. }
        rewrite <-H12, <-H13 in H11. destruct H11. elim H14. auto. }
  split; auto; intros.
  - New H5; New H6. apply Property_Value,Property_ran in H8, H9; auto.
    rewrite <-deqri in H8, H9.
    destruct (Order_Co1 (f⁻¹)[x2] (f⁻¹)[x1]) 
    as [H10 | [|]]; auto.
    + apply H3 in H10; auto. rewrite f11vi in H10, H10; auto;
      rewrite <-reqdi in H5, H6; auto. destruct H7, H10. elim H11.
      apply Leq_P2; auto.
    + assert (f[(f ⁻¹)[x2]] = f[(f ⁻¹)[x1]]). { rewrite H10. auto. }
      rewrite <-reqdi in H5, H6. rewrite f11vi in H11, H11; auto.
      destruct H7. elim H12; auto.
Qed.

(* 1.4.3 奇函数和偶函数 *)

(* 奇函数 *)
Definition OddFun f := Function f /\ dom(f) ⊂ ℝ /\ ran(f) ⊂ ℝ
  /\ (∀ x, x ∈ dom(f) -> f[-x] = -f[x]).

(* 偶函数 *)
Definition EvenFun f := Function f /\ dom(f) ⊂ ℝ /\ ran(f) ⊂ ℝ
  /\ (∀ x, x ∈ dom(f) -> f[-x] = f[x]).

(* 1.4.4 周期函数 *)
Definition PeriodicFun f := Function f /\ (∃ σ, σ ∈ ℝ -> 0 < σ 
  /\ (∀ x, x ∈ ℝ -> x ∈ dom(f) -> (x + σ ∈ dom(f) -> f[x + σ] = f[x])
  /\ (x - σ ∈ dom(f) -> f[x - σ] = f[x]))).

(* ℂ ℕ ℝ ℚ ℤ *)
(* 2 数列极限 *)

(* 2.1 数列极限 *)

(* 定义：数列 *)
Definition IsSeq f := Function f /\ dom(f) = ℕ /\ ran(f) ⊂ ℝ.

(* 定义1：数列极限 *)
Definition Limit_Seq x a := IsSeq x /\ (∀ ε, ε ∈ ℝ /\ 0 < ε
  -> (∃ N, N ∈ ℕ /\ (∀ n, n ∈ ℕ -> N < n -> Abs[x[n] - a] < ε))).

(* 定义1' 数列极限的邻域刻画 *)
Definition Limit_Seq' x a := IsSeq x /\ (∀ ε, ε ∈ ℝ /\ 0 < ε
  -> Finite \{ λ u, u ∈ ℕ /\ x[u] ∉ (Neighbor a ε) \}).

Theorem MathInd_Ma : ∀ E, E ⊂ ℕ -> 1 ∈ E
  -> (∀ x, x ∈ E -> (x + 1) ∈ E) -> E = ℕ.
Proof.
  intros. assert (IndSet E).
  { split; auto. unfold Included; intros. apply N_Subset_R; auto. }
  apply MKT27; split; auto. unfold Included; intros.
  apply AxiomII in H3 as []. apply H4. apply AxiomII; split; auto.
  apply (MKT33 ℕ); auto. apply (MKT33 ℝ).
  apply Ensemble_R; auto. apply N_Subset_R; auto.
Qed.

Corollary Nat_nle_gt : ∀ n m, n ∈ ℕ -> m ∈ ℕ -> ~ (n ≤ m) <-> (m < n).
Proof.
  intros. apply N_Subset_R in H, H0. split; intros.
  - apply NNPP; intro. destruct (Leq_P4 m n); auto.
    destruct (classic (m = n)).
    + elim H1. rewrite H4. apply Leq_P1; auto.
    + elim H2. split; auto.
  - destruct H1. intro. elim H2. apply Leq_P2; auto.
Qed.

Corollary Nat_le_ngt : ∀ n m, n ∈ ℕ -> m ∈ ℕ -> n ≤ m <-> ~ (m < n).
Proof.
  intros. apply N_Subset_R in H, H0. split; intros.
  - intro. destruct H2. elim H3. apply Leq_P2; auto.
  - apply NNPP; intro. destruct (Leq_P4 m n); auto.
    destruct (classic (m = n)).
    + elim H2; rewrite H4; apply Leq_P1; auto.
    + elim H1; split; auto.
Qed.

Corollary Nat_P4a: ∀ m n, m ∈ ℕ -> n ∈ ℕ -> (n + 1) ≤ m -> n < m.
Proof.
  intros. apply N_Subset_R in H as Ha, H0 as Hb.
  assert (n < n + 1).
  { apply (Leq_P1 n) in Hb as Hc. destruct OrderPM_Co9.
    apply (OrderPM_Co3 _ _ 0 1) in Hc; auto with real.
    rewrite Plus_P1 in Hc; auto. split; auto. intro.
    assert (n - n = n + 1 - n). { rewrite <-H4; auto. }
    rewrite Minus_P1, (Plus_P4 n), <-Plus_P3, Plus_neg2, Plus_P1 in H5;
    auto with real. }
  destruct H2. split.
  - apply (Leq_P3 _ (n + 1) _); auto with real.
  - intro. rewrite <-H4 in H1. elim H3. apply Leq_P2; auto with real.
Qed.

Corollary Nat_P4b : ∀ m n, m ∈ ℕ -> n ∈ ℕ -> n ≤ m -> n < m + 1.
Proof.
  intros. apply N_Subset_R in H as Ha, H0 as Hb.
  assert (m < m + 1).
  { apply (Leq_P1 m) in Ha as Hc. destruct OrderPM_Co9.
    apply (OrderPM_Co3 _ _ 0 1) in Hc; auto with real.
    rewrite Plus_P1 in Hc; auto. split; auto. intro.
    assert (m - m = m + 1 - m). { rewrite <-H4; auto. }
    rewrite Minus_P1, (Plus_P4 m), <-Plus_P3, Plus_neg2, Plus_P1 in H5;
    auto with real. }
  destruct H2. split.
  - apply (Leq_P3 _ m _); auto with real.
  - intro. rewrite H4 in H1. elim H3. apply Leq_P2; auto with real.
Qed.

(* 小于等于某个数的自然数集为非空有限集 *)
Theorem NatFinite : ∀ n, n ∈ ℕ -> Finite \{ λ u, u ∈ ℕ /\ u ≤ n \}
  /\ \{ λ u, u ∈ ℕ /\ u ≤ n \} <> Φ.
Proof.
  intros. split.
  - set (F := \{ λ u, u ∈ ℕ /\ Finite \{ λ v, v ∈ ℕ /\ v ≤ u \} \}).
    assert (F = ℕ).
    { apply MathInd_Ma; auto; unfold Included; intros.
      apply AxiomII in H0 as [_[]]; auto.
      - apply AxiomII; repeat split; eauto with real.
        assert (\{ λ v, v ∈ ℕ /\ v ≤ 1 \} = [1]).
        { apply AxiomI; split; intros.
          - apply AxiomII in H0 as [H0 []]. apply MKT41; eauto with real.
            apply Leq_P2; auto with real.
            apply NNPP; intro. destruct (classic (z = 1)).
            + elim H3. rewrite H4. apply Leq_P1; auto with real.
            + destruct one_is_min_in_N as [H5 []]. pose proof H1.
              apply H7 in H8. elim H4. apply Leq_P2; auto with real.
          - apply MKT41 in H0; eauto with real. rewrite H0.
            apply AxiomII; repeat split; eauto with real. 
            apply Leq_P1; apply one_in_R_Co. }
        rewrite H0. apply finsin; eauto with real.
      - apply AxiomII in H0 as [H0 []]. pose proof IndSet_N as Ha. 
        apply AxiomII; repeat split; eauto with real.
        assert (\{ λ v, v ∈ ℕ /\ v ≤ x + 1 \} 
          = \{ λ v, v ∈ ℕ /\ v ≤ x \} ∪ [x + 1]).
          { apply AxiomI; split; intros.
            - apply AxiomII in H3 as [H3 []]. apply MKT4.
              destruct (classic (z = x + 1)).
              + right. apply MKT41; eauto with real.
              + left. apply AxiomII; repeat split; auto.
                assert (z < x + 1). split; auto.
                apply Nat_P4 in H7; auto with real.
                apply Plus_Leq with (z:=-(1)) in H7; auto with real.
                rewrite <-Plus_P3, <-Plus_P3 in H7; auto with real.
                rewrite Minus_P1, Minus_P1 in H7; auto with real.
                rewrite Plus_P1, Plus_P1 in H7; auto with real.
            - apply MKT4 in H3 as [].
              + apply AxiomII in H3 as [H3 []].
                apply AxiomII; repeat split; auto.
                assert (z + 0 ≤ x + 1).
                { apply OrderPM_Co3; try apply OrderPM_Co9; auto with real. }
                rewrite Plus_P1 in H6; auto with real.
              + apply MKT41 in H3; eauto with real. rewrite H3.
                apply AxiomII; repeat split; eauto with real.
                apply Leq_P1; auto with real. }
          rewrite H3. apply MKT168; auto. apply finsin; eauto with real. }
    rewrite <-H0 in H. apply AxiomII in H as [H []]; auto.
  - apply NEexE. exists 1. apply AxiomII; repeat split; eauto with real.
    pose proof one_is_min_in_N. destruct H0 as [H0 []]. apply H2; auto.
Qed.

(* 最大值 *)
Definition maxR A r := A ⊂ ℝ /\ r ∈ A /\ (∀ x, x ∈ A -> x ≤ r).

(* 非空有限的实数集有最大值 (非空有限的自然数集有最大值易证: 自然数集是实数集的子集) *)
Theorem finite_maxR : ∀ A, A <> Φ -> Finite A -> A ⊂ ℝ
  -> (∃ r, maxR A r).
Admitted.

(* 非空有限的自然数集有最大值 *)
Theorem finite_maxN : ∀ A, A <> Φ -> Finite A -> A ⊂ ℕ
  -> (∃ r, maxR A r).
Proof.
  intros. assert (A ⊂ ℝ).
  { unfold Included; intros; apply N_Subset_R; auto. }
  apply finite_maxR; auto.
Qed.

(* 两种极限定义等价 *)
Theorem Limit_Equal : ∀ x a, IsSeq x -> a ∈ ℝ 
  -> Limit_Seq x a <-> Limit_Seq' x a.
Proof.
  intros. unfold Limit_Seq; unfold Limit_Seq'. split; intros.
  - destruct H1 as [Ha H1]. split; auto. intros.
    apply H1 in H2 as H3. destruct H3 as [N []].
    (* set写法 *)
    assert (∀ n, n ∈ ℕ -> ∃ A, A = \{ λ u, u ∈ ℕ /\ u ≤ n \}).
    { intros. exists \{ λ u, u ∈ ℕ /\ u ≤ n \}. auto. }
    New H3. apply H5 in H3. destruct H3 as [A].
    assert (\{ λ u, u ∈ ℕ /\ x[u] ∉ (Neighbor a ε) \} ⊂ A).
    { intros z H7. apply AxiomII in H7 as [H8 []]. rewrite H3.
      apply AxiomII; repeat split; auto. apply NNPP; intro.
      apply Nat_nle_gt in H10; auto. apply H4 in H10; auto.
      elim H9. destruct Ha as [Ha []].
      assert (x[z] ∈ ran(x)).
      { apply Property_dm; auto. rewrite H11; auto. }
      destruct H10. apply AxiomII; repeat split; eauto. }
    apply @ finsub with (A:= A); auto. rewrite H3. apply NatFinite; auto.
  - destruct H1 as [Ha H1]. split; auto.
    intros ε H2. pose proof H2 as Hb. 
    apply H1 in H2. apply finite_maxN in H2.
    destruct H2 as [N [H2 []]]. apply AxiomII in H3 as [H3 []].
    exists N; split; auto. intros. apply NNPP; intro.
    assert (n ∈ \{ λ u, u ∈ ℕ /\ x[u] ∉ U(a;ε) \}).
    { apply AxiomII; repeat split; eauto. 
      intro. apply AxiomII in H10 as [H10 []]. contradiction. }
    apply H4 in H10. apply Nat_le_ngt in H10; auto.
    * apply NEexE. admit.
Admitted.

(* 收敛数列 *)
Definition Convergence x := ∃ a, a ∈ ℝ /\ Limit_Seq x a.

(* 发散数列 *)
Definition Divergence x := IsSeq x /\ (∀ a, a ∈ ℝ /\ ~ Limit_Seq x a).

(* 定义2：无穷小数列 *)
Definition Infinitesimal x := Limit_Seq x 0.

(* 定理2.1 *)
Theorem Theorem2_1 : ∀ x a, a ∈ ℝ -> IsSeq x -> Limit_Seq x a  
  <-> Infinitesimal \{\ λ u v, u ∈ ℕ /\ v ∈ ℝ /\ v = x[u] - a \}\.
Proof.
  split; intros.
  - unfold Infinitesimal. unfold Limit_Seq in *. destruct H1 as [H1 H2].
    assert (IsSeq \{\ λ u v, u ∈ ℕ /\ v ∈ ℝ /\ v = x[u] - a \}\).
    { unfold IsSeq in *. destruct H1 as [H1 []]. repeat split.
      - unfold Relation. intros. apply AxiomII in H5 as [].
        destruct H6 as [x0 [y []]]. exists x0,y. auto.
      - intros. apply AxiomII' in H5 as [], H6 as [].
        destruct H7 as [H7 []], H8 as [H8 []]. subst; auto.
      - apply AxiomI; split; intros.
        + apply AxiomII in H5 as [H5 [y]].
          apply AxiomII' in H6 as [H6 [H7 []]]; auto.
        + assert ((x[z] - a) ∈ ℝ).
          { apply Plus_close; auto. apply H4.
            apply (@ Property_ran z), Property_Value; auto. rewrite H3; auto.
            apply Plus_neg1a; auto. }
          apply AxiomII; split; eauto. exists (x[z] - a).
          apply AxiomII'; repeat split; auto; apply MKT49a; eauto.
      - unfold Included; intros. apply AxiomII in H5 as [H5 [x0]].
        apply AxiomII' in H6 as [H6 [H7 []]]; auto. }
    split; auto.
    intros. pose proof H4. destruct H5. apply H2 in H4. destruct H4 as [N []]. 
    exists N. split; auto. intros. apply H7 in H9; auto.
    destruct H1 as [H1 []]; destruct H3 as [H3 []].
    assert ((x[n] - a) ∈ ℝ).
    { apply Plus_close; auto. apply H11.
      apply (@ Property_ran n),Property_Value; auto. rewrite H10; auto.
      apply Plus_neg1a; auto. }
    assert (x[n] - a = \{\ λ u v, u ∈ ℕ /\ v ∈ ℝ /\ v = x[u] - a \}\ [n]).
    { apply Property_Fun; auto. apply AxiomII'; repeat split; auto.
      apply MKT49a; eauto. }
    assert (\{\ λ u v, u ∈ ℕ /\ v ∈ ℝ /\ v = x[u] - a \}\ [n] ∈ ℝ).
    { apply H13. apply (@ Property_ran n), Property_Value; auto.
      rewrite H12; auto. }
    rewrite Minus_P2; auto. rewrite <-H15; auto.
  - unfold Infinitesimal in H1; unfold Limit_Seq in *; split; auto.
    destruct H1. destruct H1 as [H1 []]. intros. apply H2 in H5.
    destruct H5 as [N []]. exists N; split; auto. intros.
    assert ((x[n] - a) ∈ ℝ).
    { apply Plus_close; auto. destruct H0 as [H0 []]. apply H10.
      apply (@ Property_ran n), Property_Value; auto.
      rewrite H9; auto. apply Plus_neg1a; auto. }
    apply H6 in H8; auto. rewrite Minus_P2 in H8.
    assert (x[n] - a = \{\ λ u v, u ∈ ℕ /\ v ∈ ℝ /\ v = x[u] - a \}\ [n]).
    { apply Property_Fun; auto. apply AxiomII'; repeat split; auto.
      apply MKT49a; eauto. }
    rewrite <-H10 in H8; auto. apply H4. apply (@ Property_ran n),
    Property_Value; auto. rewrite H3; auto.
Qed.

(* 定义3：无穷大数列 *)
Definition Infiniteseries x := IsSeq x /\ (∀ M, M ∈ ℝ /\ 0 < M
  -> ∃ N, N ∈ ℕ /\ ∀ n, n ∈ ℕ -> N < n -> M < Abs [x[n]]).

(* 定义4：正无穷大数列 *)
Definition PosInfiniteseries x := IsSeq x /\ (∀ M, M ∈ ℝ /\ 0 < M
  -> ∃ N, N ∈ ℕ /\ ∀ n, n ∈ ℕ -> N < n -> M < x[n]).

(* 定义5：负无穷大数列 *)
Definition NegInfiniteseries x := IsSeq x /\ (∀ M, M ∈ ℝ /\ 0 < M
  -> ∃ N, N ∈ ℕ /\ ∀ n, n ∈ ℕ -> N < n -> x[n] < -M).
  
(* 2.2 收敛数列性质 *)

Corollary Minus_P3 : ∀ x y, x ∈ ℝ -> y ∈ ℝ -> - (x + y) = (- x) + (- y).
Proof.
  intros. assert ((- (1)) ∈ ℝ). 
  { apply Plus_neg1a; auto. apply one_in_R_Co; auto. }
  assert ((- (1)) · x ∈ ℝ).
  { rewrite <-PlusMult_Co3; auto. apply Plus_neg1a; auto. }
  assert (y · (-(1)) ∈ ℝ).
  { rewrite Mult_P4, <-PlusMult_Co3; auto. apply Plus_neg1a; auto. } 
  rewrite PlusMult_Co3, Mult_P4; auto with real.
  rewrite Mult_P5, Mult_P4, Plus_P4, Mult_P4; auto. symmetry.
  rewrite PlusMult_Co3, Plus_P4, PlusMult_Co3; auto with real.
Qed.

Corollary Minus_P4 : ∀ x, x ∈ ℝ -> - - x = x.
Proof.
  intros. rewrite PlusMult_Co3; auto with real.
  apply PlusMult_Co4; auto.
Qed.

Corollary RMult_eq : ∀ x y z, x ∈ ℝ -> y ∈ ℝ -> z ∈ ℝ -> 
  x = y -> x · z = y · z.
Proof.
  intros. rewrite H2; auto.
Qed.

Corollary RMult_eq' : ∀ x y z, x ∈ ℝ -> y ∈ ℝ -> z ∈ (ℝ ~ [0]) ->
  x · z = y · z -> x = y.
Proof.
  intros. New H1; apply MKT4' in H1 as [].
  assert (z⁻ ∈ (ℝ ~ [0])). { apply Mult_inv1; auto. }
  apply MKT4' in H5 as [].
  apply RMult_eq with (z:= z⁻) in H2; auto with real.
  do 2 rewrite <-Mult_P3 in H2; auto.
  rewrite Divide_P1 in H2; auto. do 2 rewrite Mult_P1 in H2; auto.
Qed. 

(* 定理2.2: 唯一性 *)
Theorem Theorem2_2 : ∀ x a b, IsSeq x -> a ∈ ℝ -> b ∈ ℝ
  -> Limit_Seq x a -> Limit_Seq x b -> a = b.
Proof.
  intros. apply NNPP; intro. apply Limit_Equal in H2 as Ha; auto. 
  assert (((Abs[b-a])/ (1 + 1)) ∈ ℝ  /\ 0 < ((Abs[b-a])/ (1 + 1))).
  { assert (0 < (1 + 1)) as Hb.
    { apply (Order_Co2 _ 1); auto with real.
      left; split; pose proof OrderPM_Co9; auto.
      apply (OrderPM_Co1 _ _ 1) in H5; auto with real.
      rewrite Plus_P4, Plus_P1 in H5; auto with real. destruct H5; auto. }
    assert ((1 + 1) ⁻ ∈ (ℝ ~ [0])) as Hc.
    { apply Mult_inv1. apply MKT4'; split; auto with real.
      apply AxiomII; split; eauto with real.
      intro. apply MKT41 in H5; eauto with real.
      rewrite H5 in Hb. destruct Hb. contradiction. }
    split.
    - apply Mult_close; auto with real. apply MKT4' in Hc; tauto.
    - assert (a ≤ b \/ b ≤ a). { destruct (Leq_P4 a b); auto. }
      destruct H5 as [|].
      + apply Plus_Leq with (z:=(-a)) in H5 as H6; auto with real. 
        rewrite Minus_P1 in H6; auto.
        assert (｜(b - a)｜ = (b - a)). { apply me_zero_Abs; auto with real. }
        rewrite H7. apply OrderPM_Co5; auto with real. apply MKT4' in Hc; tauto.
        left; split.
        * assert (a < b). { split; auto. }
          apply (OrderPM_Co1 _ _ (-a)) in H8; auto with real.
          rewrite Minus_P1 in H8; auto.
        * apply OrderPM_Co10; auto with real.
      + apply Plus_Leq with (z:=(-a)) in H5 as H6; auto with real.
        rewrite Minus_P1 in H6; auto; New H6. 
        apply le_zero_Abs in H7; auto with real.
        assert (｜(b - a)｜ = (- (b - a))). 
        { apply le_zero_Abs; auto with real. }
        assert ((- (b - a)) = (- b + a)).
        { rewrite Minus_P3; auto with real. rewrite Minus_P4; auto. }
        assert ((- b) + a = a + (- b)). { rewrite Plus_P4; auto with real. }
        rewrite H10 in H9. rewrite H9 in H8. rewrite H8.
        apply MKT4' in Hc as []. apply OrderPM_Co5; auto with real.
        left; split; New H5.
        * apply (Plus_Leq _ _ (-b)) in H13; auto with real.
          rewrite Minus_P1 in H13; auto.
          assert (b < a). { split; auto. }
          apply (OrderPM_Co1 _ _ (-b)) in H14; auto with real.
          rewrite Minus_P1 in H14; auto.
        * apply OrderPM_Co10; auto with real. } 
  destruct Ha as []. pose proof H5; destruct H5 as [].
  pose proof H8 as Hz; apply H7 in H8.
  assert (\{ λ u, u ∈ ℕ /\ x[u] ∈ U(b; (｜(b - a)｜ / (1 + 1))) \} ⊂ 
    \{ λ u, u ∈ ℕ /\ x[u] ∉ U(a; (｜(b - a)｜ / (1 + 1)))\} ).
  { unfold Included; intros. apply AxiomII in H10 as [_[]]. 
    apply AxiomII; repeat split; eauto.
    intro. apply AxiomII in H11 as [H11 []];
    apply AxiomII in H12 as [_[_]]. destruct H9, H14, H12.
    apply Abs_P4 in H14 as [], H12 as []; auto with real.
    assert (0 ≤ (b - a) \/ (b - a) ≤ 0).
    { destruct (Leq_P4 (b-a) 0) as [|]; auto with real. }
    assert (｜(b - a)｜ = b - a \/ ｜(b - a)｜ = a - b).
    { destruct H20 as [|].
      - left. apply me_zero_Abs; auto with real.
      - right. assert (｜(b - a)｜ = (- (b - a))). 
        { apply le_zero_Abs; auto with real. }
        assert ((- (b - a)) = (- b + a)).
        { rewrite Minus_P3, Minus_P4; auto with real. }
        rewrite H21; rewrite H22; apply Plus_P4; auto with real. }
    assert (0 < (1 + 1)) as Ha.
    { apply (Order_Co2 _ 1); auto with real.
      left; split; New OrderPM_Co9; auto.
      apply (OrderPM_Co1 _ _ 1) in H22; auto with real.
      rewrite Plus_P4, Plus_P1 in H22; auto with real. destruct H22; auto. } 
    assert ((1 + 1) ⁻ ∈ (ℝ ~ [0])) as Hb.
    { apply Mult_inv1. apply MKT4'; split; auto with real.
      apply AxiomII; split; eauto with real.
      intro. apply MKT41 in H22; eauto with real.
      rewrite H22 in Ha. destruct Ha; contradiction. }
    pose proof Hb as Hc; apply MKT4' in Hc as [Hd He].
    assert (a = ((1 + 1) · a) / (1 + 1)) as Hf.
    { assert (a · (1 + 1) = (1 + 1) · a).
      { rewrite Mult_P4; auto with real. }
      apply RMult_eq with (z:=(1 + 1)⁻) in H22; auto with real.
      rewrite <-Mult_P3, Divide_P1, Mult_P1 in H22; auto with real.
      apply MKT4'; split; auto with real.
      apply AxiomII; split; eauto with real. 
      intro. apply MKT41 in H23; eauto with real.
      rewrite H23 in Ha; destruct Ha; eauto. }
    assert (b = ((1 + 1) · b) / (1 + 1)) as Hg.
    { assert (b · (1 + 1) = (1 + 1) · b).
      { apply Mult_P4; auto with real. }
      apply RMult_eq with (z:=(1 + 1)⁻) in H22; auto with real.
      rewrite <-Mult_P3, Divide_P1, Mult_P1 in H22; auto with real.
      apply MKT4'; split; auto with real.
      apply AxiomII; split; eauto with real.
      intro. apply MKT41 in H23; eauto with real.
      rewrite H23 in Ha; destruct Ha; contradiction. }
    assert ((1 + 1) · a = a + a) as Hh.
    { rewrite Mult_P5, Mult_P4, Mult_P1; auto with real. }
    assert ((1 + 1) · b = b + b) as Hi.
    { rewrite Mult_P5, Mult_P4, Mult_P1; auto with real. }
    destruct H21 as [|].
    - rewrite H21 in *.
      (* b - ε *)
      apply (Plus_Leq _ _ b) in H14; auto with real;
      rewrite <-Plus_P3, (Plus_P4 (- b)), Minus_P1, Plus_P1 in H14; 
      auto with real.
      pattern b at 2 in H14; rewrite Hg in H14.
      rewrite PlusMult_Co3, Mult_P3, <-Mult_P5 in H14; auto with real.
      rewrite <-PlusMult_Co3, Minus_P3, Minus_P4 in H14; auto with real.
      rewrite Hi, Plus_P4, <-Plus_P3, Plus_P4 in H14; auto with real.
      rewrite Plus_P3, Plus_neg2, (Plus_P4 0), Plus_P1 in H14; auto with real.
      (* a + ε *) 
      apply (Plus_Leq _ _ a) in H19; auto with real; 
      rewrite <-Plus_P3, (Plus_P4 (-a)), Minus_P1, Plus_P1 in H19;
      auto with real.
      pattern a at 2 in H19; rewrite Hf in H19.
      rewrite <-Mult_P5, Hh in H19; auto with real.
      rewrite Plus_P4, (Plus_P4 b), Plus_P3 in H19; auto with real.
      rewrite <-(Plus_P3 a), Plus_neg2, Plus_P1 in H19; auto with real.
      (* elim *)
      assert (x [z] = (a + b) · ((1 + 1) ⁻)).
      { apply Leq_P2; auto with real. }
      elim H17. rewrite H22. pattern a at 2; rewrite Hf.
      rewrite PlusMult_Co3, Mult_P3, <-Mult_P5; auto with real.
      rewrite <-PlusMult_Co3, Hh, Minus_P3; auto with real.
      rewrite (Plus_P4 a), <-Plus_P3, (Plus_P3 a); auto with real.
      rewrite Plus_neg2, (Plus_P4 0), Plus_P1; auto with real.
      apply me_zero_Abs; auto.
    - rewrite H21 in *.
      (* a - ε *)
      apply (Plus_Leq _ _ a) in H12; auto with real;
      rewrite <-Plus_P3, (Plus_P4 ((- a))), Plus_neg2, (Plus_P1) in H12; 
      auto with real.
      pattern a at 2 in H12; rewrite Hf in H12.
      rewrite PlusMult_Co3, Mult_P3, <-PlusMult_Co3 in H12; auto with real.
      rewrite <-Mult_P5, Hh in H12; auto with real.
      rewrite Minus_P3, Minus_P4, (Plus_P4 (-a)) in H12; auto with real.
      rewrite <-Plus_P3,(Plus_P3 (-a)),(Plus_P4 (-a)) in H12; auto with real. 
      rewrite Plus_neg2, (Plus_P4 0), Plus_P1 in H12; auto with real.
      (* b + ε *)
      apply (Plus_Leq _ _ b) in H18; auto with real;
      rewrite <-Plus_P3, (Plus_P4 (- b)), Plus_neg2, Plus_P1 in H18;
      auto with real.
      pattern b at 2 in H18; rewrite Hg in H18.
      rewrite <-Mult_P5, Hi in H18; auto with real.
      rewrite <-Plus_P3, (Plus_P4 (- b)), <-Plus_P3 in H18; auto with real.
      rewrite Plus_neg2, Plus_P1, Plus_P4 in H18; auto with real.
      (* elim *)
      assert (x[z] = (b + a) · ((1 + 1)⁻)).
      { apply Leq_P2; auto with real. }
      elim H16. rewrite H22. pattern b at 2; rewrite Hg.
      rewrite PlusMult_Co3, Mult_P3, <-Mult_P5; auto with real.
      rewrite Hi, <-PlusMult_Co3, Minus_P3; auto with real.
      rewrite (Plus_P4 b), <-Plus_P3, (Plus_P3 b); auto with real.
      rewrite Plus_neg2, (Plus_P4 0), Plus_P1; auto with real.
      apply me_zero_Abs; auto. }
  apply finsub in H10; auto. unfold Limit_Seq in H3; destruct H3 as [].
  apply H11 in Hz as H12. destruct H12 as [N2 []]. apply finite_maxN in H10.
  destruct H10 as [N1]; unfold maxR in H10; destruct H10 as [H10 []].
  destruct (Leq_P4 N1 N2) as [Ha | Ha]; auto with real.
  - assert (N2 < (N2 + 1)).
    { pose proof H12; apply N_Subset_R in H16; apply Leq_P1 in H16.
      destruct OrderPM_Co9. apply (OrderPM_Co3 _ _ 0 1) in H16; auto with real.
      rewrite Plus_P1 in H16; auto with real. split; auto.
      intro. assert (N2 - N2 = N2 + 1 - N2). { rewrite <-H19; auto. }
      rewrite Plus_neg2, (Plus_P4 N2), <-Plus_P3, Plus_neg2, Plus_P1 in H20;
      auto with real. }
      apply H13 in H16; auto with real.
    assert (N2 + 1 ∈ \{ λ u, u ∈ ℕ /\ x[u] ∈ U(b; (｜(b - a)｜/(1 + 1))) \}).
    { destruct H6 as [H6 []]; destruct H16.
      assert (x[N2 + 1] ∈ ℝ).
      { apply H18. apply (@ Property_ran (N2 + 1)), Property_Value; auto.
        rewrite H17; auto with real. }
      apply AxiomII; repeat split; eauto with real.
      apply AxiomII; repeat split; eauto. }
    apply H15 in H17. apply AxiomII in H14 as [_[]].
    assert (N1 < N2 + 1). { apply Nat_P4b; auto. }
    destruct H19. elim H20. apply Leq_P2; auto with real.
  - apply AxiomII in H14 as [_[]].
    assert (N2 < N1 + 1). { apply Nat_P4b; auto. }
    pose proof H16. apply H13 in H17; auto with real.
    assert (N1 + 1 ∈ \{ λ u, u ∈ ℕ /\ x[u] ∈ U(b; (｜(b - a)｜/(1 + 1))) \}).
    { destruct H17; auto. destruct H6 as [H6 []].
      assert (x[N1 + 1] ∈ ℝ).
      { apply H21, (@ Property_ran (N1 + 1)), Property_Value; auto. 
        rewrite H20; auto with real. }
      apply AxiomII; repeat split; eauto with real.
      apply AxiomII; repeat split; eauto with real. }
    assert (N1 < N1 + 1).
    { apply Nat_P4b; auto; apply Leq_P1; auto with real. }
    apply H15 in H19. destruct H20. elim H21. apply Leq_P2; auto with real.
  - assert (N2 < N2 + 1).
    { apply Nat_P4b; auto; apply Leq_P1; auto with real. }
    apply H13 in H14; auto with real. apply NEexE; exists (N2 + 1).
    destruct H6 as [H6 []]; destruct H14.
    assert (x[N2 + 1] ∈ ℝ).
    { apply H16. apply (@ Property_ran (N2 + 1)), Property_Value; auto.
      rewrite H15; auto with real. }
    apply AxiomII; repeat split; eauto with real.
    apply AxiomII; repeat split; eauto with real.
  - unfold Included; intros. apply AxiomII in H14; tauto.
Qed.

Corollary Max_nat_3 : ∀ N0 N1 N2, N0 ∈ ℕ -> N1 ∈ ℕ -> N2 ∈ ℕ
  -> (∃ N, N ∈ ℕ /\ N0 < N /\ N1 < N /\ N2 < N).
Proof.
  intros. destruct (Leq_P4 N0 N1) as [Ha | Ha]; auto with real.
  - destruct (Leq_P4 N1 N2) as [Hb | Hb]; auto with real.
    + exists (N2 + 1).
      assert (N0 < N2 + 1) as []. 
      { apply Nat_P4b; auto; apply (Leq_P3 _ N1 _); auto with real. }
      assert (N1 < N2 + 1) as []. { apply Nat_P4b; auto. }
      assert (N2 < N2 + 1) as [].
      { apply Nat_P4b; auto; apply Leq_P1; auto with real. }
      repeat split; auto with real.
    + exists (N1 + 1).
      assert (N0 < N1 + 1) as []. { apply Nat_P4b; auto. }
      assert (N1 < N1 + 1) as []. 
      { apply Nat_P4b; auto; apply Leq_P1; auto with real. }
      assert (N2 < N1 + 1) as []. { apply Nat_P4b; auto. }
      repeat split; auto with real.
  - destruct (Leq_P4 N0 N2) as [Hb | Hb]; auto with real.
    + exists (N2 + 1).
      assert (N0 < N2 + 1) as []. { apply Nat_P4b; auto. }
      assert (N1 < N2 + 1) as [].
      { apply Nat_P4b; auto. apply (Leq_P3 _ N0 _); auto with real. }
      assert (N2 < N2 + 1) as [].
      { apply Nat_P4b; auto; apply Leq_P1; auto with real. }
      repeat split; auto with real.
    + exists (N0 + 1).
      assert (N0 < N0 + 1) as [].
      { apply Nat_P4b; auto; apply Leq_P1; auto with real. }
      assert (N1 < N0 + 1) as []. { apply Nat_P4b; auto. }
      assert (N2 < N0 + 1) as []. { apply Nat_P4b; auto. }
      repeat split; auto with real.
Qed.

Corollary Ntrans : ∀ x y z, x ∈ ℕ -> y ∈ ℕ -> z ∈ ℕ -> x < y -> y < z -> x < z.
Proof.
  intros. apply Nat_P4 in H2, H3; auto with real.
  assert (y ≤ y + 1).
  { apply Nat_P4b; auto. apply Leq_P1; auto with real. }
  assert (x + 1 ≤ z).
  { apply (Leq_P3 _ y _); auto with real.
    apply (Leq_P3 _ (y + 1) _); auto with real. }
  apply Nat_P4a in H5; auto.
Qed.

(* 定理2.6 迫敛性 *)
Theorem Theorem2_6 : ∀ x y z a, IsSeq x -> IsSeq y -> a ∈ ℝ
  -> Limit_Seq x a -> Limit_Seq y a -> IsSeq z
  -> (∃ N0, N0 ∈ ℕ /\ ∀ n, n ∈ ℕ -> N0 < n -> (x[n] ≤ z[n] /\ z[n] ≤ y[n]))
  -> Limit_Seq z a.
Proof.
  split; auto; intros. destruct H2 as [Ha Hb], H3 as [Hc Hd].
  destruct Ha as [Ha []], Hc as [Hc []]. pose proof H6 as [He Hf]; destruct Hf.
  apply Hb in H6 as Hg; apply Hd in H6 as Hh.
  destruct H5 as [N0 [H5]], Hg as [N1 [Hg]], Hh as [N2 [Hh]].
  pose proof (Max_nat_3 _ _ _ H5 Hg Hh) as [N [Hi [Hj [Hk Hl]]]].
  exists N; split; auto; intros. apply (Ntrans _ _ n) in Hj; auto;
  apply (Ntrans _ _ n) in Hk; auto; apply (Ntrans _ _ n) in Hl; auto.
  assert (x[n] ∈ ℝ) as Hm.
  { apply H3, (@ Property_ran n), Property_Value; auto. rewrite H2; auto. } 
  assert (y[n] ∈ ℝ) as Hn.
  { apply H8, (@ Property_ran n), Property_Value; auto. rewrite H7; auto. }
  assert (z[n] ∈ ℝ) as Ho.
  { destruct H4. destruct H16 as [].
    apply H17, (@ Property_ran n), Property_Value; auto. rewrite H16; auto. }
  apply H11 in Hj as []; apply H12 in Hk; apply H13 in Hl; auto.
  destruct Hk, Hl. apply Abs_P4 in H18 as [], H20 as []; auto with real.
  apply (Plus_Leq _ _ a) in H18; auto with real.
  rewrite <-Plus_P3,(Plus_P4 (-(a))),Plus_neg2,Plus_P1 in H18; auto with real.
  apply (Plus_Leq _ _ a) in H23; auto with real.
  rewrite <-Plus_P3,(Plus_P4 (-(a))),Plus_neg2,Plus_P1 in H23; auto with real.
  split. apply Abs_P4; auto with real. split.
  - apply (Leq_P3 _ _ (z[n])) in H18; auto with real.
    apply (Plus_Leq _ _ (-(a))) in H18; auto with real.
    rewrite <-Plus_P3, Minus_P1, Plus_P1 in H18; auto with real.
  - apply (Leq_P3 (z[n]) _ _) in H23; auto with real.
    apply (Plus_Leq _ _ (-(a))) in H23; auto with real.
    rewrite <-Plus_P3, Minus_P1, Plus_P1 in H23; auto with real.
  - intro. destruct (Leq_P4 (z[n] - a) 0); auto with real.
    + rewrite le_zero_Abs, Minus_P3, Minus_P4 in H24; auto with real.
      rewrite <-H24, Minus_P3, Minus_P4, <-Plus_P3, (Plus_P4 (-(a))),
      Plus_neg2, Plus_P1 in H18; auto with real. 
      assert (x[n] = z[n]). { apply Leq_P2; auto. }
      elim H19. rewrite H26, <-H24.
      assert (- z[n] + a = - (z[n] - a)).
      { rewrite Minus_P3, Minus_P4; auto with real. }
      rewrite H27; apply le_zero_Abs; auto with real.
    + rewrite me_zero_Abs in H24; auto with real.
      rewrite <-H24, <-Plus_P3, (Plus_P4 (-(a))), Plus_neg2, Plus_P1 in H23;
      auto with real. assert (y[n] = z[n]). { apply Leq_P2; auto. }
      elim H21. rewrite H26, <-H24. apply me_zero_Abs; auto with real.
Qed.
