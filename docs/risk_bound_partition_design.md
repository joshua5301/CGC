# 위험 상계 기반 분할 알고리즘 설계

작성일: 2026-09-25

상태: 이 문서는 이론 설계와 그 범위를 기록한다. 실제 구현은 `src/risk_partition.py`와 `src/risk_experiment.py`에 추가했다. 사용자 요청에 따라 로컬 학습과 smoke test는 실행하지 않는다. 실제 그래프 데이터의 성능, 속도, 신규성은 검증하지 않았다.

## 실제 구현에 적용한 실험 조건

사용자가 최종 평가를 2-layer GCN과 균등 soft-label CE로 고정하도록 요청했다. 아래의 질량 가중 CE 및 norm-constrained 선형 학생은 이론적 유도 조건이며, 실제 성능 평가에 사용하지 않는다. 따라서 실제 GCN 결과에 이 정리의 보장이 적용된다고 주장하지 않는다.

구현에서는 특징을 중심화하고 RMS 정규화한 공간에서 J를 최적화한 뒤 대표 특징을 원래 공간으로 복원한다. B는 이 정규화 공간의 목적함수 계수다. GPU 후보 계산은 배치로 수행하며, 배치의 정확한 전체 J가 감소할 때만 이동한다. 실패한 배치는 재귀적으로 분할해 다시 평가한다. 이는 순차 이동 설계를 GPU에 맞춘 변경이다.

Optuna는 validation만 사용한다. 선택된 응축 데이터 하나를 고정하고 서로 다른 GCN 초기화 seed로 최종 test를 평가한다. 실행용 Python 스크립트는 추가하지 않고, 업데이트·설정·라이브러리 호출을 하나의 Colab 셀로 제공한다. 탐색 가능한 항목은 B, teacher_kernel, gamma, T, basis, dropout, lr, weight_decay이다. 교사 설정은 해당 trial의 실제 라벨 생성과 응축 캐시 키에 반영한다.

## 1. 결정 사항

첫 버전은 GRIP을 호출하지 않는다. 위험 상계에서 유도한 특징–라벨 결합 공간에서 D² seeding으로 분할을 구성한 뒤, 원래 목적함수의 정확한 변화량으로 단일 노드 이동을 수행한다.

초기 구성은 더 느슨한 상계 U를 사용하고, 이후 개선은 J를 직접 감소시킨다. 초기 구성까지 J의 단조 감소를 주장하지 않는다. 첫 버전에서 spectral 목적함수를 trace 목적함수로 완화한다는 점도 명시한다.

출력은 대표 특징, 평균 소프트 라벨, 셀 질량이다. 이론에 맞는 주 평가 모델은 동일한 고정 특징 공간 위의 norm-constrained 선형 softmax 학생이다. 기존 2층 GCN은 별도의 전이 실험으로만 평가한다.

## 2. 이론적 설정

- 입력: 고정 특징 h_t ∈ R^d, 고정 라벨 분포 q_t ∈ Δ^(K-1), t=1,...,N.
- 예산: 1 ≤ m ≤ N. 모든 셀은 비어 있지 않아야 한다.
- 학생: p_W(h)=softmax(Wᵀh), ||W||_F ≤ B, B>0.
- 편향이 필요하면 특징에 상수 1을 추가하고 해당 가중치도 같은 norm 제약에 포함한다.
- 이 위험은 주어진 N개 노드의 경험적 CE이며 population/test risk 보장이 아니다.

셀별 통계:

    n_j = |C_j|, s_j = Σ_(t∈C_j) h_t, r_j = Σ_(t∈C_j) q_t
    c_j = s_j/n_j, y_j = r_j/n_j, π_j = n_j/N

원본 목적함수와 응축 목적함수:

    R(W) = (1/N) Σ_t CE(q_t, p_W(h_t))
    R_P(W) = Σ_j π_j CE(y_j, p_W(c_j))

여기서 대표 특징은 기하학적 중앙값이 아니라 산술평균이고, 학습은 균등 셀 CE가 아니라 질량 가중 CE이다.

다음 통계를 정의한다. E라는 이름은 기존 대화의 전역 교차 공분산 C_P와 같다.

    S(P) = (1/N) Σ_t (h_t-c_a(t))(h_t-c_a(t))ᵀ
    V(P) = tr S(P)
    M = (1/N) Σ_t h_t q_tᵀ
    E(P) = M - Σ_j π_j c_j y_jᵀ

E는 셀별 공분산의 가중합이다. 셀별 norm의 합으로 바꾸면 다른, 더 느슨한 목적함수가 된다. 전역 상쇄는 선형 학생의 전체 CE 관점에서 정당하지만 비선형 학생에 그대로 일반화되지 않는다.

## 3. 학습 후 위험 보장과 선택한 목적함수

A_W(h)=log Σ_k exp(w_kᵀh)라 하면 정확히

    R(W)-R_P(W) = G_P(W)-<W,E(P)>_F
    G_P(W) = (1/N)Σ_t A_W(h_t)-Σ_j π_j A_W(c_j)
    0 ≤ G_P(W) ≤ (1/4) tr(WᵀS(P)W)

가 성립한다. 마지막 부등식은 log-sum-exp의 Hessian이 spectral norm 1/2 이하인 것과 셀 내 평균 편차가 0인 것으로부터 나온다.

W*가 동일 norm ball 위 원본 목적함수의 최적점이고, W_hat이 응축 목적함수를 ε_opt 이내로 최소화했다면

    R(W_hat)-R(W*)
      ≤ (B²/4) λ_max(S(P)) + 2B ||E(P)||_F + ε_opt
      ≤ (B²/4) V(P) + 2B ||E(P)||_F + ε_opt.

증명: 위험 차이에 R_P(W_hat), R_P(W*)를 더하고 뺀다. 가운데 항은 ε_opt 이하이다. 나머지는 G_P(W_hat)-G_P(W*)-<W_hat-W*,E>이며 G_P(W*)≥0, ||W_hat-W*||_F≤2B를 적용한다.

첫 버전의 목적함수는

    α = B²/4, β = 2B
    J(P) = α V(P) + β ||E(P)||_F.

추가 KL 항이나 임의의 label weight를 넣지 않는다. B는 클러스터링 가중치와 실제 학생 함수군을 함께 결정하는 값이다. B를 고정하지 않고 단지 J가 작다는 이유로 작은 B를 선택하면 비교하는 최적 학생 자체가 달라진다. B는 validation으로 선택하고 동일 B에서 원본/응축 학생을 비교한다.

J 감소는 보장된 상계의 감소이며 실제 위험이나 정확도의 매 단계 개선을 의미하지 않는다.

## 4. GRIP 없이 초기 분할 구성

라벨의 셀 내 분산을 V_q(P)=(1/N)Σ_t ||q_t-y_a(t)||²라 하자. Cauchy–Schwarz와 Young 부등식으로, 모든 η>0에 대해

    ||E(P)||_F ≤ sqrt(V(P)V_q(P))
    J(P) ≤ (α+Bη)V(P)+(B/η)V_q(P) = U_η(P).

따라서 다음 결합 공간의 squared k-means distortion은 U_η와 정확히 같다.

    z_t = [sqrt(α+Bη) h_t ; sqrt(B/η) q_t].

이 연결은 상계에 근거한 초기화라는 의미이며, U를 줄인다고 J도 항상 감소한다는 의미는 아니다.

η 초기값은 전체를 한 셀로 보았을 때의 V_h^0, V_q^0에서 정한다.

    η_0 = sqrt(V_q^0 / V_h^0).

이는 한 셀에서 U_η를 최소화하는 해이다. 최종 m개 셀에 대한 최적 η라는 주장은 하지 않는다. 첫 버전은 η를 이후에 다시 조정하지 않는다.

구성 절차:

1. z 공간에서 첫 노드 인덱스를 균등 선택한다.
2. 가장 가까운 선택된 seed까지의 제곱 거리에 비례해 다음 seed를 선택한다.
3. m개 seed를 선택한 뒤 각 노드를 가장 가까운 seed에 할당한다.
4. 각 셀의 c_j, y_j, π_j를 원본 h,q에서 다시 계산한다.
5. J를 계산하고 이후 exact relocation 단계로 넘어간다. 초기 U에 대한 Lloyd 반복은 첫 버전에 넣지 않는다.

동일 z 때문에 모든 남은 거리가 0이 되면 미선택 인덱스에서 seed를 선택한다. 동일 seed 좌표 간 tie는 seed 자신의 셀 소속을 유지하는 방식으로 처리하여 m개 비어 있지 않은 셀을 만든다. 좌표가 같은 seed의 tie 분배는 U를 증가시키지 않는다.

퇴화 입력:

- V_h^0=0: 모든 특징이 같으므로 어떤 비어 있지 않은 분할도 V=E=0이다. 결정적인 균형 분할을 사용한다.
- V_q^0=0, V_h^0>0: 모든 라벨 분포가 같아 E=0이다. 특징 공간에서 D² seeding을 한다.
- m=1: 단일 셀을 반환한다.
- m=N: singleton을 반환하며 J=0이다.
- 잘못된 simplex 라벨/비유한 입력/잘못된 예산은 명시적으로 거부한다. 큰 오류를 clamp로 숨기지 않는다.

D² seeding 자체는 기존 알고리즘이다. 기존 근사 보장은 여기서는 U에 적용되며 J의 최적해에 대한 같은 근사비를 주장할 수 없다.

## 5. 전체 목적함수에 따른 노드 이동

한 노드 t를 현재 셀 a에서 셀 b로 옮긴다. n_a>1이고 b≠a인 이동만 허용한다. 이동 전 중심으로 다음을 계산한다.

    u_a=h_t-c_a, v_a=q_t-y_a, k_a=n_a/[N(n_a-1)]
    u_b=h_t-c_b, v_b=q_t-y_b, k_b=n_b/[N(n_b+1)]

모든 관련 평균을 다시 계산했을 때의 변화량은 정확히

    ΔV = -k_a ||u_a||² + k_b ||u_b||²
    ΔE = -k_a u_a v_aᵀ + k_b u_b v_bᵀ
    ΔJ = α ΔV + β (||E+ΔE||_F-||E||_F).

각 노드에 대해 모든 목적 셀을 평가해 가장 작은 ΔJ를 선택하고, 수치 허용오차보다 확실하게 음수이면 즉시 이동을 확정한다. 그 뒤 n,s,r,V,E 및 두 중심을 갱신한다. 다음 노드는 갱신된 상태를 기준으로 평가한다.

핵심: 고정 중심까지의 거리로 평가하지 않는다. 노드 이동에 따른 출발/도착 셀의 평균 변화까지 포함한다. 셀별 비용이 독립적이지 않으므로 여러 노드의 음수 ΔJ를 단순히 더하거나 동시에 실행하지 않는다.

첫 버전은 Hartigan 방식의 순차 best-improvement relocation을 사용한다. 이 최적화 도구 자체는 신규 기여가 아니다.

## 6. 계산과 수치 처리

유지할 통계:

    assignment: N
    counts: m
    feature_sums: m×d
    label_sums: m×K
    E: d×K
    V: scalar

상태를 다시 계산할 때는 다음 식을 사용할 수 있다.

    V = (1/N)Σ_t ||h_t||² - (1/N)Σ_j ||s_j||²/n_j
    E = M - (1/N)Σ_j s_j r_jᵀ/n_j.

안정성 검증용으로 편차를 직접 합산하는 별도 계산 경로를 둔다. 처음에는 정확성을 위해 float64 통계를 사용한다. 각 sweep 끝에 통계를 재계산하고 incremental 상태, 라벨 정규화, Σπ=1, 빈 셀 유무, J 감소 여부를 확인한다.

후보별 d×K 행렬 할당은 rank-one inner-product 식으로 피할 수 있다.

    E_minus=E-k_a u_a v_aᵀ
    ||E_minus+k_b u_b v_bᵀ||_F²
      = ||E_minus||_F² + 2k_b u_bᵀE_minus v_b
        + k_b²||u_b||²||v_b||².

단순 구현의 sweep 비용은 O(N m d K)이다. 작은 그래프에서 정합성을 확인한 뒤 최적화한다. 대규모 데이터를 Python 노드 루프로 빠르게 처리할 수 있다고 가정하지 않는다.

향후 가속은 후보 셀 shortlist, compiled sequential kernel, 일괄 제안 후 전체 ΔJ 재평가 등을 검토한다. shortlist를 쓰면 후보 집합에 대한 국소 정지만 보장한다. batch 수용은 정확한 전체 변화량을 계산해야 한다. power iteration의 Rayleigh quotient를 λ_max의 상계처럼 사용하는 것은 금지한다.

## 7. 종료와 보장

- 모든 목적 셀을 검사한 한 sweep에서 허용오차를 넘는 개선이 없으면 종료한다.
- 이때 비어 있지 않은 m개 셀을 유지하는 단일 노드 이동에 대한 허용오차 수준의 국소 정지 상태이다.
- strict decrease와 유한한 분할 수 때문에 exact arithmetic에서 종료한다. 실용적인 sweep 수의 상한은 별개다.
- max_sweeps 도달은 converged=False로 반환한다. 이것을 수렴으로 표시하지 않는다.
- m=N이면 처음부터 종료한다. m=1도 이동 후보가 없어 종료한다.
- 여러 seed를 사용할 경우 학생 test 성능이 아니라 J가 가장 작은 분할을 선택한다.

## 8. 이론에 맞는 학생 학습

반환 데이터:

    x=c, y=y, sample_weight=π, assignment, counts,
    objective_history, V, moment_error_norm, converged, sweeps, seed, B.

학습 손실은 반드시 Σ_j π_j CE(y_j,p_W(c_j))이다. π의 합이 1이므로 추가로 m으로 나누지 않는다.

첫 검증 모델은 고정 h 공간의 선형 softmax이며 Frobenius ball에 projected optimization을 한다. 단순 L2 weight decay를 ||W||≤B 제약과 동일하다고 취급하지 않는다. 추가 ridge/dropout 없이 제시한 목적함수와 맞춘다.

볼 제약의 경계에서는 gradient가 0이 아닐 수 있으므로 gradient norm만으로 수렴 판정하지 않는다. 볼 위 convex 목적함수의 Frank–Wolfe gap

    gap(W) = <∇R_P(W),W>_F + B ||∇R_P(W)||_F

은 suboptimality의 상계이다. feasibility와 gap≤tolerance를 함께 확인하고 ε_opt로 기록한다. 원본 전체 데이터 학생에도 동일 B와 수렴 기준을 사용한다.

현재 저장소의 H2 전파와 교사 생성은 재사용할 수 있지만 src/partition.py, metric_alpha, sgc_refine은 이 알고리즘의 경로에서 호출하지 않는다. 기존 GCN 평가에는 이 선형 학생 정리를 적용하지 않는다.

## 9. 구현 단위 제안

별도 모듈을 만들고 기존 GRIP 기본 실행을 유지한다.

    src/risk_partition.py
      validate_inputs
      compute_partition_stats
      seed_from_bound
      relocation_delta
      optimize_partition

    src/bounded_student.py
      fit_weighted_softmax_ball
      frank_wolfe_gap
      evaluate_empirical_risk

먼저 독립된 실행 진입점에서 검증한 뒤 main.py 선택 옵션으로 통합한다. 함수는 전역 args에 의존하지 않게 한다. 초기에는 spectral/merge-split/GNN-guided refinement를 섞지 않는다.

## 10. 필요한 검증과 연구 판단 기준

수학·구현 검증:

1. direct recomputation과 incremental ΔV, ΔE, ΔJ 일치.
2. J≤U_η 및 위험의 exact decomposition 일치.
3. norm ball 위 임의 W에서 Jensen gap의 상하계 확인.
4. 매 수용 이동의 J 감소, 정확한 m 유지, simplex/weight 보존.
5. singleton, 중복 특징, 일정 라벨, 일정 특징, m=1/N 처리.
6. 작은 입력의 전 이동 검사로 종료 조건 확인.
7. 학습된 학생에서 R(W_hat)-R(W*)≤J+ε_opt 확인. 수치 검산은 증명의 대체가 아니다.

비교 실험은 동일 h,q,m,B,학생 최적화 기준을 사용한다. Euclidean k-means, 결합 공간 초기화만, exact relocation까지의 세 조건을 먼저 비교한다. GRIP 결과도 같은 가중 선형 학생으로 재평가하는 통제 비교와 원래 파이프라인 비교를 분리한다.

위험/정확도 외에 J, V, ||E||, 학생 norm, 최적화 gap, 시간, 메모리를 기록한다. 특히 J 감소가 학습 후 실제 excess risk 감소와 연결되는지를 확인한다. 상계만 줄고 실제 risk가 일관되게 개선되지 않으면 목적함수의 실용성을 재검토한다.

교사 q를 쓴 경우 경험적 teacher risk에 대한 정리다. 정답 risk에는 별도의 교사 오차가 필요하다. test 라벨을 분할 목적함수나 B 선택에 사용하지 않는다. GCN 전이 성능은 별도 가설이다.

## 11. 설계 단계의 수치 검산 기록

저장소 구현 없이 독립된 NumPy 계산으로 N=90,d=5,K=3,m=7,B=2의 합성 사례 6개를 검사했다.

- 분석적 ΔJ와 전체 재계산의 최대 차이: 약 1.4×10^-15.
- J≤U_η 위반: 없음.
- 모든 사례에서 7개 비어 있지 않은 셀을 유지하고 3~10 sweeps에 정지했다.
- J: 3.822044→2.915155, 2.604061→2.499712, 3.475612→2.861773,
  3.714512→2.719980, 3.723574→2.998067, 3.532782→2.764125.

이는 업데이트 수식의 정합성 검산이다. 그래프 벤치마크 성능 향상의 증거가 아니다.

## 12. 관련 기존 도구와 신규성 범위

- Arthur & Vassilvitskii, k-means++: https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf
- Telgarsky & Vattani, Hartigan's Method: https://proceedings.mlr.press/v9/telgarsky10a.html
- Munteanu et al., On Coresets for Logistic Regression: https://arxiv.org/abs/1805.08571

Seeding, relocation, coreset 관점은 기존 연구다. 검토할 연구 기여는 graph-propagated feature와 soft-label 환경에서 학습 후 위험 상계가 특정 분할 통계 및 효율적 갱신으로 이어지는지, 그리고 그 설계가 실용적으로 유효한지에 있다. 선행연구 대비 신규성은 별도 확인이 필요하다.
