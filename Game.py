import numpy as np
import matplotlib.pyplot as plt

#MARKET PARAMETERS  — change any value to re-run

a  = 100   # demand intercept  P = a - b(qA + qB)
b  = 1     # demand slope
cA = 10    # Firm A marginal cost
cB = 10    # Firm B marginal cost


#PAYOFF FUNCTIONS

def market_price(qA, qB):
    return a - b * (qA + qB)

def profit_A(qA, qB):
    return (market_price(qA, qB) - cA) * qA

def profit_B(qA, qB):
    return (market_price(qA, qB) - cB) * qB


#BEST RESPONSE FUNCTIONS
#    Derived by maximising each firm's profit:
#    dπA/dqA = 0  →  BR_A(qB) = (a - cA - b·qB) / 2b
#    dπB/dqB = 0  →  BR_B(qA) = (a - cB - b·qA) / 2b

def br_A(qB):
    return max((a - cA - b * qB) / (2 * b), 0)

def br_B(qA):
    return max((a - cB - b * qA) / (2 * b), 0)


#NASH EQUILIBRIUM  (analytical solution)
#Solve BR_A and BR_B simultaneously:
#qA* = (a - 2cA + cB) / 3b
#qB* = (a - 2cB + cA) / 3b

def nash_equilibrium(a_=a, b_=b, cA_=cA, cB_=cB):
    qA = (a_ - 2*cA_ + cB_) / (3*b_)
    qB = (a_ - 2*cB_ + cA_) / (3*b_)
    P  = a_ - b_*(qA + qB)
    return qA, qB, P, (P - cA_)*qA, (P - cB_)*qB


#STACKELBERG EQUILIBRIUM  (Leader A, Follower B)
# A maximises profit substituting BR_B into its own profit:
#qA* = (a - 2cA + cB) / 2b      [leader]
#qB* = BR_B(qA*)                 [follower]

def stackelberg_equilibrium(a_=a, b_=b, cA_=cA, cB_=cB):
    qA = (a_ - 2*cA_ + cB_) / (2*b_)
    qB = max((a_ - cB_ - b_*qA) / (2*b_), 0)
    P  = a_ - b_*(qA + qB)
    return qA, qB, P, (P - cA_)*qA, (P - cB_)*qB


#NORMAL FORM REPRESENTATION  (discrete payoff matrix)

def normal_form(n=4):
    q_max  = (a - min(cA, cB)) / b
    strats = np.linspace(0, q_max * 0.75, n)
    PA = np.array([[profit_A(qA, qB) for qB in strats] for qA in strats])
    PB = np.array([[profit_B(qA, qB) for qB in strats] for qA in strats])
    return strats, PA, PB


#EXTENSIVE FORM  (game tree — Stackelberg two-stage)

class Node:
    def __init__(self, player, quantity=None, payoff=None):
        self.player   = player
        self.quantity = quantity
        self.payoff   = payoff      # (piA, piB) at terminal nodes
        self.children = []

def build_game_tree(n=3):
    q_max   = (a - min(cA, cB)) / b
    actions = np.linspace(10, q_max * 0.6, n)
    root    = Node("A")
    for qA in actions:
        a_node = Node("B", quantity=round(qA, 1))
        for qB in actions:
            payoff = (round(profit_A(qA, qB), 1), round(profit_B(qA, qB), 1))
            a_node.children.append(Node("terminal", quantity=round(qB, 1), payoff=payoff))
        root.children.append(a_node)
    return root

def print_tree(node, depth=0):
    pad = "    " * depth
    if node.player == "A":
        print(f"{pad}[Firm A chooses]")
        for child in node.children:
            print_tree(child, depth)
    elif node.player == "B":
        print(f"{pad}  qA = {node.quantity}")
        print(f"{pad}  └─ [Firm B responds]")
        for child in node.children:
            print_tree(child, depth + 1)
    else:
        print(f"{pad}qB = {node.quantity}  →  payoff (πA, πB) = {node.payoff}")


#DOMINATED STRATEGY CHECK

def dominated_strategies():
    strats, PA, PB = normal_form(n=5)
    dom_A = [round(strats[i], 1) for i in range(len(strats))
             if all(PA[i, j] >= PA[k, j]
                    for j in range(len(strats))
                    for k in range(len(strats)) if k != i)]
    dom_B = [round(strats[j], 1) for j in range(len(strats))
             if all(PB[i, j] >= PB[i, k]
                    for i in range(len(strats))
                    for k in range(len(strats)) if k != j)]
    return dom_A, dom_B


#SCENARIOS

scenarios = [
    {"name": "Baseline",       "a": 100, "b": 1, "cA": 10, "cB": 10, "type": "cournot"},
    {"name": "Demand Shift",   "a": 130, "b": 1, "cA": 10, "cB": 10, "type": "cournot"},
    {"name": "Cost Variation", "a": 100, "b": 1, "cA": 30, "cB": 10, "type": "cournot"},
    {"name": "Stackelberg",    "a": 100, "b": 1, "cA": 10, "cB": 10, "type": "stackelberg"},
]


#TERMINAL OUTPUT

print("=" * 52)
print("GAME THEORY ANALYSIS")
print("=" * 52)
print(f"   Parameters: a={a}, b={b}, cA={cA}, cB={cB}")

print("\n── BEST RESPONSE FUNCTIONS ──────────────────────")
print("   BR_A(qB) = (a - cA - b·qB) / 2b")
print("   BR_B(qA) = (a - cB - b·qA) / 2b")
print(f"   BR_A(0)  = {br_A(0):.2f}  |  BR_B(0) = {br_B(0):.2f}")

print("\n── NASH EQUILIBRIUM (Cournot) ───────────────────")
qA_ne, qB_ne, P_ne, piA_ne, piB_ne = nash_equilibrium()
print(f"   qA* = {qA_ne:.2f}   qB* = {qB_ne:.2f}   P* = {P_ne:.2f}")
print(f"   πA  = {piA_ne:.2f}   πB  = {piB_ne:.2f}")

print("\n── STACKELBERG EQUILIBRIUM (A leads) ────────────")
qA_sk, qB_sk, P_sk, piA_sk, piB_sk = stackelberg_equilibrium()
print(f"   qA* = {qA_sk:.2f}   qB* = {qB_sk:.2f}   P* = {P_sk:.2f}")
print(f"   πA  = {piA_sk:.2f}   πB  = {piB_sk:.2f}")

print("\n── NORMAL FORM (4×4 payoff matrix) ──────────────")
strats, PA, PB = normal_form(n=4)
print(f"   Strategy grid: {[round(s,1) for s in strats]}")
print("   Payoff matrix πA:")
print(np.round(PA, 1))
print("   Payoff matrix πB:")
print(np.round(PB, 1))

print("\n── DOMINATED STRATEGIES ─────────────────────────")
dom_A, dom_B = dominated_strategies()
print(f"   Firm A dominant qty: {dom_A if dom_A else 'None (no pure dominant)'}")
print(f"   Firm B dominant qty: {dom_B if dom_B else 'None (no pure dominant)'}")
print("   Note: q=0 is weakly dominated for both firms.")
print(f"   Any q > {br_A(0):.1f} is dominated for Firm A (above BR at qB=0)")
print(f"   Any q > {br_B(0):.1f} is dominated for Firm B (above BR at qA=0)")

print("\n── EXTENSIVE FORM (Game Tree) ───────────────────")
tree = build_game_tree(n=3)
print_tree(tree)

print("\n── SCENARIO SIMULATIONS ─────────────────────────")
print(f"   {'Scenario':<16} {'qA*':>6} {'qB*':>6} {'P*':>6} {'πA':>8} {'πB':>8}")
print("   " + "-" * 52)
for sc in scenarios:
    fn = stackelberg_equilibrium if sc["type"] == "stackelberg" else nash_equilibrium
    qA_, qB_, P_, pA_, pB_ = fn(sc["a"], sc["b"], sc["cA"], sc["cB"])
    print(f"   {sc['name']:<16} {qA_:>6.2f} {qB_:>6.2f} {P_:>6.2f} {pA_:>8.2f} {pB_:>8.2f}")

print("\n── NE VERIFICATION ──────────────────────────────")
print(f"   BR_A(qB*={qB_ne:.2f}) = {br_A(qB_ne):.2f}  (should = qA*={qA_ne:.2f})")
print(f"   BR_B(qA*={qA_ne:.2f}) = {br_B(qA_ne):.2f}  (should = qB*={qB_ne:.2f})")
print("=" * 52)

#Visulisation need.
q = np.linspace(0, 70, 400)

plt.figure(figsize=(6,5))

plt.plot([br_A(qB) for qB in q], q, label="BR_A")
plt.plot(q, [br_B(qA) for qA in q], label="BR_B")

plt.scatter(qA_ne, qB_ne, s=100, label="Nash Equilibrium")

plt.title("Best Response Curves")
plt.xlabel("Firm A Quantity")
plt.ylabel("Firm B Quantity")

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# Plot 2 — Profit vs Strategy
# ============================================================

q_range = np.linspace(0, 80, 400)

plt.figure(figsize=(6,5))

plt.plot(
    q_range,
    [profit_A(qA, qB_ne) for qA in q_range],
    label="Firm A Profit"
)

plt.plot(
    q_range,
    [profit_B(qA_ne, qB) for qB in q_range],
    label="Firm B Profit"
)

plt.axvline(qA_ne, linestyle="--", label="Nash Quantity")

plt.title("Profit vs Strategy")
plt.xlabel("Quantity")
plt.ylabel("Profit")

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# Plot 3 — Equilibrium Point
# ============================================================

plt.figure(figsize=(6,5))

plt.scatter(qA_ne, qB_ne, s=120)

plt.title("Equilibrium Point")
plt.xlabel("Firm A Quantity")
plt.ylabel("Firm B Quantity")

plt.grid(True)

plt.show()


# ============================================================
# Plot 4 — Scenario Comparison
# ============================================================

labels = []
piAs = []
piBs = []

for sc in scenarios:

    fn = (
        stackelberg_equilibrium
        if sc["type"] == "stackelberg"
        else nash_equilibrium
    )

    _, _, _, pA_, pB_ = fn(
        sc["a"],
        sc["b"],
        sc["cA"],
        sc["cB"]
    )

    labels.append(sc["name"])
    piAs.append(pA_)
    piBs.append(pB_)

x = np.arange(len(labels))

plt.figure(figsize=(7,5))

plt.bar(x - 0.2, piAs, 0.4, label="Firm A")
plt.bar(x + 0.2, piBs, 0.4, label="Firm B")

plt.xticks(x, labels)

plt.title("Scenario Comparison")
plt.ylabel("Profit")

plt.legend()
plt.grid(True, axis="y")

plt.show()

#Additive one (EXTENSIVE FORM )
# Simple Stackelberg Game Tree
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
plt.title("Stackelberg Game Tree")
# Root node
ax.plot(5, 9, "o")
ax.text(5, 9.3, "Firm A", ha="center")
# Firm A choices
choices_A = [10, 20]
for i, qA in enumerate(choices_A):
    xA = 3 + i * 4
    # Arrow from Firm A
    ax.annotate(
        "",
        xy=(xA, 7),
        xytext=(5, 8.7),
        arrowprops=dict(arrowstyle="->")
    )
    ax.text((5 + xA)/2, 8, f"qA={qA}", fontsize=8)
    # Firm B node
    ax.plot(xA, 7, "o")
    ax.text(xA, 7.3, "Firm B", ha="center", fontsize=8)
    # Firm B choices
    choices_B = [10, 20]
    for j, qB in enumerate(choices_B):
        xB = xA - 1 + j * 2
        ax.annotate(
            "",
            xy=(xB, 5),
            xytext=(xA, 6.7),
            arrowprops=dict(arrowstyle="->")
        )
        ax.text(xB, 5.2, f"qB={qB}", fontsize=7, ha="center")
        # Terminal node
        ax.plot(xB, 4.5, "s")
        pA = round(profit_A(qA, qB), 0)
        pB = round(profit_B(qA, qB), 0)
        ax.text(xB, 4.0, f"A={pA}", fontsize=7, ha="center")
        ax.text(xB, 3.6, f"B={pB}", fontsize=7, ha="center")
plt.tight_layout()
plt.show()