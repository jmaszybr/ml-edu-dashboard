import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="🔵 KMeans Explorer", page_icon="🔵", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f4ff; }
    h1, h2, h3 { color: #5b2d8e; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: #5b2d8e !important; }
    .info-box {
        background: #ede7f6; border-left: 4px solid #7b3db5;
        border-radius: 0 10px 10px 0; padding: 0.8rem 1rem;
        color: #3d1a6e; font-size: 0.92rem; margin: 0.8rem 0;
    }
    .step-box {
        background: white; border-radius: 12px;
        padding: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        text-align: center; margin-bottom: 0.5rem;
    }
    .step-num { font-size: 2rem; font-weight: 700; color: #7b3db5; }
    .step-label { font-size: 0.8rem; color: #888; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

# ── Paleta kolorów ────────────────────────────────────────────────────────────
KOLORY = ["#e63946", "#2a9d8f", "#e9c46a", "#457b9d", "#f4a261", "#a8dadc"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔵 KMeans Explorer")
    st.markdown("Interaktywna nauka algorytmu K-Means")
    st.divider()

    st.markdown("### 📐 Dane")
    typ_danych = st.selectbox("Kształt danych:", [
        "Kuliste skupiska (blobs)",
        "Dwa półksiężyce (moons)",
        "Koncentryczne okręgi (circles)",
        "Własne dane (losowe)"
    ])

    n_punktow = st.slider("Liczba punktów:", 50, 500, 200, 50)

    if typ_danych == "Kuliste skupiska (blobs)":
        n_true = st.slider("Prawdziwe skupiska:", 2, 6, 3)
        szum = st.slider("Rozrzut skupisk:", 0.3, 2.5, 1.0, 0.1)
    else:
        n_true = None
        szum = st.slider("Szum:", 0.0, 0.3, 0.05, 0.01)

    st.divider()
    st.markdown("### ⚙️ KMeans")
    k = st.slider("Liczba klastrów K:", 2, 8, 3)
    max_iter = st.slider("Max iteracji:", 1, 20, 10)
    n_init = st.selectbox("Inicjalizacja centroidów:", ["k-means++", "random"])

    st.divider()
    st.markdown("### 🎬 Animacja kroków")
    pokaz_kroki = st.checkbox("Pokaż kroki algorytmu", value=True)
    krok = st.slider("Krok:", 1, max_iter, max_iter) if pokaz_kroki else max_iter

    losuj = st.button("🎲 Losuj nowe dane", use_container_width=True)

# ── Generowanie danych ────────────────────────────────────────────────────────
@st.cache_data
def generuj_dane(typ, n, szum, n_true, seed):
    np.random.seed(seed)
    if typ == "Kuliste skupiska (blobs)":
        X, y = make_blobs(n_samples=n, centers=n_true, cluster_std=szum, random_state=seed)
    elif typ == "Dwa półksiężyce (moons)":
        X, y = make_moons(n_samples=n, noise=szum, random_state=seed)
    elif typ == "Koncentryczne okręgi (circles)":
        X, y = make_circles(n_samples=n, noise=szum, factor=0.5, random_state=seed)
    else:
        X = np.random.randn(n, 2) * 2
        y = np.zeros(n, dtype=int)
    return StandardScaler().fit_transform(X), y

seed = np.random.randint(0, 9999) if losuj else 42
X, y_true = generuj_dane(typ_danych, n_punktow, szum,
                          n_true if n_true else 3, seed)

# ── KMeans krok po kroku ──────────────────────────────────────────────────────
def kmeans_steps(X, k, max_iter, init):
    np.random.seed(42)
    if init == "k-means++":
        idx = [np.random.randint(len(X))]
        for _ in range(k - 1):
            dists = np.min([np.sum((X - X[i])**2, axis=1) for i in idx], axis=0)
            probs = dists / dists.sum()
            idx.append(np.random.choice(len(X), p=probs))
        centroids = X[idx].copy()
    else:
        centroids = X[np.random.choice(len(X), k, replace=False)].copy()

    history = [centroids.copy()]
    labels_history = []

    for _ in range(max_iter):
        dists = np.array([np.sum((X - c)**2, axis=1) for c in centroids])
        labels = np.argmin(dists, axis=0)
        labels_history.append(labels.copy())
        new_centroids = np.array([X[labels == j].mean(axis=0)
                                   if (labels == j).any() else centroids[j]
                                   for j in range(k)])
        centroids = new_centroids
        history.append(centroids.copy())

    return history, labels_history

history, labels_history = kmeans_steps(X, k, max_iter, n_init)
krok_idx = min(krok, len(labels_history)) - 1
labels = labels_history[krok_idx]
centroids = history[krok_idx + 1]

# ── Metryki ───────────────────────────────────────────────────────────────────
inertia = sum(np.sum((X[labels == j] - centroids[j])**2)
              for j in range(k) if (labels == j).any())
sil = silhouette_score(X, labels) if len(set(labels)) > 1 else 0

# ── Nagłówek ──────────────────────────────────────────────────────────────────
st.title("🔵 KMeans Explorer — Nauka Klastrowania")
st.markdown("Obserwuj jak algorytm K-Means **krok po kroku** grupuje punkty w przestrzeni 2D.")
st.divider()

# ── Metryki górne ─────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("📍 Punktów", n_punktow)
m2.metric("🔵 Klastrów K", k)
m3.metric("📉 Inercja", f"{inertia:.1f}")
m4.metric("🏆 Silhouette", f"{sil:.3f}")

st.divider()

# ── Zakładki ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎬 Animacja kroków", "📊 Metoda łokcia", "🏆 Silhouette", "📖 Teoria"
])

# ══ TAB 1 — Animacja ══════════════════════════════════════════════════════════
with tab1:
    st.subheader(f"Krok {krok_idx + 1} z {max_iter}")

    col_plot, col_info = st.columns([3, 1])

    with col_plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor("#f8f4ff")
        ax.set_facecolor("#fdfbff")

        # Punkty
        for j in range(k):
            maska = labels == j
            ax.scatter(X[maska, 0], X[maska, 1],
                       c=KOLORY[j % len(KOLORY)], alpha=0.6, s=50,
                       edgecolors="white", linewidths=0.4, label=f"Klaster {j+1}")

        # Centroidy poprzednie
        prev_c = history[krok_idx]
        ax.scatter(prev_c[:, 0], prev_c[:, 1],
                   c="white", s=200, zorder=4,
                   edgecolors="#5b2d8e", linewidths=2, marker="o", alpha=0.5)

        # Centroidy aktualne
        ax.scatter(centroids[:, 0], centroids[:, 1],
                   c=[KOLORY[j % len(KOLORY)] for j in range(k)],
                   s=300, zorder=5, edgecolors="black", linewidths=2,
                   marker="*", label="Centroidy")

        # Strzałki ruchu centroidów
        for j in range(k):
            dx = centroids[j, 0] - prev_c[j, 0]
            dy = centroids[j, 1] - prev_c[j, 1]
            if abs(dx) + abs(dy) > 0.01:
                ax.annotate("", xy=centroids[j], xytext=prev_c[j],
                            arrowprops=dict(arrowstyle="->", color="#5b2d8e",
                                           lw=1.5, alpha=0.7))

        ax.set_title(f"K-Means — krok {krok_idx+1}/{max_iter}  |  K={k}",
                     fontsize=13, fontweight="bold", color="#5b2d8e")
        ax.legend(fontsize=9, loc="upper right")
        ax.spines[["top","right"]].set_visible(False)
        ax.set_xlabel("X₁", fontsize=11)
        ax.set_ylabel("X₂", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)

    with col_info:
        st.markdown("### 📋 Co się dzieje?")
        st.markdown(f"""
        <div class="step-box">
            <div class="step-num">{krok_idx+1}</div>
            <div class="step-label">Aktualny krok</div>
        </div>
        <div class="step-box">
            <div class="step-num">{k}</div>
            <div class="step-label">Klastrów</div>
        </div>
        <div class="step-box">
            <div class="step-num">{inertia:.0f}</div>
            <div class="step-label">Inercja</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
        ⭐ <b>Gwiazda</b> = centroid klastra<br><br>
        ➡️ <b>Strzałka</b> = ruch centroidu<br><br>
        🎨 <b>Kolor</b> = przynależność do klastra
        </div>
        """, unsafe_allow_html=True)

        # Rozmiary klastrów
        st.markdown("**Rozmiary klastrów:**")
        for j in range(k):
            n_j = int((labels == j).sum())
            pct = n_j / len(X) * 100
            st.markdown(f"Klaster {j+1}: **{n_j}** ({pct:.0f}%)")

    # Wykres inercji w czasie
    st.markdown("#### 📉 Inercja przez kolejne kroki")
    inercje = []
    for step_i, (lbl, cen) in enumerate(zip(labels_history, history[1:])):
        iner = sum(np.sum((X[lbl == j] - cen[j])**2)
                   for j in range(k) if (lbl == j).any())
        inercje.append(iner)

    fig2, ax2 = plt.subplots(figsize=(10, 2.5))
    fig2.patch.set_facecolor("#f8f4ff")
    ax2.set_facecolor("#fdfbff")
    ax2.plot(range(1, len(inercje)+1), inercje, "o-",
             color="#7b3db5", linewidth=2, markersize=7)
    ax2.axvline(krok_idx+1, color="#e63946", linestyle="--",
                linewidth=1.5, label=f"Krok {krok_idx+1}")
    ax2.scatter([krok_idx+1], [inercje[krok_idx]],
                color="#e63946", s=120, zorder=5)
    ax2.set_xlabel("Krok", fontsize=10)
    ax2.set_ylabel("Inercja", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)

# ══ TAB 2 — Metoda łokcia ══════════════════════════════════════════════════════
with tab2:
    st.subheader("Metoda łokcia — jak wybrać K?")
    st.markdown("Szukamy punktu gdzie inercja przestaje gwałtownie spadać — to nasz optymalny **K**.")

    k_range = range(1, 11)
    inercje_k = []
    for ki in k_range:
        km = KMeans(n_clusters=ki, init="k-means++", n_init=10, random_state=42)
        km.fit(X)
        inercje_k.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#f8f4ff")
    ax.set_facecolor("#fdfbff")
    ax.plot(k_range, inercje_k, "o-", color="#7b3db5", linewidth=2.5,
            markersize=9, markerfacecolor="white", markeredgewidth=2)
    ax.axvline(k, color="#e63946", linestyle="--", linewidth=2,
               label=f"Twój K = {k}")
    ax.scatter([k], [inercje_k[k-1]], color="#e63946", s=180, zorder=5)

    # Annotacja łokcia
    ax.annotate(f"  K={k}\n  Inercja={inercje_k[k-1]:.0f}",
                xy=(k, inercje_k[k-1]),
                xytext=(k+0.5, inercje_k[k-1] + max(inercje_k)*0.05),
                fontsize=10, color="#e63946",
                arrowprops=dict(arrowstyle="->", color="#e63946"))

    ax.set_xlabel("Liczba klastrów K", fontsize=12)
    ax.set_ylabel("Inercja (WCSS)", fontsize=12)
    ax.set_title("Metoda łokcia", fontsize=14, fontweight="bold", color="#5b2d8e")
    ax.legend(fontsize=11)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_xticks(list(k_range))
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown('<div class="info-box">💡 <b>Jak czytać wykres?</b> Szukaj "łokcia" — miejsca gdzie krzywa wyraźnie się zgina. Tam inercja nie spada już tak gwałtownie. To sugerowany K.</div>', unsafe_allow_html=True)

# ══ TAB 3 — Silhouette ════════════════════════════════════════════════════════
with tab3:
    st.subheader("Współczynnik Silhouette — jakość klastrów")
    st.markdown("Wartość od **-1** (źle) do **+1** (idealnie). Im wyżej, tym lepiej dopasowane klastry.")

    k_range2 = range(2, 11)
    sil_scores = []
    for ki in k_range2:
        km = KMeans(n_clusters=ki, init="k-means++", n_init=10, random_state=42)
        lbl = km.fit_predict(X)
        sil_scores.append(silhouette_score(X, lbl))

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#f8f4ff")
    ax.set_facecolor("#fdfbff")

    bars = ax.bar(k_range2, sil_scores,
                  color=[("#e63946" if ki == k else "#7b3db5") for ki in k_range2],
                  alpha=0.8, edgecolor="white", linewidth=1.5)
    ax.axhline(sil_scores[k-2], color="#e63946", linestyle="--",
               alpha=0.5, linewidth=1)

    for bar, val in zip(bars, sil_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, color="#333")

    ax.set_xlabel("Liczba klastrów K", fontsize=12)
    ax.set_ylabel("Silhouette Score", fontsize=12)
    ax.set_title("Silhouette Score dla różnych K", fontsize=14,
                 fontweight="bold", color="#5b2d8e")
    ax.set_xticks(list(k_range2))
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    best_k = list(k_range2)[np.argmax(sil_scores)]
    st.success(f"🏆 Najwyższy Silhouette Score dla **K = {best_k}** ({max(sil_scores):.3f})")
    st.markdown('<div class="info-box">💡 <b>Silhouette Score</b> mierzy jak dobrze punkt pasuje do swojego klastra vs sąsiednich. Najwyższy słupek = sugerowane K.</div>', unsafe_allow_html=True)

# ══ TAB 4 — Teoria ════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📖 Jak działa K-Means?")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Algorytm krok po kroku

        **1. Inicjalizacja**
        Wybierz K losowych punktów jako centroidy startowe.
        Opcja `k-means++` wybiera je mądrzej — dalej od siebie.

        **2. Przypisanie**
        Każdy punkt przypisz do najbliższego centroidu
        (minimalna odległość euklidesowa).

        **3. Aktualizacja**
        Przesuń centroid do środka ciężkości (średniej)
        wszystkich punktów w danym klastrze.

        **4. Powtarzaj**
        Kroki 2-3 aż centroidy przestaną się poruszać
        lub osiągniemy max_iter.
        """)

    with col2:
        st.markdown("""
        ### Wzory matematyczne

        **Odległość euklidesowa:**
        """)
        st.latex(r"d(x, c) = \sqrt{\sum_{i=1}^{n}(x_i - c_i)^2}")

        st.markdown("**Inercja (WCSS):**")
        st.latex(r"J = \sum_{k=1}^{K} \sum_{x \in C_k} ||x - \mu_k||^2")

        st.markdown("**Nowy centroid:**")
        st.latex(r"\mu_k = \frac{1}{|C_k|} \sum_{x \in C_k} x")

        st.markdown("**Silhouette:**")
        st.latex(r"s = \frac{b - a}{\max(a, b)}")
        st.caption("a = średnia odl. do własnego klastra, b = średnia odl. do najbliższego klastra")

    st.divider()
    st.markdown("### ⚠️ Kiedy K-Means zawodzi?")

    c1, c2, c3 = st.columns(3)
    c1.error("❌ **Niekuliste kształty**\nMoons i circles — K-Means nie radzi sobie z niespójnymi kształtami")
    c2.warning("⚠️ **Różne rozmiary klastrów**\nAlgorytm zakłada podobne liczebności klastrów")
    c3.info("💡 **Alternatywy**\nDBSCAN, Agglomerative Clustering, Gaussian Mixture Models")

# ── Stopka ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center style='color:#aaa; font-size:0.85rem'>🔵 KMeans Explorer • Zbudowany w Streamlit • Edukacja Data Science</center>",
    unsafe_allow_html=True
)
