"""
Dashboard Streamlit — EuroMillions ML
Analyse exploratoire + résultats des modèles
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

sys.path.insert(0, str(Path(__file__).parent / "src"))

# ── Chemins ──────────────────────────────────────────────────────────────────
FIGURES_DIR = Path("outputs/figures")
RESULTS_DIR = Path("outputs/results")
DATA_PATH   = Path("data/raw/euromillions.csv")

# ── Config page ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EuroMillions ML",
    page_icon="🎰",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎰 EuroMillions ML — Démonstration des limites du Machine Learning")
st.caption(
    "**Objectif pédagogique :** montrer qu'aucun modèle ML ne fait mieux qu'un tirage "
    "aléatoire face à un processus stochastique certifié."
)
st.info(
    "⚠️ Ce projet est un **exercice d'apprentissage**, pas un outil de prédiction. "
    "L'EuroMillions utilise un générateur aléatoire certifié — il est mathématiquement "
    "impossible de prédire les tirages.",
    icon="🔬",
)

# ── Onglets ───────────────────────────────────────────────────────────────────
tab_eda, tab_models, tab_predict = st.tabs([
    "📊 Analyse exploratoire",
    "🤖 Modèles & Prédictions",
    "🔮 Prédictions vs Réalité",
])


# ════════════════════════════════════════════════════════════════════════════
# ONGLET 1 — Analyse exploratoire
# ════════════════════════════════════════════════════════════════════════════
with tab_eda:

    # Résumé des données
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, parse_dates=["date"])
        col1, col2, col3 = st.columns(3)
        col1.metric("Tirages analysés", f"{len(df):,}")
        col2.metric("Premier tirage", df["date"].min().strftime("%d/%m/%Y"))
        col3.metric("Dernier tirage", df["date"].max().strftime("%d/%m/%Y"))
    else:
        st.warning("Données non trouvées. Lance `python src/collect.py` d'abord.")

    st.divider()

    # Test chi²
    chi2_path = RESULTS_DIR / "chi2_test.csv"
    if chi2_path.exists():
        st.subheader("Test chi² d'uniformité")
        st.markdown(
            "Le **test du chi²** mesure si la distribution observée des tirages s'écarte "
            "significativement d'une distribution parfaitement uniforme. "
            "C'est le test statistique de référence pour répondre à la question : "
            "**« certains numéros sortent-ils vraiment plus souvent que d'autres ? »**"
        )

        chi2 = pd.read_csv(chi2_path)
        chi2["p_value"] = chi2["p_value"].map(lambda x: f"{x:.4f}")

        def style_conclusion(val):
            color = "#1a7a1a" if val == "uniforme" else "#b00020"
            return f"color: {color}; font-weight: bold"

        st.dataframe(
            chi2.style.map(style_conclusion, subset=["conclusion"]),
            use_container_width=True,
            hide_index=True,
        )

        col_a, col_b = st.columns(2)
        col_a.markdown(
            "**Comment lire ce tableau ?**\n\n"
            "- **chi²** : mesure l'écart total entre fréquences observées et attendues. "
            "Plus il est élevé, plus les numéros semblent inégalement répartis.\n"
            "- **p-value** : probabilité d'obtenir un écart au moins aussi grand *par pur hasard*, "
            "en supposant que le tirage est uniforme.\n"
            "- **Seuil classique** : si p-value > 0.05, on n'a pas de preuve suffisante pour "
            "rejeter l'uniformité."
        )
        col_b.markdown(
            "**Ce que les résultats disent ici :**\n\n"
            "- Numéros (1-50) : p = 0.55 → **aucun biais détecté**. "
            "Les 50 numéros sortent avec une probabilité indiscernable de 1/10 par tirage.\n"
            "- Étoiles (1-12) : p = 0.20 → **aucun biais détecté**. "
            "Le filtre post-2016 a bien éliminé l'artefact lié au changement de règles.\n\n"
            "**Conséquence directe :** un modèle ML qui chercherait à exploiter des "
            "déséquilibres de fréquence parierait sur du bruit statistique pur."
        )
    else:
        st.info("Lance `python src/analyse.py` pour générer le test chi².")

    st.divider()

    # ── Fréquences d'apparition ───────────────────────────────────────────────
    st.subheader("① Fréquences d'apparition")
    st.markdown(
        "Ces histogrammes montrent combien de fois chaque numéro (1-50) et chaque étoile (1-12) "
        "est sorti depuis septembre 2016. La **ligne rouge** représente la fréquence théorique "
        "attendue si le tirage est parfaitement aléatoire."
    )

    col_n, col_e = st.columns(2)
    freq_n = FIGURES_DIR / "freq_numeros.png"
    freq_e = FIGURES_DIR / "freq_etoiles.png"

    if freq_n.exists():
        col_n.image(str(freq_n), use_container_width=True)
        col_n.markdown(
            "**Ce qu'on observe :** les barres fluctuent autour de la ligne rouge sans jamais "
            "s'en éloigner de façon systématique. Aucun numéro ne ressort significativement "
            "plus ou moins souvent que les autres sur la durée.\n\n"
            "**Ce que ça implique pour le ML :** il n'existe pas de numéro « chaud » ou "
            "« froid » fiable — la notion de numéro favori est une illusion statistique."
        )
    else:
        col_n.info("Lance `python src/analyse.py` pour générer ce graphique.")

    if freq_e.exists():
        col_e.image(str(freq_e), use_container_width=True)
        col_e.markdown(
            "**Ce qu'on observe :** même constat pour les étoiles. Sur les données "
            "post-2016 (format unifié à 12 étoiles), les fréquences convergent bien "
            "vers l'attendu théorique.\n\n"
            "**Rappel :** avant septembre 2016 il n'y avait que 11 étoiles — c'est "
            "pourquoi on a filtré ces données pour éviter un biais artificiel."
        )
    else:
        col_e.info("Lance `python src/analyse.py` pour générer ce graphique.")

    st.divider()

    # ── Heatmap des écarts ────────────────────────────────────────────────────
    st.subheader("② Heatmap des écarts")
    st.markdown(
        "Chaque cellule indique depuis combien de tirages un numéro (axe Y) n'est pas sorti, "
        "sur les **100 derniers tirages** (axe X). Plus la couleur est foncée, plus le numéro "
        "est « absent » depuis longtemps."
    )
    ecarts = FIGURES_DIR / "ecarts_heatmap.png"
    if ecarts.exists():
        st.image(str(ecarts), use_container_width=True)
        col_a, col_b = st.columns(2)
        col_a.markdown(
            "**Ce qu'on observe :** les zones sombres (long écart) apparaissent et disparaissent "
            "de façon imprévisible, sans motif vertical ni diagonal récurrent. "
            "Il n'y a pas de \"cycle\" de retour pour un numéro absent."
        )
        col_b.markdown(
            "**Ce que ça implique pour le ML :** un modèle entraîné à prédire le prochain tirage "
            "à partir des écarts ne trouvera aucune règle généralisable — les patterns visibles "
            "dans le train set ne se répètent pas dans le test set."
        )
    else:
        st.info("Lance `python src/analyse.py` pour générer ce graphique.")

    st.divider()

    # ── Évolution temporelle ──────────────────────────────────────────────────
    st.subheader("③ Convergence vers l'équiprobabilité dans le temps")
    st.markdown(
        "Ce graphique suit la **fréquence cumulée** de 5 numéros choisis arbitrairement "
        "depuis le début de la période analysée. La **ligne noire pointillée** est la "
        "valeur théorique attendue : 5 numéros tirés parmi 50, soit 5/50 = **0.10** (10 %)."
    )
    evol = FIGURES_DIR / "evolution_temps.png"
    if evol.exists():
        st.image(str(evol), use_container_width=True)
        col_a, col_b = st.columns(2)
        col_a.markdown(
            "**Ce qu'on observe :** toutes les courbes partent de valeurs très variables "
            "(peu de tirages = forte variance) puis convergent progressivement vers 0.10 "
            "au fil du temps. C'est la **loi des grands nombres** en action."
        )
        col_b.markdown(
            "**Ce que ça implique pour le ML :** les écarts à la moyenne qu'on observe "
            "sur une courte fenêtre (et que le modèle pourrait « apprendre ») sont de "
            "simples fluctuations aléatoires — ils ne persistent pas et ne sont pas "
            "prédictibles. Plus la fenêtre est courte, plus le bruit domine."
        )
    else:
        st.info("Lance `python src/analyse.py` pour générer ce graphique.")


# ════════════════════════════════════════════════════════════════════════════
# ONGLET 2 — Modèles & Prédictions
# ════════════════════════════════════════════════════════════════════════════
with tab_models:

    eval_path = RESULTS_DIR / "evaluation.csv"
    comp_path = FIGURES_DIR / "comparaison_modeles.png"

    if not eval_path.exists():
        # ── Pas encore entraîné ───────────────────────────────────────────
        st.subheader("Les modèles ne sont pas encore entraînés")
        st.markdown(
            "Lance le pipeline pour entraîner les modèles et voir les résultats ici :\n"
            "```powershell\n"
            "python run_pipeline.py --skip-collect --skip-analyse\n"
            "```"
        )

        st.divider()
        st.subheader("Ce que tu verras après l'entraînement")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Modèles comparés**")
            st.markdown("""
- 🎲 **Baseline aléatoire** — tirage uniforme (référence)
- 🌲 **Random Forest** — fenêtre glissante 10 tirages
- 🧠 **LSTM** (PyTorch) — séquence temporelle
- 🔥 **Fréquence hot** — numéros les plus sortis
- ❄️ **Fréquence cold** — numéros les moins sortis
""")
        with col2:
            st.markdown("**Métriques**")
            st.markdown("""
- Hits moyens par tirage (numéros + étoiles corrects)
- Comparaison statistique : t-test de Welch vs baseline
- p-value : significativité de la différence
- Distribution des hits (violinplot)
""")

        st.divider()
        st.subheader("Résultat attendu")
        st.markdown("""
| Modèle | Hits moyens / tirage | p-value vs baseline |
|---|---|---|
| Baseline aléatoire | ~0.83 | — |
| Random Forest | ~0.83 | > 0.05 *(non significatif)* |
| LSTM | ~0.83 | > 0.05 *(non significatif)* |
| Fréquence hot | ~0.83 | > 0.05 *(non significatif)* |
| Fréquence cold | ~0.83 | > 0.05 *(non significatif)* |
""")
        st.info(
            "**Conclusion attendue :** aucun modèle ne batte la baseline. "
            "p-value > 0.05 sur tous les modèles = absence de signal exploitable.",
            icon="📐",
        )

    else:
        # ── Résultats disponibles ─────────────────────────────────────────
        df_eval = pd.read_csv(eval_path)
        baseline_hits = df_eval.loc[df_eval["modèle"] == "baseline", "hits_moyen"].values[0]
        all_ns = all(~df_eval["significatif (p<0.05)"])

        LABELS = {
            "baseline":  "🎲 Baseline aléatoire",
            "rf":        "🌲 Random Forest",
            "lstm":      "🧠 LSTM",
            "freq_hot":  "🔥 Fréquence hot",
            "freq_cold": "❄️ Fréquence cold",
        }
        DESCRIPTIONS = {
            "baseline":  "Tirage uniforme sans mémoire — notre référence théorique",
            "rf":        "Fenêtre glissante 10 tirages · 200 arbres",
            "lstm":      "Séquence temporelle · 2 couches · PyTorch",
            "freq_hot":  "Sélectionne les numéros historiquement les plus sortis",
            "freq_cold": "Sélectionne les numéros historiquement les moins sortis",
        }

        # ── Conclusion en bannière ────────────────────────────────────────
        if all_ns:
            st.success(
                "**Résultat confirmé — aucun modèle ne bat le hasard.** "
                "Toutes les p-values sont > 0.05 : la différence entre chaque modèle "
                "et la baseline aléatoire n'est pas statistiquement significative.",
                icon="🎓",
            )
        else:
            st.warning(
                "⚠️ Certains modèles semblent battre la baseline — "
                "vérifie l'absence de data leakage.",
                icon="⚠️",
            )

        st.divider()

        # ── Métriques comparatives ────────────────────────────────────────
        st.subheader("① Hits moyens par tirage")
        st.markdown(
            "Un **« hit »** = un numéro ou une étoile correctement prédit pour un tirage donné. "
            "Maximum possible : 7 (5 numéros + 2 étoiles). "
            "La baseline théorique analytique est **0.833** — calculée comme "
            "5×(5/50) + 2×(2/12)."
        )

        cols = st.columns(len(df_eval))
        for col, (_, row) in zip(cols, df_eval.iterrows()):
            nom = row["modèle"]
            delta = row["hits_moyen"] - baseline_hits
            delta_str = f"{delta:+.3f} vs baseline" if nom != "baseline" else None
            col.metric(
                label=LABELS.get(nom, nom),
                value=f"{row['hits_moyen']:.3f}",
                delta=delta_str,
                delta_color="off",
            )

        st.divider()

        # ── Graphique comparaison ─────────────────────────────────────────
        st.subheader("② Distribution des hits par modèle")
        st.markdown(
            "Le **barplot** (gauche) montre les hits moyens avec l'écart-type. "
            "Le **violinplot** (droite) montre la distribution complète des scores sur les "
            f"{int(df_eval['n_tirages_test'].iloc[0])} tirages du jeu de test."
        )
        if comp_path.exists():
            st.image(str(comp_path), use_container_width=True)
        col_a, col_b = st.columns(2)
        col_a.markdown(
            "**Ce qu'on observe :** toutes les barres sont à la même hauteur, "
            "avec des intervalles d'incertitude qui se chevauchent largement. "
            "Aucun modèle ne se détache visuellement."
        )
        col_b.markdown(
            "**Ce que ça signifie :** la variabilité naturelle d'un tirage (0 à 7 bons numéros) "
            "est tellement grande que les légères différences de moyenne entre modèles "
            "sont noyées dans le bruit — c'est exactement ce que le t-test confirme."
        )

        st.divider()

        # ── Tableau statistique complet ───────────────────────────────────
        st.subheader("③ Tableau statistique complet")
        st.markdown(
            "Le **t-test de Welch** compare la distribution des hits de chaque modèle "
            "à celle de la baseline. Une p-value > 0.05 signifie qu'on ne peut pas "
            "distinguer les performances du modèle d'un simple tirage aléatoire."
        )

        display = df_eval.copy()
        display["modèle"] = display["modèle"].map(lambda x: LABELS.get(x, x))
        display["description"] = df_eval["modèle"].map(lambda x: DESCRIPTIONS.get(x, ""))
        display = display[["modèle", "description", "hits_moyen", "hits_std",
                            "t_stat_vs_baseline", "p_value_vs_baseline",
                            "significatif (p<0.05)", "n_tirages_test"]]
        display.columns = ["Modèle", "Description", "Hits moyen", "Écart-type",
                           "t-stat", "p-value", "Significatif ?", "N tirages test"]

        def style_row(row):
            styles = [""] * len(row)
            if row["Significatif ?"]:
                styles = ["background-color: #ffdddd"] * len(row)
            elif row["Modèle"] == "🎲 Baseline aléatoire":
                styles = ["background-color: #f0f0f0"] * len(row)
            return styles

        def color_sig(val):
            if val is True:
                return "color: #b00020; font-weight: bold"
            return "color: #1a7a1a; font-weight: bold"

        st.dataframe(
            display.style
                .apply(style_row, axis=1)
                .map(color_sig, subset=["Significatif ?"]),
            use_container_width=True,
            hide_index=True,
        )

        col_a, col_b = st.columns(2)
        col_a.markdown(
            "**Comment lire ce tableau ?**\n\n"
            "- **Hits moyen** : nombre moyen de bons numéros prédits par tirage (sur 7 max)\n"
            "- **t-stat** : plus il est proche de 0, plus le modèle ressemble à la baseline\n"
            "- **p-value** : probabilité d'observer cet écart par pur hasard\n"
            "- **Significatif ?** : `False` partout = aucun modèle ne dépasse le hasard"
        )
        col_b.markdown(
            f"**Nos résultats :**\n\n"
            f"- Baseline : **{baseline_hits:.3f}** hits/tirage\n"
            + "".join(
                f"- {LABELS.get(r['modèle'], r['modèle'])} : **{r['hits_moyen']:.3f}** "
                f"(p = {r['p_value_vs_baseline']:.3f})\n"
                for _, r in df_eval[df_eval["modèle"] != "baseline"].iterrows()
            )
        )

        st.divider()

        # ── Conclusion pédagogique ────────────────────────────────────────
        st.subheader("④ Pourquoi le ML échoue ici ?")
        c1, c2 = st.columns(2)
        c1.markdown(
            "**Raisons mathématiques**\n\n"
            "1. **Indépendance des tirages** : le tirage N+1 est statistiquement "
            "indépendant des tirages précédents — il n'y a rien à mémoriser.\n"
            "2. **Distribution uniforme** : confirmée par le test chi² — "
            "tous les numéros sont équiprobables.\n"
            "3. **Entropie maximale** : un processus purement aléatoire a une entropie "
            "maximale, ce qui signifie que toute compression de l'information (= tout modèle) "
            "est impossible."
        )
        c2.markdown(
            "**Raisons pratiques**\n\n"
            "1. **Le LSTM ne trouve aucune séquence** : la loss stagne à ~0.349 dès "
            "l'epoch 5 — il apprend uniquement la fréquence marginale de chaque bit.\n"
            "2. **Le Random Forest overfit le bruit** : les patterns appris sur le train "
            "set sont des corrélations spurieuses qui ne se généralisent pas.\n"
            "3. **Hot/Cold = gambler's fallacy** : croire qu'un numéro « doit » sortir "
            "parce qu'il est absent depuis longtemps est un biais cognitif classique — "
            "les données le confirment ici empiriquement."
        )


# ════════════════════════════════════════════════════════════════════════════
# ONGLET 3 — Prédictions vs Réalité
# ════════════════════════════════════════════════════════════════════════════
with tab_predict:
    from datetime import date, timedelta
    from predict_next import generate_predictions, load_log_with_results, LOG_PATH
    from scheduler import (
        get_next_draw_date, get_prev_draw_date,
        should_generate, should_fetch, LOG_PATH as SCHED_LOG,
    )

    models_ready = (RESULTS_DIR / "rf_model.pkl").exists()

    # ── Bandeau statut automatique ────────────────────────────────────────
    today      = date.today()
    next_draw  = get_next_draw_date(today)
    days_until = (next_draw - today).days

    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Prochain tirage", next_draw.strftime("%A %d/%m/%Y").capitalize(), f"dans {days_until} jour(s)")
    if DATA_PATH.exists():
        df_data = pd.read_csv(DATA_PATH, parse_dates=["date"])
        col_s2.metric("Dernier tirage connu", df_data["date"].max().strftime("%d/%m/%Y"))
    if LOG_PATH.exists():
        log_raw = pd.read_csv(LOG_PATH)
        col_s3.metric("Sessions de prédiction", len(log_raw["made_at"].unique()))

    # Statut du jour
    if should_generate(today):
        st.info(
            f"📅 **Aujourd'hui c'est la veille du tirage ({next_draw.strftime('%d/%m/%Y')}).**  \n"
            "Le système doit générer les prédictions — clique sur le bouton ci-dessous "
            "ou laisse le planificateur automatique le faire à 9h.",
            icon="🔮",
        )
    elif should_fetch(today):
        prev_draw = get_prev_draw_date(today)
        st.info(
            f"📥 **Le tirage d'hier ({prev_draw.strftime('%d/%m/%Y')}) vient d'avoir lieu.**  \n"
            "Le système doit récupérer les vrais résultats — clique sur **Rafraîchir les données** "
            "ou laisse le planificateur automatique le faire à 9h.",
            icon="📡",
        )
    else:
        day_names = ["lundi","mardi","mercredi","jeudi","vendredi","samedi","dimanche"]
        st.success(
            f"✅ Pas d'action requise aujourd'hui ({day_names[today.weekday()]}).  \n"
            f"Le planificateur reprendra **{day_names[(next_draw - timedelta(days=1)).weekday()]}** "
            f"pour générer les prédictions du tirage du {next_draw.strftime('%d/%m/%Y')}.",
            icon="🤖",
        )

    st.divider()

    if not models_ready:
        st.warning(
            "Les modèles ne sont pas encore entraînés. "
            "Lance `python run_pipeline.py --skip-collect --skip-analyse` d'abord.",
            icon="⚠️",
        )
    else:
        # ── Boutons manuels ───────────────────────────────────────────────
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            if st.button("🔮 Générer les prédictions", use_container_width=True,
                         type="primary" if should_generate(today) else "secondary"):
                with st.spinner("Génération en cours..."):
                    generate_predictions()
                st.success("Prédictions générées !")
                st.rerun()
        with col_btn2:
            if st.button("📡 Rafraîchir les données", use_container_width=True,
                         type="primary" if should_fetch(today) else "secondary"):
                with st.spinner("Récupération des tirages..."):
                    import sys
                    sys.path.insert(0, "src")
                    from collect import collect
                    collect(force=True)
                st.success("Données mises à jour !")
                st.rerun()
        with col_btn3:
            if st.button("🔄 Tout mettre à jour", use_container_width=True):
                with st.spinner("Mise à jour complète..."):
                    from scheduler import run_daily
                    msg = run_daily(force=True)
                st.success(msg)
                st.rerun()

        st.divider()

        if LOG_PATH.exists():
            log_raw = pd.read_csv(LOG_PATH)
            last_run = log_raw["made_at"].max()
            st.caption(f"Dernière prédiction : **{last_run}**")

        st.divider()

        if not LOG_PATH.exists():
            st.info("Clique sur **Générer les prédictions** pour démarrer le suivi.")
        else:
            df_log = load_log_with_results()

            if df_log.empty:
                st.info("Aucune prédiction enregistrée.")
            else:
                # ── Sélecteur de session de prédiction ───────────────────
                sessions = sorted(df_log["made_at"].unique(), reverse=True)
                session_labels = {s: f"Prédiction du {s}" for s in sessions}
                selected = st.selectbox(
                    "Session de prédiction",
                    options=sessions,
                    format_func=lambda s: session_labels[s],
                )
                df_view = df_log[df_log["made_at"] == selected].copy()
                after_date = df_view["after_date"].iloc[0]
                draw_date  = df_view["draw_date"].iloc[0]

                # ── Statut du tirage ──────────────────────────────────────
                if draw_date is None or str(draw_date) == "None":
                    st.info(
                        f"⏳ Tirage prévu après le **{after_date}** — résultat pas encore disponible. "
                        "Rafraîchis les données une fois le tirage passé.",
                        icon="📅",
                    )
                    result_known = False
                else:
                    actual_nums  = df_view["actual_nums"].iloc[0]
                    actual_stars = df_view["actual_stars"].iloc[0]
                    st.success(
                        f"✅ Tirage du **{draw_date}** — Résultat réel : "
                        f"**{actual_nums}** ★ **{actual_stars}**",
                        icon="🎰",
                    )
                    result_known = True

                st.divider()

                # ── Construction du tableau ───────────────────────────────
                rows = []
                for _, r in df_view.iterrows():
                    pred_nums  = f"{r['n1']:02d} · {r['n2']:02d} · {r['n3']:02d} · {r['n4']:02d} · {r['n5']:02d}"
                    pred_stars = f"{r['etoile1']:02d} · {r['etoile2']:02d}"

                    if result_known:
                        hits_n = int(r["hits_nums"])  if r["hits_nums"]  is not None else 0
                        hits_s = int(r["hits_stars"]) if r["hits_stars"] is not None else 0
                        hits_t = int(r["hits_total"]) if r["hits_total"] is not None else 0
                        row = {
                            "Modèle":           r["label"],
                            "Numéros prédits":  pred_nums,
                            "Étoiles prédites": pred_stars,
                            "Résultat réel — numéros":  r["actual_nums"],
                            "Résultat réel — étoiles":  r["actual_stars"],
                            "✓ Numéros": hits_n,
                            "✓ Étoiles": hits_s,
                            "Total hits": hits_t,
                        }
                    else:
                        row = {
                            "Modèle":           r["label"],
                            "Numéros prédits":  pred_nums,
                            "Étoiles prédites": pred_stars,
                            "Résultat réel — numéros":  "En attente...",
                            "Résultat réel — étoiles":  "En attente...",
                            "✓ Numéros": "—",
                            "✓ Étoiles": "—",
                            "Total hits": "—",
                        }
                    rows.append(row)

                df_table = pd.DataFrame(rows)

                # ── AgGrid ────────────────────────────────────────────────
                gb = GridOptionsBuilder.from_dataframe(df_table)
                gb.configure_default_column(resizable=True, wrapText=True, autoHeight=True)

                # Largeurs
                gb.configure_column("Modèle",           width=190, pinned="left")
                gb.configure_column("Numéros prédits",  width=210)
                gb.configure_column("Étoiles prédites", width=120)
                gb.configure_column("Résultat réel — numéros",  width=210)
                gb.configure_column("Résultat réel — étoiles",  width=130)
                gb.configure_column("✓ Numéros",  width=110)
                gb.configure_column("✓ Étoiles",  width=110)
                gb.configure_column("Total hits", width=110)

                # Coloration des hits si résultat connu
                if result_known:
                    hits_style = JsCode("""
                        function(params) {
                            const v = params.value;
                            if (v === 0) return { 'color': '#b00020', 'fontWeight': 'bold' };
                            if (v === 1) return { 'color': '#e67e00', 'fontWeight': 'bold' };
                            if (v >= 2) return { 'color': '#1a7a1a', 'fontWeight': 'bold' };
                            return {};
                        }
                    """)
                    total_style = JsCode("""
                        function(params) {
                            const v = params.value;
                            if (v === 0) return { 'backgroundColor': '#fff0f0', 'fontWeight': 'bold' };
                            if (v <= 2) return { 'backgroundColor': '#fff8e1', 'fontWeight': 'bold' };
                            if (v >= 3) return { 'backgroundColor': '#e8f5e9', 'fontWeight': 'bold' };
                            return {};
                        }
                    """)
                    gb.configure_column("✓ Numéros",  cellStyle=hits_style)
                    gb.configure_column("✓ Étoiles",  cellStyle=hits_style)
                    gb.configure_column("Total hits", cellStyle=total_style)

                # Ligne baseline en gris
                row_style = JsCode("""
                    function(params) {
                        if (params.data['Modèle'].includes('Baseline')) {
                            return { 'backgroundColor': '#f5f5f5', 'fontStyle': 'italic' };
                        }
                        return {};
                    }
                """)
                gb.configure_grid_options(getRowStyle=row_style)
                gb.configure_grid_options(domLayout="autoHeight")

                grid_opts = gb.build()

                AgGrid(
                    df_table,
                    gridOptions=grid_opts,
                    allow_unsafe_jscode=True,
                    fit_columns_on_grid_load=False,
                    height=280,
                    theme="streamlit",
                )

                # ── Explication pédagogique ───────────────────────────────
                st.divider()
                st.markdown(
                    "**Comment lire ce tableau ?**\n\n"
                    "- Chaque ligne = une grille générée par un modèle différent pour le même tirage\n"
                    "- **✓ Numéros** : combien des 5 numéros prédits correspondent au tirage réel (max 5)\n"
                    "- **✓ Étoiles** : combien des 2 étoiles prédites correspondent au tirage réel (max 2)\n"
                    "- **Total hits** : score final sur 7 — la baseline aléatoire en obtient en moyenne **0.83**\n\n"
                    "La baseline sert de référence : si les modèles ne font pas mieux qu'elle de façon "
                    "systématique, c'est la preuve que les tirages ne sont pas prédictibles."
                )

        st.divider()

        # ── Historique du scheduler ───────────────────────────────────────
        st.subheader("📋 Journal du planificateur automatique")
        st.markdown(
            "Le planificateur Windows (`auto_update.py`) s'exécute chaque jour à 9h et "
            "décide automatiquement de l'action à effectuer selon le calendrier des tirages."
        )

        if SCHED_LOG.exists():
            sched_df = pd.read_csv(SCHED_LOG).sort_values("date", ascending=False)
            ACTION_ICONS = {"predict": "🔮", "fetch": "📡", "idle": "💤"}
            sched_df["action"] = sched_df["action"].map(
                lambda a: f"{ACTION_ICONS.get(a, '')} {a}"
            )
            sched_df.columns = ["Date", "Action", "Détail"]
            st.dataframe(sched_df, use_container_width=True, hide_index=True)
        else:
            col_sched, col_cmd = st.columns(2)
            col_sched.info(
                "Aucune exécution automatique enregistrée pour l'instant.\n\n"
                "Le journal apparaîtra ici une fois le planificateur configuré.",
                icon="📋",
            )
            col_cmd.markdown(
                "**Pour activer le planificateur automatique :**\n\n"
                "```powershell\n"
                "# En tant qu'administrateur :\n"
                ".\\setup_scheduler.ps1\n"
                "```\n\n"
                "**Ou manuellement :**\n"
                "```powershell\n"
                "python auto_update.py\n"
                "```"
            )
