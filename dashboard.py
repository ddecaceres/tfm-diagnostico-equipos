"""
Dashboard de diagnóstico de dinámicas comunicativas de equipos
TFM · Máster Business Intelligence y Data Analytics — UCM

Principio de arquitectura (decisión cerrada del proyecto): este dashboard
SOLO LEE parquets, nunca calcula. Todo el cálculo ocurre en los notebooks
01-09. Aquí solo filtramos, agregamos visualmente y presentamos.

Unidad de análisis: GRUPO-SEMANA (o GRUPO para rasgos estructurales fijos,
Fase 9). Nunca se muestran métricas de personas individuales ni rankings
de personas (coherencia con RGPD / AI Act).

Cómo ejecutar:
    streamlit run dashboard.py
Los parquets deben estar en ./data/ (ver DATA_DIR más abajo).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuración y rutas
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Diagnóstico de dinámicas comunicativas",
    page_icon="📌",
    layout="wide",
)

DATA_DIR = Path(__file__).parent / "data"

# Paleta de color proporcionada por el usuario
PALETA = ["#82413e", "#c58269", "#a48b64", "#498555", "#70adbb", "#363356", "#f74194"]

st.markdown(f"""
<style>
    html, body, [class*="css"] {{ font-size: 17px; }}
    h1 {{ color: {PALETA[5]}; font-size: 2.1em; }}
    h2, h3 {{ color: {PALETA[5]}; }}
    .stApp {{ background-color: #faf7f2; }}
    section[data-testid="stSidebar"] {{ background-color: {PALETA[5]}; }}
    section[data-testid="stSidebar"] * {{ color: #f5f0ea !important; }}
    div[data-testid="stMetricValue"] {{ color: {PALETA[0]}; font-size: 1.6em; }}
    div[data-testid="stMetricLabel"] {{ font-size: 1.0em; }}
</style>
""", unsafe_allow_html=True)

# Las 5 dimensiones y su métrica representativa para radar/score (SPEC 3).
DIMENSIONES = {
    "Participación": "entropia_norm",
    "Fluidez": "tasa_respuesta",
    "Coordinación": "ratio_accion",
    "Sinergia": "cohesion_semantica",   # U-invertida, se transforma aparte
    "Riesgo (invertido)": "indice_riesgo",
}

# Explicaciones en lenguaje llano, pensadas para un técnico de RR.HH. sin
# formación en la literatura científica del proyecto.
LEYENDAS_DIMENSIONES = {
    "Participación": (
        "**¿Habla todo el mundo o solo unos pocos?** Mide si los mensajes "
        "de la semana están repartidos entre los miembros del equipo o "
        "concentrados en dos o tres personas. Un valor alto es buena "
        "señal: indica una conversación más equilibrada."
    ),
    "Fluidez": (
        "**¿Se contestan entre ellos?** Mide qué proporción de mensajes "
        "que necesitaban respuesta la obtuvieron en menos de 72 horas. "
        "Un valor alto indica un equipo receptivo; un valor bajo puede "
        "señalar sobrecarga o desconexión."
    ),
    "Coordinación": (
        "**¿Se decide o solo se informa?** Mide qué proporción de los "
        "mensajes son peticiones, decisiones o compromisos, frente a "
        "simple información de cortesía. Un valor alto sugiere un equipo "
        "orientado a la acción."
    ),
    "Sinergia": (
        "**¿De qué hablan, ¿de lo mismo o de cosas distintas?** Ojo con "
        "esta: **no es \"cuanto más alto, mejor\"**. Un valor muy alto "
        "puede indicar que el equipo repite lo mismo sin avanzar "
        "(redundancia); un valor muy bajo, que cada uno va por su lado sin "
        "conectar temas. Lo saludable está en un punto intermedio."
    ),
    "Riesgo (invertido)": (
        "**¿Hay señales de alarma esta semana?** Combina cuatro alertas: "
        "retrasos largos en responder, aislamiento de alguien del equipo, "
        "caída brusca de actividad, y concentración excesiva en una sola "
        "persona. Aquí lo mostramos invertido (más lleno el radar = menos "
        "riesgo) para que se lea igual que el resto de ejes."
    ),
}

ETIQUETAS_HUMANAS = {
    5: "Asuntos Regulatorios California",
    60: "Contratos Documentación Gas",
    4: "Legal Crédito EnronOnline",
    48: "Trading Gas Ventas",
    47: "Crédito Garantías ISDA",
    21: "Gasoducto Transwestern",
    53: "Proyecto Netco GTV",
    72: "Back Office Confirmaciones",
    2: "Desarrollo Proyectos Generación",
    3: "Medición Deals Gas",
}

NOMBRES_CLUSTER = {
    0: "Núcleos temáticos estables",
    1: "Participación repartida, en exploración ⚠️",
    2: "Deliberativo, poco resolutivo",
    3: "Concentración jerárquica de riesgo alto",
    4: "Semanas de ritmo pausado y resolutivo",
}


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

@st.cache_data
def cargar_parquet(nombre):
    ruta = DATA_DIR / nombre
    if not ruta.exists():
        return None
    return pd.read_parquet(ruta)


def primera_columna_presente(df, candidatas):
    for c in candidatas:
        if c in df.columns:
            return c
    return None


features = cargar_parquet("features_weekly.parquet")
clusters = cargar_parquet("clusters.parquet")
grupos = cargar_parquet("grupos.parquet")
estructura = cargar_parquet("estructura_grupos.parquet")
mapa_departamentos = cargar_parquet("mapa_departamentos.parquet")

if features is None:
    st.error(
        "No encuentro `data/features_weekly.parquet`. Copia los parquets "
        "necesarios a la carpeta `data/` junto a este script."
    )
    st.stop()

COL_GRUPO = primera_columna_presente(features, ["grupo_de", "grupo_id"])
COL_SEMANA = primera_columna_presente(features, ["anyo_semana", "semana"])

if clusters is not None:
    col_grupo_cl = primera_columna_presente(clusters, ["grupo_de", "grupo_id"])
    col_semana_cl = primera_columna_presente(clusters, ["anyo_semana", "semana"])
    col_cluster = primera_columna_presente(clusters, ["cluster", "cluster_id", "cluster_label", "labels"])
    if col_grupo_cl and col_semana_cl:
        features = features.merge(
            clusters[[col_grupo_cl, col_semana_cl] + ([col_cluster] if col_cluster else [])],
            left_on=[COL_GRUPO, COL_SEMANA], right_on=[col_grupo_cl, col_semana_cl],
            how="left", suffixes=("", "_cl"),
        )
        if col_cluster and col_cluster != "cluster":
            features = features.rename(columns={col_cluster: "cluster"})

etiquetas_por_grupo = dict(ETIQUETAS_HUMANAS)
if grupos is not None:
    col_grupo_g = primera_columna_presente(grupos, ["grupo_id", "grupo_de"])
    col_etq = primera_columna_presente(grupos, ["etiqueta_humana", "etiqueta", "nombre_grupo"])
    if col_grupo_g and col_etq:
        for _, fila in grupos[[col_grupo_g, col_etq]].drop_duplicates().iterrows():
            if pd.notna(fila[col_etq]):
                etiquetas_por_grupo[int(fila[col_grupo_g])] = fila[col_etq]


def etiqueta_de(grupo_id):
    try:
        gid = int(grupo_id)
    except (TypeError, ValueError):
        return str(grupo_id)
    return etiquetas_por_grupo.get(gid, f"Grupo {gid}")


def color_de_grupo(grupo_id):
    try:
        gid = int(grupo_id)
    except (TypeError, ValueError):
        gid = abs(hash(grupo_id))
    return PALETA[gid % len(PALETA)]


# ---------------------------------------------------------------------------
# Transformación de Sinergia (U-invertida) para radar y scoring
# ---------------------------------------------------------------------------

def transformar_valor(col, v):
    """Devuelve un valor en [0,1] donde 'más alto siempre es mejor',
    para poder combinarlo de forma justa en radares y en el score."""
    if pd.isna(v):
        return 0.0
    if col == "indice_riesgo":
        return float(np.clip(1 - (v / 4), 0, 1))
    if col == "cohesion_semantica":
        # U-invertida: lo saludable es el punto medio (0.5), no los extremos.
        return float(np.clip(1 - abs(v - 0.5) * 2, 0, 1))
    return float(np.clip(v, 0, 1))


def score_compuesto(fila):
    valores = [transformar_valor(col, fila.get(col, np.nan)) for col in DIMENSIONES.values()]
    return float(np.mean(valores)) * 100


if not features.empty:
    features["score_compuesto"] = features.apply(score_compuesto, axis=1)


def radar_dimensiones(fila, titulo):
    etiquetas = list(DIMENSIONES.keys())
    valores = [transformar_valor(col, fila.get(col, np.nan)) for col in DIMENSIONES.values()]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=valores + [valores[0]], theta=etiquetas + [etiquetas[0]],
        fill="toself", name=titulo,
        line=dict(color=PALETA[5]), fillcolor=PALETA[5], opacity=0.75,
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False, title=titulo, margin=dict(t=60, b=20),
    )
    return fig


def mostrar_leyendas():
    with st.container(border=True):
        st.markdown("**¿Qué significa cada eje del radar?**")
        cols = st.columns(len(LEYENDAS_DIMENSIONES))
        for col, (nombre, texto) in zip(cols, LEYENDAS_DIMENSIONES.items()):
            with col:
                st.markdown(f"*{nombre}*")
                st.caption(texto)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("📌 Diagnóstico de equipos")
st.sidebar.caption(
    "Unidad de análisis: **grupo-semana** (o grupo, para rasgos "
    "estructurales). Nunca se evalúan personas individuales (RGPD / AI Act)."
)

paginas = ["Portada · Mapa de departamentos", "Vista general", "Evolución temporal",
           "Perfiles (clusters)", "Comparativa de grupos"]
pagina = st.sidebar.radio("Página", paginas)

grupos_disponibles = sorted(features[COL_GRUPO].dropna().unique())
opciones_grupo = {etiqueta_de(g): g for g in grupos_disponibles}

st.sidebar.markdown("---")
grupo_sel_label_sidebar = st.sidebar.selectbox(
    "Ir directo a un equipo", ["(elige uno)"] + list(opciones_grupo.keys())
)


# ---------------------------------------------------------------------------
# Página: Portada · Mapa de departamentos (chinchetas)
# ---------------------------------------------------------------------------

if pagina == "Portada · Mapa de departamentos":
    st.title("Mapa de la comunicación en la empresa")
    st.caption(
        "Cada chincheta es un equipo. Su POSICIÓN es lo que importa: los "
        "equipos con más conexiones e intercambio quedan hacia el centro; "
        "los que dependen sobre todo de una sola relación quedan más "
        "alejados, cerca de esa relación concreta. La intensidad del hilo "
        "es una señal secundaria de afinidad relativa entre cada par."
    )

    if estructura is None or mapa_departamentos is None:
        st.warning(
            "No encuentro `estructura_grupos.parquet` o "
            "`mapa_departamentos.parquet` en `data/`. Ejecuta primero "
            "`09_rrhh_estructura.ipynb` y copia sus salidas aquí."
        )
    else:
        col_grupo_est = primera_columna_presente(estructura, ["grupo_id", "grupo_de"])
        col_arq = primera_columna_presente(estructura, ["arquetipo"])
        col_posx = primera_columna_presente(estructura, ["pos_x"])
        col_posy = primera_columna_presente(estructura, ["pos_y"])

        pos = {
            int(row[col_grupo_est]): (row[col_posx], row[col_posy])
            for _, row in estructura.iterrows()
            if pd.notna(row[col_posx]) and pd.notna(row[col_posy])
        }

        mapa_visible = mapa_departamentos[mapa_departamentos.get("mostrar_en_mapa", True) == True].copy()

        fig = go.Figure()

        max_afinidad = mapa_visible["afinidad"].max() if not mapa_visible.empty and "afinidad" in mapa_visible else 1
        for _, fila in mapa_visible.iterrows():
            try:
                a, b = int(fila["grupo_a"]), int(fila["grupo_b"])
            except (ValueError, TypeError):
                continue
            if a not in pos or b not in pos:
                continue
            afinidad = fila.get("afinidad", 0)
            opacidad = float(np.clip(afinidad / max_afinidad if max_afinidad else 0, 0.05, 0.75))
            fig.add_trace(go.Scatter(
                x=[pos[a][0], pos[b][0]], y=[pos[a][1], pos[b][1]],
                mode="lines",
                line=dict(width=1.6, color=PALETA[5]),
                opacity=opacidad,
                hoverinfo="skip", showlegend=False,
            ))

        xs, ys = [], []
        for grupo_id, (x, y) in pos.items():
            xs.append(x); ys.append(y)

        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text",
            marker=dict(size=28, color=[color_de_grupo(g) for g in pos.keys()], line=dict(width=2, color="white")),
            text=[etiqueta_de(g) for g in pos.keys()],
            textposition="bottom center",
            textfont=dict(size=14),
            hovertext=[
                f"{etiqueta_de(g)}<br>Perfil comunicativo día: "
                f"{estructura.loc[estructura[col_grupo_est] == g, col_arq].values[0] if col_arq else '—'}"
                for g in pos.keys()
            ],
            hoverinfo="text", showlegend=False,
        ))

        fig.update_layout(
            xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
            yaxis=dict(visible=False),
            height=650, margin=dict(t=20, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Perfil comunicativo día")
        st.caption(
            "Cómo está construido cada equipo por dentro, de forma "
            "permanente (no cambia semana a semana): quién concentra la "
            "comunicación, y si el equipo tiene un comunicador clave del "
            "que depende estructuralmente."
        )
        if col_arq:
            tabla = estructura[[col_grupo_est, col_arq]].copy()
            tabla["Equipo"] = tabla[col_grupo_est].apply(etiqueta_de)
            col_frag = primera_columna_presente(estructura, ["fragmentacion_sin_hub"])
            if col_frag:
                tabla["Comunicador clave"] = estructura[col_frag].apply(
                    lambda v: "⚠️ Sí, existe" if v and v > 0 else "No detectado"
                )
            st.dataframe(
                tabla[["Equipo", col_arq] + (["Comunicador clave"] if col_frag else [])]
                .rename(columns={col_arq: "Perfil comunicativo día"}),
                use_container_width=True, hide_index=True,
            )


# ---------------------------------------------------------------------------
# Página: Vista general
# ---------------------------------------------------------------------------

elif pagina == "Vista general":
    st.title("Comunicación de equipo - semana")

    col1, col2 = st.columns(2)
    with col1:
        default_idx = (
            list(opciones_grupo.keys()).index(grupo_sel_label_sidebar) + 1
            if grupo_sel_label_sidebar in opciones_grupo else 0
        )
        grupo_sel_label = st.selectbox("Seleccionar equipo", list(opciones_grupo.keys()), index=max(default_idx - 1, 0))
        grupo_sel = opciones_grupo[grupo_sel_label]

    df_g = features[features[COL_GRUPO] == grupo_sel].sort_values(COL_SEMANA)
    semanas_disp = df_g[COL_SEMANA].tolist()

    with col2:
        semana_sel = st.selectbox("Seleccionar semana", semanas_disp, index=len(semanas_disp) - 1 if semanas_disp else 0)

    if not semanas_disp:
        st.warning("Este equipo no tiene semanas con datos válidos.")
    else:
        fila = df_g[df_g[COL_SEMANA] == semana_sel].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Emails esa semana", int(fila.get("n_emails", 0)))
        c2.metric("Miembros activos", int(fila.get("n_miembros_activos", 0)))
        if "cluster" in fila and pd.notna(fila.get("cluster")):
            c3.metric("Perfil comunicativo semana", NOMBRES_CLUSTER.get(int(fila["cluster"]), f"Cluster {int(fila['cluster'])}"))

        # Scoring doble: ranking 1-100 (1 = más óptimo) vs empresa y vs
        # histórico propio del grupo. Partimos del percentil (más alto =
        # mejor) y lo invertimos a un ranking donde 1 es la mejor posición.
        score_actual = fila.get("score_compuesto", np.nan)
        if pd.notna(score_actual):
            percentil_empresa = (features["score_compuesto"] < score_actual).mean() * 100
            hist_propio = df_g["score_compuesto"]
            percentil_propio = (hist_propio < score_actual).mean() * 100 if len(hist_propio) > 1 else 50.0
            ranking_empresa = int(np.clip(round((1 - percentil_empresa / 100) * 99) + 1, 1, 100))
            ranking_propio = int(np.clip(round((1 - percentil_propio / 100) * 99) + 1, 1, 100))

            c4.metric("Valor de comunicación", f"{score_actual:.0f}/100")

            cc1, cc2 = st.columns(2)
            cc1.metric("Ranking empresa (1-100)", ranking_empresa)
            cc2.metric("Ranking histórico (1-100)", ranking_propio)
            st.caption(
                "El valor de comunicación combina las 5 dimensiones "
                "(Sinergia transformada para que el punto medio puntúe más "
                "que los extremos). En los rankings, **1 es la mejor "
                "posición posible** y 100 la peor — 'Ranking empresa' "
                "compara esta semana contra todas las semanas de todos los "
                "equipos; 'Ranking histórico' la compara solo contra el "
                "propio historial de este equipo."
            )

        st.plotly_chart(radar_dimensiones(fila, f"{grupo_sel_label} — {semana_sel}"), use_container_width=True)
        mostrar_leyendas()

        with st.expander("Ver todas las métricas de esta fila"):
            st.dataframe(fila.astype(str).to_frame("valor"))


# ---------------------------------------------------------------------------
# Página: Evolución temporal (con trayectoria como vectores)
# ---------------------------------------------------------------------------

elif pagina == "Evolución temporal":
    st.title("Evolución temporal de un equipo")
    st.caption(
        "Un mismo equipo puede pasar por distintos perfiles en semanas "
        "distintas — el clustering describe estados semanales, no "
        "identidades fijas de equipo."
    )

    grupo_sel_label = st.selectbox("Equipo", list(opciones_grupo.keys()), key="evol_grupo")
    grupo_sel = opciones_grupo[grupo_sel_label]
    df_g = features[features[COL_GRUPO] == grupo_sel].sort_values(COL_SEMANA)

    if df_g.empty:
        st.warning("Sin datos para este equipo.")
    else:
        st.subheader("Trayectoria como vectores")
        st.caption(
            "Cada punto es una semana; la línea conecta las semanas en "
            "orden cronológico y la flecha final muestra hacia dónde se "
            "mueve el equipo ahora mismo."
        )
        eje_x_nombre = st.selectbox("Eje X", list(DIMENSIONES.keys()), index=0, key="eje_x")
        eje_y_nombre = st.selectbox("Eje Y", list(DIMENSIONES.keys()), index=4, key="eje_y")
        col_x, col_y = DIMENSIONES[eje_x_nombre], DIMENSIONES[eje_y_nombre]

        xs = [transformar_valor(col_x, v) for v in df_g[col_x]]
        ys = [transformar_valor(col_y, v) for v in df_g[col_y]]

        fig_tray = go.Figure()
        fig_tray.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers",
            line=dict(color=PALETA[5], width=2),
            marker=dict(size=9, color=list(range(len(xs))), colorscale=[[0, PALETA[1]], [1, PALETA[5]]]),
            text=df_g[COL_SEMANA].tolist(), hoverinfo="text",
        ))
        if len(xs) >= 2:
            fig_tray.add_annotation(
                x=xs[-1], y=ys[-1], ax=xs[-2], ay=ys[-2],
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=2,
                arrowcolor=PALETA[6],
            )
        fig_tray.update_layout(
            xaxis_title=f"{eje_x_nombre} →", yaxis_title=f"{eje_y_nombre} →",
            xaxis=dict(range=[-0.05, 1.05]), yaxis=dict(range=[-0.05, 1.05]),
            height=450,
        )
        st.plotly_chart(fig_tray, use_container_width=True)

        st.subheader("Métricas semana a semana")
        NOMBRE_AMIGABLE = {v: k for k, v in DIMENSIONES.items()}
        metricas_disponibles = [c for c in DIMENSIONES.values() if c in df_g.columns]
        metricas_sel = st.multiselect(
            "Métricas a mostrar", metricas_disponibles,
            default=metricas_disponibles[:3],
            format_func=lambda c: NOMBRE_AMIGABLE.get(c, c),
        )
        if metricas_sel:
            fig = px.line(
                df_g, x=COL_SEMANA, y=metricas_sel, markers=True,
                title=f"Evolución — {grupo_sel_label}",
                color_discrete_sequence=PALETA,
                labels={"value": "Valor", "variable": "Métrica", COL_SEMANA: "Semana"},
            )
            fig.for_each_trace(lambda t: t.update(name=NOMBRE_AMIGABLE.get(t.name, t.name)))
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df_g, use_container_width=True)


# ---------------------------------------------------------------------------
# Página: Perfiles (clusters)
# ---------------------------------------------------------------------------

elif pagina == "Perfiles (clusters)":
    st.title("Reparto de equipos por perfiles comunicativos y semanas")

    if "cluster" not in features.columns:
        st.warning("No encuentro la columna de cluster. Verifica `clusters.parquet` en `data/`.")
    else:
        conteo = (
            features.dropna(subset=["cluster"])
            .assign(nombre=lambda d: d["cluster"].map(lambda c: NOMBRES_CLUSTER.get(int(c), f"Cluster {int(c)}")))
            .groupby("nombre").size().reset_index(name="n").sort_values("n", ascending=False)
        )
        fig_bar = px.bar(
            conteo, x="nombre", y="n", title="Tamaño de cada perfil comunicativo semana (nº de grupo-semana)",
            labels={"nombre": "Perfil comunicativo semana", "n": "Grupo-semanas"}, color_discrete_sequence=PALETA,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Radar de comportamiento")
        st.caption(
            "Cada línea de color es un perfil comunicativo semana. Cuanto "
            "más se acerca una línea al borde exterior en un eje, más "
            "presente está esa dimensión en las semanas clasificadas con "
            "ese perfil — compara la forma de las líneas entre sí, no solo "
            "su tamaño. El significado de cada eje está explicado justo "
            "debajo del gráfico."
        )
        fig = go.Figure()
        etiquetas = list(DIMENSIONES.keys())
        for i, c in enumerate(sorted(features["cluster"].dropna().unique())):
            sub = features[features["cluster"] == c]
            valores = [transformar_valor(col, sub[col].mean()) for col in DIMENSIONES.values()]
            fig.add_trace(go.Scatterpolar(
                r=valores + [valores[0]], theta=etiquetas + [etiquetas[0]], fill="toself",
                name=NOMBRES_CLUSTER.get(int(c), f"Cluster {int(c)}"),
                line=dict(color=PALETA[i % len(PALETA)]),
            ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
        st.plotly_chart(fig, use_container_width=True)
        mostrar_leyendas()

        st.subheader("Lista de equipos por perfil")
        cluster_sel = st.selectbox(
            "Elige un perfil comunicativo semana", sorted(features["cluster"].dropna().unique()),
            format_func=lambda c: NOMBRES_CLUSTER.get(int(c), f"Cluster {int(c)}"),
        )
        grupos_en_cluster = features[features["cluster"] == cluster_sel][COL_GRUPO].dropna().unique()
        st.write(f"{len(grupos_en_cluster)} equipos distintos han tenido al menos una semana con este perfil:")
        st.write(", ".join(etiqueta_de(g) for g in sorted(grupos_en_cluster)))


# ---------------------------------------------------------------------------
# Página: Comparativa de grupos
# ---------------------------------------------------------------------------

elif pagina == "Comparativa de grupos":
    st.title("Comparativa entre equipos")
    st.caption("Compara hasta 3 equipos usando la media de sus métricas.")

    grupos_sel_labels = st.multiselect(
        "Elige entre 2 y 3 equipos", list(opciones_grupo.keys()),
        default=list(opciones_grupo.keys())[: min(2, len(opciones_grupo))], max_selections=3,
    )

    if len(grupos_sel_labels) < 2:
        st.info("Selecciona al menos 2 equipos para comparar.")
    else:
        fig = go.Figure()
        etiquetas = list(DIMENSIONES.keys())
        tabla_resumen = {}
        for i, label in enumerate(grupos_sel_labels):
            gid = opciones_grupo[label]
            sub = features[features[COL_GRUPO] == gid]
            medias = sub[list(DIMENSIONES.values())].mean()
            tabla_resumen[label] = medias
            valores = [transformar_valor(col, medias.get(col, np.nan)) for col in DIMENSIONES.values()]
            fig.add_trace(go.Scatterpolar(
                r=valores + [valores[0]], theta=etiquetas + [etiquetas[0]], fill="toself",
                name=label, line=dict(color=PALETA[i % len(PALETA)]),
            ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
        st.plotly_chart(fig, use_container_width=True)
        mostrar_leyendas()

        st.subheader("Tabla comparativa (medias)")
        st.dataframe(pd.DataFrame(tabla_resumen).T, use_container_width=True)


st.sidebar.markdown("---")
st.sidebar.caption(
    "El sistema ofrece diagnóstico descriptivo — señala dónde mirar, no "
    "dónde va a ocurrir un problema. La interpretación es humana."
)
