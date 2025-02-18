import streamlit as st  # ✅ Toujours en premier !
from streamlit_option_menu import option_menu
st.set_page_config(page_title="Creusix", page_icon="🎥", layout="wide")

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 💂️ Chargement des fichiers
chemin_all = "all_in_cleaned.csv"
chemin_aggregated = "aggregated.csv"
chemin_model = "model.pickle"
chemin_logo = "CREUSIX-11-02-2025.png"

def load_data(url):
    return pd.read_csv(url)

def load_model():
    with open(chemin_model, 'rb') as f:
        return pickle.load(f)

df = load_data(chemin_all)  # DF principal pour affichage
aggregatedDf = load_data(chemin_aggregated)  # DF utilisé pour les suggestions
model = load_model()

# Nettoyage & transformations
df = df.drop_duplicates(subset=['frenchTitle', 'genre_imdb', 'averageRating'], keep='first')
df = df.dropna(subset=['runtimeMinutes', 'numVotes', 'averageRating', 'poster_path'])
df['startYear'] = pd.to_datetime(df['startYear'], errors='coerce').dt.year.astype('Int64')
df['runtimeMinutes'] = pd.to_numeric(df['runtimeMinutes'], errors='coerce')

df['search_display'] = df['frenchTitle'] + " (" + df['startYear'].astype(str) + ")"

# 🏠 MENU DE NAVIGATION
with st.sidebar:
    st.image(chemin_logo, width=250)  # Affiche le logo dans la barre latérale

    selection = option_menu(
        menu_title="Menu",        
        options=["Accueil", "Recherche"],  
        icons=["house", "search"],  
        menu_icon="cast",          
        default_index=0            
    )

# ➞ PAGE ACCUEIL
if selection == "Accueil":
    st.title("Accueil")
    st.write("Bienvenue sur Creusix ")
    st.write("Recherchez un film et découvrez des recommandations personnalisées !")
    
    films_random = df[df['poster_path'].notna()].sample(12)  
    colonnes_films = [st.columns(6) for i in range(2)] 

    indice_film = 0  
    for ligne in colonnes_films:  
        for colonne in ligne:  
            if indice_film < len(films_random):  
                film_selectionne = films_random.iloc[indice_film]  # ✅ Récupère le film actuel
                colonne.image(f"https://image.tmdb.org/t/p/w500{film_selectionne['poster_path']}", width=150)
                colonne.write(f"🎬 {film_selectionne['frenchTitle']} ({film_selectionne['startYear']})")
                indice_film += 1

    # ✅ Ajoute une ligne de séparation
    st.markdown("---")


# ➞ PAGE RECHERCHE
elif selection == "Recherche":
    st.title("🔍 Recherche de films : ")
    search_query = st.selectbox("Saisissez le film que vous recherchez :", [""] + list(df['search_display']))

    if search_query != "":
        film_principal = df[df['search_display'] == search_query].iloc[0]
        st.markdown("🎥 Le film que vous avez sélectionné : ")
        col1, col2 = st.columns([1, 2])

        with col1:
            if pd.notna(film_principal['poster_path']):
                st.image(f"https://image.tmdb.org/t/p/w500{film_principal['poster_path']}", width=300)

        with col2:
            st.write(f"🎬 Titre : {film_principal['frenchTitle']} ({film_principal['startYear']})")
            st.write(f"🕑 Durée : {int(film_principal['runtimeMinutes'])} minutes")
            st.write(f"⭐ Note moyenne : {film_principal['averageRating']}")
            st.write(f"💫 Genres : {film_principal['genre_imdb']}")
            st.write(f"🗒️ Synopsis : {film_principal['overview']}")
            imdb_url = f" https://www.imdb.com/title/{film_principal['tconst']}"
            st.write(f"➡️ Pour plus d'informations 🔗 [Voir sur IMDb]({imdb_url})")

        st.markdown("---")

        indices = model.kneighbors(return_distance=False)
        tconst = film_principal['tconst']
        indexMovies = aggregatedDf.loc[aggregatedDf['tconst'] == tconst]

        if not indexMovies.empty:
            indexMovies = indices[indexMovies.index[0]]

            # Récupérer les films similaires
            films_similaires = []
            for indexMovie in indexMovies[:9]:
                tconst_similaire = aggregatedDf['tconst'].iloc[indexMovie]
                film_similaire_df = df[df['tconst'] == tconst_similaire]

                if not film_similaire_df.empty:
                    film_similaire = film_similaire_df.iloc[0]
                    films_similaires.append({
                        "tconst": tconst_similaire,
                        "frenchTitle": film_similaire['frenchTitle'],
                        "distance": indexMovie
                    })

            df_resultats = pd.DataFrame(films_similaires)

            # 🔥 **Ajout du score de similarité avec TF-IDF**
            if not df_resultats.empty:
                titles = [film_principal['frenchTitle']] + df_resultats['frenchTitle'].tolist()
                vectorizer = TfidfVectorizer().fit_transform(titles)
                similarity_matrix = cosine_similarity(vectorizer)

                # Ajout de la similarité à la DataFrame
                df_resultats['similarité'] = similarity_matrix[0][1:]  # Exclure le film de référence
                df_resultats['similarité'] = df_resultats['similarité'].apply(lambda x: 0 if x < 0.25 else x)

                # Trier par score de similarité puis distance
                df_resultats = df_resultats.sort_values(by=["similarité", "distance"], ascending=[False, True])

            # 📌 **Affichage des suggestions sous forme de grille 3x3**
            st.markdown("📌 Les films susceptibles de vous intéresser en fonction de vos recherches : ")
            cols = st.columns(3)

            count = 0
            for _, film_similaire in df_resultats.iterrows():
                film_info = df[df['tconst'] == film_similaire['tconst']]

                if not film_info.empty:
                    film_info = film_info.iloc[0]
                    with cols[count % 3]:
                        if pd.notna(film_info['poster_path']):
                            st.image(f"https://image.tmdb.org/t/p/w500{film_info['poster_path']}", width=150)
                        st.write(f"🎬 {film_info['frenchTitle']} ({film_info['startYear']})")
                        st.write(f"🕑 Durée : {int(film_info['runtimeMinutes'])} minutes")
                        st.write(f"⭐ Note moyenne {film_info['averageRating']}")
                        st.write(f"💫 Genres : {film_info['genre_imdb']}")

                    count += 1

        else:
            st.write("⚠️ Aucune suggestion trouvée pour ce film.")

        st.markdown("---")

# 📌 **Ajout du style CSS**
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
