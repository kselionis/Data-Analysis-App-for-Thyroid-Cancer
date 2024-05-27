import streamlit as st
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans,SpectralClustering

st.title("Data Analysis App for Thyroid Cancer")

tabs = ["Επισκόπηση Δεδομένων", "Μηχανική Μάθηση", "2D Απεικόνιση", "Πληροφορίες"]
selected_tab = st.radio("Επιλογή Καρτέλας", tabs)

if selected_tab == "Επισκόπηση Δεδομένων":
    uploaded_file = st.file_uploader("Ανεβάστε ένα αρχείο σε μορφή CSV,TXT (no_header)", type=["csv", "txt"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, header=None)

        st.write("Επισκόπηση Δεδομένων:")
        st.write(data)

        categorical_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        encoder = OneHotEncoder(sparse_output=False)
        encoded_cols = pd.DataFrame(encoder.fit_transform(data[categorical_cols]))
        data = pd.concat([data.drop(categorical_cols, axis=1), encoded_cols], axis=1)

elif selected_tab == "Μηχανική Μάθηση":
      uploaded_file = st.file_uploader("Ανεβάστε ένα αρχείο σε μορφή CSV,TXT (no_header)", type=["csv", "txt"])
    
      if uploaded_file is not None:
          data = pd.read_csv(uploaded_file, header=None)

          st.write("Επισκόπηση Δεδομένων:")
          st.write(data)

          categorical_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
          encoder = OneHotEncoder(sparse_output=False)
          encoded_cols = pd.DataFrame(encoder.fit_transform(data[categorical_cols]))
          data = pd.concat([data.drop(categorical_cols, axis=1), encoded_cols], axis=1)

          st.write("Επιλέξτε αλγόριθμο Μηχανικής Μάθησης:")
          method = st.radio("Αλγόριθμοι", ("Αλγόριθμοι Ομαδοποίησης", "Αλγόριθμοι Κατηγοριοποίησης"))

          if method == "Αλγόριθμοι Ομαδοποίησης":
             st.write("Εκπαίδευση Αλγόριθμου Ομαδοποίησης")

             X = data.iloc[:, :-1]
             y = data.iloc[:, -1]

             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

             knn = KNeighborsClassifier()
             knn.fit(X_train, y_train)
        
             logistic_model = LogisticRegression()
             logistic_model.fit(X_train, y_train)

             knn_y_pred = knn.predict(X_test)

             logistic_y_pred = logistic_model.predict(X_test)

             knn_accuracy = accuracy_score(y_test, knn_y_pred)

             logistic_accuracy = accuracy_score(y_test, logistic_y_pred)

             st.write(f"Η ακρίβεια του ταξινομητή KNeighbours είναι: {knn_accuracy}")
             st.write(f"Η ακρίβεια της Λογιστικής Παλινδρόμησης είναι: {logistic_accuracy}")
             
          elif method == "Αλγόριθμοι Κατηγοριοποίησης":
              clustering_algorithm = st.selectbox("Επέλεξε Αλγόριθμο Κατηγοριοποίησης", ("K-Means", "Spectral Clustering"))

              if clustering_algorithm == "K-Means":
                  kmeans = KMeans(n_clusters=2, random_state=0)
                  kmeans.fit(data)
                  labels = kmeans.labels_
                  st.write("Cluster Labels:")
                  st.write(labels)

              elif clustering_algorithm == "Spectral Clustering":
                  spectral_clustering = SpectralClustering(n_clusters=2, random_state=0)
                  spectral_clustering.fit(data)
                  labels = spectral_clustering.labels_
                  st.write("Cluster Labels:")
                  st.write(labels)

elif selected_tab == "2D Απεικόνιση":
    uploaded_file = st.file_uploader("Ανεβάστε ένα αρχείο σε μορφή CSV,TXT (no_header)", type=["csv", "txt"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, header=None)

        st.write("Επισκόπηση Δεδομένων:")
        st.write(data)

        categorical_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        encoder = OneHotEncoder(sparse_output=False)
        encoded_cols = pd.DataFrame(encoder.fit_transform(data[categorical_cols]))
        data = pd.concat([data.drop(categorical_cols, axis=1), encoded_cols], axis=1)

        st.write("Επιλέξτε Αλγόριθμο Μείωσης Διάστασης:")
        reduction_algorithm = st.selectbox("Μείωση Διαστάσεων", ("PCA", "t-SNE"))

        if reduction_algorithm == "PCA":
            pca = PCA(n_components=2)
            X_reduced = pca.fit_transform(data)
            plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
            st.pyplot()
        
        elif reduction_algorithm == "t-SNE":
            tsne = TSNE(n_components=2)
            X_reduced = tsne.fit_transform(data)
            plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
            st.pyplot()
        
        st.write("Διαγράμματα διερευνητικής ανάλυσης δεδομένων (EDA):")
        st.set_option('deprecation.showPyplotGlobalUse', False)

elif selected_tab == "Πληροφορίες":
    st.write("## Σχετικά με την εφαρμογή")
    st.write("Αυτή η εφαρμογή έχει σχεδιαστεί για την ανάλυση δεδομένων που σχετίζονται με τον καρκίνο του θυρεοειδούς. Επιτρέπει στους χρήστες να μεταφορτώνουν τα σύνολα δεδομένων τους, να εκτελούν ομαδοποίηση και ταξινόμηση χρησιμοποιώντας διάφορους αλγορίθμους και να απεικονίζουν τα αποτελέσματα χρησιμοποιώντας τεχνικές μείωσης της διαστατικότητας")

    st.write("## Πως λειτουργεί?")
    st.write("1. **Ανασκόπηση δεδομένων**: Ανεβάστε το σύνολο δεδομένων σας και δείτε το περιεχόμενό του.")
    st.write("2. **Μηχανική Μάθηση**: Επιλογή και εφαρμογή διαφορετικών αλγορίθμων μηχανικής μάθησης για ομαδοποίηση και ταξινόμηση.")
    st.write("3. **Οπτική απεικόνιση 2D**: Οπτικοποίηση των δεδομένων με χρήση PCA ή t-SNE για μείωση της διαστατικότητας.")

    st.write("## Ομάδα Ανάπτυξης")
    st.write("Αυτή η εφαρμογή αναπτύχθηκε από μια ομάδα φοιτητών του τμήματος Πληροφορικής του Ιονίου Πανεπιστημίου:")
    st.write("- **Σελιώνης Κωνσταντίνος**: Ανάπτυξη της ενσωμάτωσης του αλγορίθμου μηχανικής μάθησης και δημιουργία των χαρακτηριστικών απεικόνισης 2D.")
    st.write("- **Μουρίκης Ιωσήφ**: Σχεδίασε τη διεπαφή χρήστη και χειρίστηκε τα διαγράμματα EDA.")
    st.write("- **Τζούνης Παναγιώτης**: Υλοποιήθηκε η λειτουργία μεταφόρτωσης και επισκόπησης δεδομένων.")

    st.write("## Συγκεκριμένα Καθήκοντα")
    st.write("- **Σελιώνης Κωνσταντίνος**: Ενσωμάτωση του K-means, Φασματική ομαδοποίηση και Εφαρμογή αλγορίθμων ταξινόμησης και μετρικών ακρίβειας.")
    st.write("- **Μουρίκης Ιωσήφ**: Σχεδιασμός και διάταξη της εφαρμογής, δημιουργία της καρτέλας Info.")
    st.write("- **Τζούνης Παναγιώτης**: Μεταφόρτωση και επισκόπηση δεδομένων, ενσωμάτωση PCA και t-SNE.")
