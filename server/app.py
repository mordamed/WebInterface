from flask import Flask, request, jsonify
from flask_cors import CORS
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
import pickle
from sklearn.cluster import KMeans

app = Flask(__name__)
CORS(app)

# Définition de la classe de clustering
class ClusteringKMeans:
    def __init__(self, n_clusters=3):
        self.kmeans = KMeans(n_clusters=n_clusters)
        
    def fit(self, X):
        self.kmeans.fit(X)
        return self
        
    def predict(self, X):
        return self.kmeans.predict(X)

# Charger le modèle et éventuellement les données d'entraînement
with open("clustering.pkl", "rb") as f:
    clustering = pickle.load(f)

# Afficher les attributs disponibles
print(vars(clustering))
print(dir(clustering))

# Si l'attribut feature_names_in_ existe
if hasattr(clustering, 'feature_names_in_'):
    print("Les features attendues :", clustering.feature_names_in_)
else:
    print("L'attribut 'feature_names_in_' n'existe pas.")

# Définir la liste des features (doit correspondre à l'entraînement du modèle)
training_features = [
    "AGE", "GENRE", "SMOKER",
    "Insuffisance_Cardiaque", "Arteriopathie", "Syndrome_Apnee_Sommeil",
    "Thrombose_Veineuse_Profonde", "Embolie_Pulmonaire", "Trouble_Rythme_Cardiaque",
    "Depression", "Epilepsie", "Cancer", "Hypertension_Arterielle",
    "Diabete", "Hypercholesterolemie", "Hypertriglyceridemie", "Tabagisme"
]

def create_graph(categories, values):
    plt.figure(figsize=(10, 6))
    plt.bar(categories, values)
    plt.xticks(rotation=45)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def prepare_patient_data(input_data: dict):
    new_row = {}
    genre_mapping = {"M": 0, "F": 1}
    
    for feature in training_features:
        if feature in input_data:
            val = input_data[feature]
            if isinstance(val, list):
                val = val[0]
                
            if feature.upper() == "GENRE":
                try:
                    new_row[feature] = float(genre_mapping[val])
                except KeyError:
                    raise ValueError(f"Valeur inconnue pour GENRE : {val}")
            else:
                try:
                    new_row[feature] = float(val)
                except ValueError:
                    raise ValueError(f"Impossible de convertir {val} pour le champ {feature}")
    return pd.DataFrame(new_row, index=[0])

def create_cluster_graph(cluster_number):
    plt.figure(figsize=(8, 6))
    plt.text(0.5, 0.5, f'Cluster {cluster_number}', 
             horizontalalignment='center',
             verticalalignment='center',
             fontsize=30)
    plt.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')



@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'Aucune donnée reçue'}), 400

        # Analyse par clustering
        patient_data = prepare_patient_data(data)
        predicted_cluster = clustering.predict(patient_data)
        cluster_graph = create_cluster_graph(predicted_cluster[0])
        
        return jsonify({
            'cluster_graph': cluster_graph,
            'cluster': int(predicted_cluster[0])
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
