# File: app.py
from flask import Flask, request, jsonify, send_from_directory
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from flask_cors import CORS

# --- Sample company dataset (same as original, extendable) ---
company_data = {
    'Company': ['Acme Inc.', 'Globex Corporation', 'TechSavvy Solutions', 'DataDynamic', 'EnergySmart'],
    'Industry': ['Technology', 'Manufacturing', 'IT Services', 'Data Analytics', 'Energy'],
    'Skills Needed': [['Python', 'Data Analysis', 'Machine Learning'],
                     ['Project Management', 'CAD', 'Mechanical Engineering'],
                     ['Web Development', 'Cloud Computing', 'Cybersecurity'],
                     ['SQL', 'Statistics', 'Visualization'],
                     ['Renewable Energy', 'Electrical Engineering', 'Energy Efficiency']],
    'Openings': [5, 3, 8, 4, 2]
}

# Build a global set of skills used by companies
ALL_COMPANY_SKILLS = sorted({s for skills in company_data['Skills Needed'] for s in skills})

# Vectorize company skill vectors
company_skill_matrix = []
for skills in company_data['Skills Needed']:
    company_skill_matrix.append([1 if s in skills else 0 for s in ALL_COMPANY_SKILLS])
company_skill_df = pd.DataFrame(company_skill_matrix, columns=ALL_COMPANY_SKILLS)

# Normalize once (companies)
scaler = MinMaxScaler()
company_scaled = scaler.fit_transform(company_skill_df)

# Train KNN on companies
knn = NearestNeighbors(n_neighbors=3)
knn.fit(company_scaled)

app = Flask(__name__, static_folder='static')
CORS(app)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/match', methods=['POST'])
def api_match():
    data = request.get_json()
    # Expecting: {"name": "...", "skills": ["Python","SQL"], "interests": [...], "goals": "..."}
    user_skills = data.get('skills', [])
    # Build user skill vector aligned to ALL_COMPANY_SKILLS
    user_vector = [1 if s.strip() in user_skills else 0 for s in ALL_COMPANY_SKILLS]
    user_scaled = scaler.transform([user_vector])

    distances, indices = knn.kneighbors(user_scaled)
    matches = []
    for idx, dist in zip(indices[0], distances[0]):
        matches.append({
            'company': company_data['Company'][idx],
            'industry': company_data['Industry'][idx],
            'openings': company_data['Openings'][idx],
            'distance': float(dist)
        })

    return jsonify({'matches': matches})

if __name__ == '__main__':
    app.run(debug=True)

