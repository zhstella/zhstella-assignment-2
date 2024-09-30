from flask import Flask, render_template, jsonify, request
import numpy as np
from kmeans import KMeans

app = Flask(__name__)

# Global state for the web application
state = {
    "data": None,
    "kmeans": None,
    "clusters": None,
    "step": 0
}

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate-data', methods=['POST'])
def generate_data():
    # 随机生成 200 个 2D 数据点
    data = np.random.rand(200, 2)
    state["data"] = data.tolist()

    return jsonify({
        "data": state["data"],  # 返回数据点
    })

@app.route('/initialize', methods=['POST'])
def initialize():
    init_method = request.json.get('method', 'random')
    num_clusters = request.json.get('numClusters', 3)
    data = np.array(state["data"])  # 使用之前生成的数据
    kmeans = KMeans(k=num_clusters, init_method=init_method)
    kmeans.initialize_centroids(data)

    state["kmeans"] = kmeans

    return jsonify({
        "centroids": kmeans.centroids.tolist()
    })


@app.route('/step', methods=['POST'])
def step():
    kmeans = state.get("kmeans")
    if kmeans:
        clusters, converged = kmeans.iterate()  # Returns clusters and convergence status
        state["clusters"] = clusters.tolist()
        return jsonify({
            "clusters": state["clusters"],
            "centroids": kmeans.centroids.tolist(),
            "converged": converged  # Send convergence status
        })
    return jsonify({"error": "KMeans not initialized"}), 400
    
@app.route('/run-to-end', methods=['POST'])
def run_to_end():
    kmeans = state.get("kmeans")
    if kmeans:
        converged = False
        while not converged:  # 迭代直到收敛
            clusters, converged = kmeans.iterate()

        state["clusters"] = clusters.tolist()

        return jsonify({
            "clusters": state["clusters"],
            "centroids": kmeans.centroids.tolist(),
            "converged": converged
        })
    return jsonify({"error": "KMeans not initialized"}), 400

@app.route('/initialize-manual', methods=['POST'])
def initialize_manual():
    data = np.array(state["data"])  # 获取之前生成的数据点
    manual_centroids = request.json.get('centroids', [])  # 获取手动选择的质心
    num_clusters = request.json.get('numClusters', 3)

    if len(manual_centroids) != num_clusters:
        return jsonify({"error": "Number of centroids does not match the number of clusters."}), 400

    kmeans = KMeans(k=num_clusters)
    kmeans.centroids = np.array(manual_centroids)  # 使用手动选择的质心
    kmeans.data = data  # 确保将数据点传递给 KMeans 实例

    state["kmeans"] = kmeans  # 将 KMeans 实例保存到 state

    return jsonify({
        "data": state["data"],
        "centroids": kmeans.centroids.tolist()
    })



if __name__ == '__main__':
    app.run(debug=True, port=3000)
