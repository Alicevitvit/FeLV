from flask import Flask, request, jsonify, render_template
import subprocess
from classifier.main import get_prediction

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run-script', methods=['POST'])
def run_script():
    data = request.json
    hgb = data.get('hgb')
    hct = data.get('hct')
    mcv = data.get('mcv')
    rbc = data.get('rbc')

    prediction = get_prediction([float(hgb), float(hct), float(mcv), float(rbc)])

    # result = subprocess.run(
    #     ["python3", "script.py", str(hgb), str(hct), str(mcv), str(rbc)],
    #     capture_output=True, text=True
    # )

    return jsonify({"result": int(prediction)})


if __name__ == '__main__':
    app.run(debug=True)