from flask import Flask, request, jsonify
import os
import webbrowser

app = Flask(__name__)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/run_script', methods=['POST'])
def run_script():
    try:
        data = request.get_json()
        folder_path = data.get('folder')
        if folder_path and os.path.exists(folder_path):
            # Replace 'your_script.py' with the name of your Python script file
            texty = "python"
            os.system(f'{texty} main.py {folder_path}')

            # Open plotted_map.html in a new tab after script execution
            webbrowser.open_new_tab('plotted_map.html')

            return jsonify({"message": "Python script executed successfully"})
        else:
            return jsonify({"error": "Invalid folder path or folder does not exist"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run()
