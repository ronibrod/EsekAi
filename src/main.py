import os
from flask import Flask
from flask_cors import CORS
from .routes.index import routes

app = Flask(__name__)
# CORS(app)
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "https://localhost:5173", "https://esekpro.lm.r.appspot.com"], "supports_credentials": True}})

app.register_blueprint(routes)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
    # app.run(debug=True, port=5001)

