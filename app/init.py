from flask import Flask

def create_app():
    app = Flask(__name__)

    # Import and register blueprints here, if any
    # from . import routes
    # app.register_blueprint(routes.bp)

    # You can also configure your app here if needed
    # app.config.from_object('config.Config')

    return app
