import logging
import dill
import pandas as pd
from flask import Flask, request
from werkzeug.exceptions import BadRequest


application = Flask(__name__)

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")

feature_name = ['breed', 'date of last vet visit', 'hair length', 'height',
                'number of vet visits', 'weight']

def load_model(model_path):
    with open(model_path, "rb") as f:
        model = dill.load(f)
    return model


def launch_model(model, request):

    try:
        features = request.json
        features_as_df = pd.io.json.json_normalize(features)
        features_as_df['date of last vet visit'] = pd.to_datetime(features['date of last vet visit'], format="%Y-%m-%d %H:%M:%S")
        features_as_df = features_as_df[feature_name]

        logger.debug("Predicting {}".format(features_as_df))
        predictions = model.predict(features_as_df)

    except Exception as e:
        logger.exception("Error in pipeline")
        print("Error {0}".format(str(e)).encode("utf-8"))
        raise BadRequest(description=e)

    return '\nExpected Live for this cat: '+''.join(str(round(e, 2)) for e in predictions)+'\n\n'


@application.route("/", methods=["POST"])
def handle_request():
    global model
    return launch_model(model, request)


if __name__ == "__main__":
    try:
        model = load_model("./model.pk")
        application.run()
    except KeyboardInterrupt:
        logger.exception("Shutting down")
    except Exception as e:
        logger.exception("Error in initialization chain")