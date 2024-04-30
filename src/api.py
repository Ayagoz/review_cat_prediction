import json
import logging
import logging.config
import time
import os
import sys
from typing import List

from model import load_model
from load_data import get_all_tags

import numpy as np
import uvicorn
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from sentence_transformers import SentenceTransformer

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir)))

settings = json.load(open(os.path.join(os.path.dirname(__file__), '../infra/config.json')))

app = FastAPI()
model = load_model(settings['model_path'], load_optimizer=False)
embedding = SentenceTransformer(settings['embedding_model'])
all_tags = get_all_tags(settings['tags_path'])

logging.config.fileConfig(os.path.join(os.path.dirname(__file__), settings['logging_config_path']),
                          disable_existing_loggers=False)

logger = logging.getLogger(__name__)


def get_tags_for_text(text:str) -> List[str]:
    tags = None
    with torch.no_grad():
        embed = embedding.encode(text)
        embed = torch.tensor(embed).float()
        preds = (model(embed[None]) > 0.5).numpy().astype(int)
        idx = np.where(preds[0] == 1)[0]
        tags = set()
        for i in idx:
            tags.add(all_tags[i])
    return tags


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home():
    """
    Human readable home page with helpful info.
    """

    html_tpl = '''
    <html>
        <head><title>{title} (version {version})</title></head>
        <body>
            <h1>{title} (version {version})</h1>
            <p>{description}</p>
            {list}
        </body>
    </html>
    '''

    list_html = '''
        <h2>Documentation</h2>
        <ol>
            <li><a href="docs">Swagger API Docs</a></li>
            <li><a href="redoc">Redoc API Docs</a></li>
        </ol>
    '''

    info = {
        "title": "StandardModel API",
        "description": "StandardModel API",
        "version": "0.0.1",
        "default_model": "standard_model",
    }
    info['list'] = list_html

    return html_tpl.format(**info)


@app.get('/liveness')
async def liveness():
    """
    Returns a 200 to signal that the server is up.
    """
    return {'status': 'live'}


@app.get("/readiness")
async def readiness():
    """
    Returns a 200 if the server is ready for inference.
    """
    return {'status': 'ready'}


@app.get("/predict_tags")
async def predict_tags(text: str) -> List[str]:

    tags = get_tags_for_text(text)

    return tags


if __name__ == "__main__":
    logger.info('starting app!')
    uvicorn.run(app, host="0.0.0.0", port=8777)
