#!/bin/bash

#bash

# Using flask development server
#python agora/agora.py deploy -r flask ./files/model.pkl

# Using gunicorn
gunicorn --chdir agora --workers=2 -b 0.0.0.0:5000 "agora:_deploy('../files/model.pkl', True)"