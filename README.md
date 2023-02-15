# Descartes ML Article Description API 
This repository contains the scripts and code to host an API for the [Descartes ML model](https://arxiv.org/abs/2205.10012)
for generating article descriptions for Wikipedia articles.

## Components

### config
Scripts for setting up a Cloud VPS instance, the necessary web serving architecture, and updating the API with code updates.
For more details about the specific configuration files and their roles, see [the original template](https://github.com/wikimedia/research-api-endpoint-template/blob/master/README.md).

### artdescapi
Code for running the Flask app and model. A few components:
* `transformers`: modified HuggingFace code that runs the underlying Descartes model.
* `utils/utils.py`: utilities for loading in the model and making predictions.
* `wsgi_template.py`: Flask app with code for taking article names, gathering model features, and returning model outputs.

## Setup
This repository assumes two things already are in place:
* A Cloud VPS instance has been created. The current one has 8GB RAM and 8 VCPUs but likely the API can run on 4GB RAM and 4 VCPUs.
* The model binary / config has been uploaded to the server. Currently those dependencies are stored on a [Cinder volume](https://wikitech.wikimedia.org/wiki/Help:Adding_Disk_Space_to_Cloud_VPS_instances#Cinder) under `/srv/model-25lang-all/`.
* You can ssh onto the service once you have the [correct Cloud VPS config](https://wikitech.wikimedia.org/wiki/Help:Accessing_Cloud_VPS_instances) via something like `ssh isaacj@android-machine-generated-desc.recommendation-api.eqiad1.wikimedia.cloud`

Then, the `cloudvps_setup.sh` script can be copied onto the server and run (`sudo. /cloudvps_setup.sh`).
The instance takes about a minute to load in the model. Progress can be checked via `sudo tail -n50 /var/log/uwsgi/uwsgi.log` and the process is complete when uWSGI has spawned a worker.
Code updates can be incorporated by running `release.sh`

## Debugging
* If no responses are working, ssh onto the server and try `systemctl status model` to see the status of the service
* Run `sudo tail -n100 /var/log/uwsgi/uwsgi.log` to see if there are any errors being displayed 

If the whole system is down for some reason, there are a few things to try:
* Restart the server manually via [Horizon](https://wikitech.wikimedia.org/wiki/Help:Horizon_FAQ)
* ssh onto the server and run `sudo /home/isaacj/release.sh` to generally reset things