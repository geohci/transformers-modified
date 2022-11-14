#!/usr/bin/env bash
# setup Cloud VPS instance with initial server etc.

# these can be changed but most other variables should be left alone
APP_LBL='api-endpoint'  # descriptive label for endpoint-related directories
REPO_LBL='ml-article-desc'  # directory where repo code will go
GIT_CLONE_HTTPS='https://github.com/geohci/transformers-modified'  # for `git clone`

ETC_PATH="/etc/${APP_LBL}"  # app config info, scripts, ML models, etc.
SRV_PATH="/srv/${APP_LBL}"  # application resources for serving endpoint
TMP_PATH="/tmp/${APP_LBL}"  # store temporary files created as part of setting up app (cleared with every update)
LOG_PATH="/var/log/uwsgi"  # application log data
LIB_PATH="/var/lib/${APP_LBL}"  # where virtualenv will sit
MOD_PATH="/srv/model-25lang-all/"  # where the model binary is stored

# clean up old versions
rm -rf ${TMP_PATH}
mkdir -p ${TMP_PATH}

git clone ${GIT_CLONE_HTTPS} ${TMP_PATH}/${REPO_LBL}

# reinstall virtualenv
#rm -rf ${LIB_PATH}/p3env
#echo "Setting up virtualenv..."
#python3 -m venv ${LIB_PATH}/p3env
#source ${LIB_PATH}/p3env/bin/activate

#echo "Installing repositories..."
#pip install wheel
#pip install -r ${TMP_PATH}/${REPO_LBL}/versions.txt

echo "Copying configuration files..."
cp -r ${TMP_PATH}/${REPO_LBL}/src/* ${ETC_PATH}
cp ${TMP_PATH}/${REPO_LBL}/config/model.nginx ${ETC_PATH}
cp ${TMP_PATH}/${REPO_LBL}/config/model.service ${ETC_PATH}
cp ${TMP_PATH}/${REPO_LBL}/config/uwsgi.ini ${ETC_PATH}
cp ${TMP_PATH}/${REPO_LBL}/config/flask_config.yaml ${ETC_PATH}
cp ${ETC_PATH}/model.nginx /etc/nginx/sites-available/model
if [[ -f "/etc/nginx/sites-enabled/model" ]]; then
    unlink /etc/nginx/sites-enabled/model
fi
ln -s /etc/nginx/sites-available/model /etc/nginx/sites-enabled/
cp ${ETC_PATH}/model.service /etc/systemd/system/

echo "Enabling and starting services..."
systemctl enable model.service  # uwsgi starts when server starts up
systemctl daemon-reload  # refresh state

systemctl restart model.service  # start up uwsgi
systemctl restart nginx  # start up nginx