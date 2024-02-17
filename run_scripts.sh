source setup.sh RELEASE

SCRIPTS_DIR="${PWD}/scripts/"

python ${SCRIPTS_DIR}/device_query.py

LIST=$( ls -1 ${SCRIPTS_DIR}/run_* )

for SCRIPT in ${LIST}
do
    echo " --- RUNNING ${SCRIPT} --- "
    python ${SCRIPT}
done
