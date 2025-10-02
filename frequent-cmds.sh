# convert line endings from windows to unix /r/n to /n
dos2unix pull-models.sh

# rebuild and restart docker containers
docker compose down && docker compose up --build

# fix ModuleNotFoundError: No module named 'some_module' by adding current directory to PYTHONPATH
export PYTHONPTH=$PWD

# activate virtual environment in windows
source venv/Scripts/activate