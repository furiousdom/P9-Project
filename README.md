# P9 Project

P9 project is a system for drug discovery.
The intention is to have the user give a specific protein as input and get candidate drugs in the form of small molecules.
The candidates are produced by our algorithm which capitalizes on the idea that similar drugs interact with similar molecules and vice versa.

**Make sure to have a folder named `data` in the root of the project with the next file**
* full_database.xml

The file needs to be obtained through DrugBank, otherwise you won't be able to populate the database.

## Prerequisites

Before you begin, ensure you have met the following requirements:
<!--- These are just example requirements. Add, duplicate or remove as required --->
* You have a **Windows 10** machine. It may work on both **Linux and macOS** but they are not supported.
* You have installed [Git for Windows](https://git-scm.com/)
* You have installed [Anaconda 3](https://www.anaconda.com/), and have it in the path
* You have installed [Node.js](https://nodejs.org/en/) (version >=12.14.1)

## Installation

To install p9, follow these steps:

Windows:

1. Open `cmd` or `PowerShell` and run command:
```
git clone https://github.com/icedandreas/P9-Project.git
```

2. Change to project directory:
```
cd p9
```

## Installing the dependencies
1. Create an Anaconda environment with python version 3.7 by running (replace `env_name` with the name that you want):
```
conda create -n env_name  python=3.7
```

2. Activated the conda enviroment that you just created, if it's not done automatically after creation:
```
conda activate env_name
```

3. Install dependencies (make sure the conda environment created in step 1 is activated)
```
pip install tensorflow==2.3.*
pip install --pre deepchem
conda install -y -c conda-forge rdkit
pip install django django-rest-framework django-cors-headers lxml xmltodict psycopg2
```

## Run the migrations

1. To position the working directory in the `backend/` folder (assuming you are in the same terminal instance from the start) run:
```
cd backend
```

2. Run the migrations
```
python manage.py migrate
```

## Using p9
1. To build and run the backend (server) application (assuming you are in the same terminal instance from the start) you can just run:
```
python manage.py runserver
```

2. To build and run the frontend (client) application open another `cmd` or `PowerShell` instance and position the working directory in the `frontend/` folder under the root folder of the project and run:
```
npm run dev
```


## Contributing to p9
Pull requests are not welcome as this is a university project and as such can't be developed by anybody other than the contributors listed here.

## Contributors

* [@Andreas Hald](https://github.com/icedandreas)
* [@Dominik Tabak](https://github.com/furiousdom)
* Alexandr Dyachenko
* Christian Galasz Nielsen

## License
<!--- If you're not sure which open license to use see https://choosealicense.com/--->

This project is not meant to be used or developed by anybody other than the contributors listed here.
