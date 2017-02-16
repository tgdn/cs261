# README

Basically the project for CS261

## Installation instructions

I am not aware of how installation works on Microsoft machines,
but the steps are still in the same order (but are not done the same).

### Step 1: Homebrew

Homebrew is the missing package manager for mac.
Just get it by typing this into a terminal.

    /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"


### Step 2: Install Python and deps

Type in your terminal, it will install Python, pip and setuptools:

    brew install python

Now you need to install `virtualenv` (to create virtual environments)

    sudo pip install virtualenv

### Step 3: Rethinkdb

It will install the main binary for our database

    brew install rethinkdb


### Step 4: NodeJS

While our backend is in Python, our frontend runs on a light
`node` app called `expressjs` coupled with a real-time framework
called `horizon`.

    brew install node

You should now be able to have the `npm` command available to you.

## Build: Setting it all up

Go to the project's root (cs261/) and type one after the other:

### Python environment and python dependencies

    virtualenv env
    source env/bin/activate
    pip install --upgade pip
    pip install -r requirements.txt

You should now be able to type

    rethinkdb &
    python main.py

That will launch rethinkdb in the background and will setup the
database. Don't worry if it tells you it needs some input.

### [Database setup](docs/Database.md)

Setup the database [here](docs/Database.md)

### JavaScript bundle and dependencies

    cd frontend/
    npm install --save

Now you can
- Run the server
- Compile the javascript

But you have to do that in two different terminal windows (or tabs). So open up another tab and go to the same directory (cs261/frontend).

Compile the code and watch for changes with:

    npm run build

Don't worry if you see a bunch of errors, they are linting errors
and not syntax.

On the other tab you can start the server with

    node server.js

I guess you can now go and visit `http://127.0.0.1:8181/`

On a third terminal window you can run the python script
and it will simultaneously store the trades on the DB.

    python main.py trades.csv

I wouldn't run that command until the end (it takes about 5 minutes or so to read the whole file and store everything in the db). Ctrl-C when you're tired of it. The server and db are still up, its just the python backend that's off.

Good hacking!
