#!/usr/bin/env bash

function setup {
    echo "### main dependencies have been installed  ###"

    echo "Initialising Postgres"
    psql -f database.sql

    # python deps
    virtualenv env
    source env/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt

    # rethinkdb
    echo "### Launching rethinkdb in background ###"
    rethinkdb &

    # frontend, node etc
    cd frontend
    npm install --save

    echo "\n\n"
    echo "###################################"
    echo "### Everything is now installed ###"
    echo "\n\nYou are provided with a JS bundle, you do not need to compile again, but if you wish"
    echo "to compile the javascript and watch for changes type in the frontend/ folder: `npm run build`"

    echo "\n\n\nTO LAUNCH SERVER:"
    echo "in frontend/ type `node server.js` to launch the server"

}

function installosx {
    which -s brew
    if [[ $? != 0 ]] ; then
        # install homebrew
        echo "### installing Homebrew ###"
        /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    else
        echo "### updating homebrew ###"
        brew update
    fi

    # install brew deps
    brew install python virtualenv rethinkdb postgresql node

    # common
    setup
}

function installlinux {
    echo "### Installing on Ubuntu ###"
    sudo apt-get update
    sudo apt-get install python-pip python-dev build-essential postgresql postgresql-contrib nodejs npm
    sudo pip install virtualenv

    source /etc/lsb-release && echo "deb http://download.rethinkdb.com/apt $DISTRIB_CODENAME main" | sudo tee /etc/apt/sources.list.d/rethinkdb.list
    wget -qO- https://download.rethinkdb.com/apt/pubkey.gpg | sudo apt-key add -
    sudo apt-get update
    sudo apt-get install rethinkdb

    setup
}

if [ "$(uname)" == "Darwin" ]; then
    installosx
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    installlinux
else
    echo "You need either Linux or MacOS to run this piece of software"
    echo "\"Here's a nickel, kid. Get yourself a better computer.\""
fi
