#!/usr/bin/env bash

function setup {
    echo "### main dependencies have been installed  ###"

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

    echo "### Everything is now installed ###"
    echo "to compile the javascript and watch for changes type in the frontend/ folder: `npm run build`"
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

}

if [ "$(uname)" == "Darwin" ]; then
    installosx
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    installlinux
else
    echo "You need either Linux or MacOS to run this piece of software"
    echo "\"Here's a nickel, kid. Get yourself a better computer.\""
fi
