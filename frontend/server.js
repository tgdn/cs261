'use strict'

const express = require('express');
const horizon = require('@horizon/server');
const path = require('path');

const app = express();
const http_server = app.listen(8181);
const options = {
    auto_create_collection: true,
    auto_create_index: true,
    auth: { token_secret: 'bdvU6/Cs+QFMYSvuiB3aTgZiUzcXbc1ca0v7abg4Fq3BP2U5IIvkXp/cBPg/vj6pLfDGxfnG8fmv7KEZgI8KJQ==' }
};
const horizon_server = horizon(http_server, options);

app.use(express.static(__dirname + '/dist'));

app.get('/', function(req, res) {
    res.send(path.join(__dirname + '/index.html'));
})

console.log('Listening on port 8181.');

// horizon_server.add_auth_provider(
//     horizon_instance.auth.facebook,
//     { id: 'id', secret: 'secret', path: 'facebook' }
// );
