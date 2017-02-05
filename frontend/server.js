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

app.use(express.static(path.join(__dirname, 'dist')));

app.get('/', (req, res) => {
    res.sendFile('index.html');
})
