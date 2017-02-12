'use strict'

const path = require('path');
const http = require('http');
const express = require('express');
const r = require('rethinkdb');

const app = express();
const server = http.Server(app);

const io = require('socket.io')(server);
const horizon = require('@horizon/server');

const httpServer = server.listen(8181)

const options = {
    project_name: 'purple',
    auto_create_collection: true,
    auto_create_index: true,
    permissions: false,
    auth: {
        token_secret: 'bdvU6/Cs+QFMYSvuiB3aTgZiUzcXbc1ca0v7abg4Fq3BP2U5IIvkXp/cBPg/vj6pLfDGxfnG8fmv7KEZgI8KJQ==',
        allow_unauthenticated: true,
    }
};

const horizonServer = horizon(httpServer, options);

// below will close connection to db
// const closeDb = () => {
//     if (global.db) {
//         global.db.close((err) => {
//             if (err) {
//                 throw err
//             }
//         })
//     }
// }

app.use(express.static(path.join(__dirname, 'dist')));

app.get('/', (req, res) => {
    res.sendFile('index.html');
})

r.connect({
    host: 'localhost',
    port: 28015,
    db: 'purple'
})
.then((conn) => {
    global.db = conn

    io.on('connection', (socket) => {
        r.table('trades')
            .pluck('price', 'symbol')
            .changes({ squash: 4, includeInitial: true }).run(global.db)
            .then((cursor) => {
                cursor.each((err, item) => {
                    socket.emit('tradechange', item)
                })
            })
    })
})
.error((error) => {
    console.log(`Connection to db could not be established:\n${error}`)
    process.exit(1)
})
