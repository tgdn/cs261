'use strict'

const path = require('path');
const http = require('http');
const express = require('express');
const bodyParser = require('body-parser');
const multer = require('multer');
const r = require('rethinkdb');
const spawn = require('child_process').spawn;

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

const closeDb = () => {
    if (global.db) {
        global.db.close((err) => {
            if (err) {
                throw err
            }
        })
    }
}

const distPath = path.join(__dirname, 'dist')
app.use(express.static(distPath));

app.use(bodyParser.json()); // for parsing application/json
app.use(bodyParser.urlencoded({ extended: true })); // for parsing application/x-www-form-urlencoded

const upload = multer({
    dest: path.join(__dirname, 'uploads'),
    limits: { files: 1 },
})
const uploadHandler = upload.single('file')

app.post('/upload', (req, res) => {
    uploadHandler(req, res, (err) => {
        if (err) {
            res.json({ success: false })
            return
        }

        /* At this point upload is complete, start analysing file:
         * spawn python process that does analysis in the background
         * docs from: nodejs.org/api/child_process.html#child_process_child_process
        */
        // spawn('python', ['../main.py', '-f', req.file.path], {
        //     //detached: true,
        //     stdio: 'inherit'
        // }).on('close', (code) => {
        //     console.log(`process exit code: ${code}`)
        // })

        res.json({
            success: true,
            filename: req.file.originalname,
            size: req.file.size,
        })
    })
})

app.post('/setstream', (req, res) => {
    const streamUrl = req.body.streamUrl
    const port = req.body.port || 80

    if (streamUrl && port) {
        console.log(streamUrl);
        console.log(port);
        /* At this point, start analysing stream:
         * spawn python process that does analysis in the background
        */
        // spawn('python', ['../main.py', '-s', streamUrl, '-p', port], {
        //     stdio: 'inherit'
        // }).on('close', (code) => {
        //     console.log(`process exit code: ${code}`)
        // })

        res.json({
            success: true,
            streamUrl: req.body.streamUrl,
            port: req.body.port
        })
        return
    }
    res.json({
        success: false,
    })
})

app.get(['/', '*'], (req, res) => {
    res.sendFile('index.html', { root: distPath });
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
    /* eslint-disable no-console */
    console.log(`Connection to db could not be established:\n${error}`)
    process.exit(1)
    /* eslint-enable */
})
