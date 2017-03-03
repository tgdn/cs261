
const pgp = require('pg-promise')({})

const dbConfig = {
    host: 'localhost',
    port: 5432,
    database: 'cs261',
    user: 'purple',
    password: 'purple123'
};
const db = pgp(dbConfig);

const handleException = (err, res) => {
    console.error(err) // eslint-disable-line
    res.status(500)
        .json({ success: false })
}

const getSymbols = (req, res) => {
    db.many('SELECT * FROM symbols ORDER BY name')
    .then((symbols) => {
        res.status(200)
            .json({
                success: true,
                symbols
            })
    })
    .catch(err => handleException(err, res))
}

const getTrade = (req, res) => {
    const tradeid = parseInt(req.body.tradeid, 10)
    db.one('SELECT * FROM trades WHERE id = $1', tradeid)
    .then((trade) => {
        res.status(200)
            .json({ trade })
    })
    .catch(err => handleException(err, res))
}

module.exports = {
    getSymbols,
    getTrade
}
