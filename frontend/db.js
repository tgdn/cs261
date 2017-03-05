
const pgp = require('pg-promise')({})

const dbConfig = {
    host: 'localhost',
    port: 5432,
    database: 'cs261',
    user: 'purple',
    password: 'purple123'
};
const db = pgp(dbConfig);

const handleException = (err, res, reason) => {
    console.error(err) // eslint-disable-line
    res.status(500)
        .json({ success: false, reason })
}

const getSymbols = (req, res) => {
    db.any('SELECT * FROM symbols ORDER BY name')
    .then((symbols) => {
        res.status(200)
            .json({
                success: true,
                symbols
            })
    })
    .catch(err => handleException(err, res))
}

const getSymbol = (req, res) => {
    const symbol = req.params.symbol || null
    if (symbol !== null) {
        // Get latest 1000 trades
        // Nest query to reorder trades by datetime ASC
        db.any(
            `SELECT * FROM (
                SELECT id, price, size, flagged, datetime
                FROM trades WHERE symbol_name = $1
                ORDER BY datetime DESC
                LIMIT 1000
            ) AS derivedTable ORDER BY datetime ASC`, symbol)
        .then((trades) => {
            res.status(200)
                .json({
                    success: true,
                    trades
                })
        })
        .catch(err => handleException(err, res))
    }
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
    getSymbol,
    getTrade
}
